from .data_format import create_cky_data_input
import psutil
import resource
import numpy as np
import h5py
import nltk
import os
import json
import string
import re

# Node class to backtrack the best tree
class Node:
    def __init__(self, value, child1, child2):
        self.SENT = 0
        self.ATTN = 1
        self.TXT = 2
        self.child_nuclearity = []
        self.value = value
        self.child1 = child1
        self.child2 = child2

    def get_childs(self):
        return self.child1, self.child2
    
    def get_value(self):
        return self.value
    
    def binary_nuc_calc_function(self, span_calc_function, nuc_calc_function, gold_label, child1_nuc=None, child2_nuc=None):
        # MAX
        if span_calc_function == 'max':
            self.value = [(self.child1.value[self.SENT]*self.child1.value[self.ATTN]
                           +self.child2.value[self.SENT]*self.child2.value[self.ATTN])
                          /(self.child1.value[self.ATTN]+self.child2.value[self.ATTN])
                          ,max(self.child1.value[self.ATTN],self.child2.value[self.ATTN]), 
                          (self.child1.value[self.TXT]+" "+self.child2.value[self.TXT])]
        # AVG
        if span_calc_function == 'avg':
            self.value = [(self.child1.value[self.SENT]*self.child1.value[self.ATTN]
                           +self.child2.value[self.SENT]*self.child2.value[self.ATTN])
                          /(self.child1.value[self.ATTN]+self.child2.value[self.ATTN]),
                          (self.child1.value[self.ATTN]+self.child2.value[self.ATTN])/2, 
                          (self.child1.value[self.TXT]+" "+self.child2.value[self.TXT])]
        # ATTN
        child1_attn = self.child1.value[self.ATTN]
        child2_attn = self.child2.value[self.ATTN]
        if child1_attn >= child2_attn:
            self.child_nuclearity = ['Nucleus', 'Satellite']
        elif child1_attn < child2_attn:
            self.child_nuclearity = ['Satellite', 'Nucleus']
            
    def nuc_nuc_calc_function(self, span_calc_function, nuc_calc_function, gold_label, child1_nuc=None, child2_nuc=None):
        self.child_nuclearity = ['Nucleus', 'Nucleus']
        NN_attention = (self.child1.value[self.ATTN]+self.child2.value[self.ATTN])/2
        # MAX
        if span_calc_function == 'max':
            self.value = [(self.child1.value[self.SENT]*NN_attention
                           +self.child2.value[self.SENT]*NN_attention)
                          /(NN_attention+NN_attention),max(NN_attention, NN_attention), 
                          (self.child1.value[self.TXT]+" "+self.child2.value[self.TXT])]
        # AVG
        if span_calc_function == 'avg':
            self.value = [(self.child1.value[self.SENT]*NN_attention
                           +self.child2.value[self.SENT]*NN_attention)
                          /(NN_attention+NN_attention),(NN_attention+NN_attention)/2, 
                          (self.child1.value[self.TXT]+" "+self.child2.value[self.TXT])]
    
    def calc_value(self, span_calc_function, nuc_calc_function, gold_label, child1_nuc=None, child2_nuc=None):
        if self.child1 == None or self.child2 == None:
            print("At least one child is None, make sure to have valud children")
        else:
            # binary Nuclearity
            if nuc_calc_function == 'binary':
                self.binary_nuc_calc_function(span_calc_function, nuc_calc_function, 
                                              gold_label, child1_nuc=None, child2_nuc=None)
            
            # ternary Nuclearity
            if nuc_calc_function == 'ternary':
                if child1_nuc == child2_nuc == 'Nucleus':
                    self.nuc_nuc_calc_function(span_calc_function, nuc_calc_function, 
                                               gold_label, child1_nuc=None, child2_nuc=None)
                else:
                    self.binary_nuc_calc_function(span_calc_function, nuc_calc_function, 
                                                  gold_label, child1_nuc=None, child2_nuc=None)


class CKY:
    def __init__(self, reduce_to, stoch, mode, real_out, greedy='merge', nuc_calc_function='attn'):
        self.matrix = None
        self.reduce_to = reduce_to
        self.stoch = stoch
        self.number_trees_computed = 0
        assert greedy in [None,'merge','sentence'], 'Greedy procedure unknown'
        self.greedy = greedy
        self.mode = mode
        self.real_out = real_out
        possible_span_calc_functions = ['avg','max']
        if mode in possible_span_calc_functions:
            self.mode = mode
        else:
            raise Exception('span_calc_function must be one of {}, but {} found'
                            .format(possible_span_calc_functions, span_calc_function))
    
        possible_nuc_calc_functions = ['binary', 'ternary']
        if nuc_calc_function in possible_nuc_calc_functions:
            self.nuc_calc_function = nuc_calc_function
        else:
            raise Exception('nuc_calc_function must be one of {}, but {} found'
                            .format(possible_nuc_calc_functions, nuc_calc_function))

    def parse(self, scores):
        # Create empty matrix of apropriate size
        self.matrix = [[[] for x in range(len(scores))] for y in range(len(scores))]
        # Initialize the matrix with the terminals
        for score_idx in range(0,len(scores),1):
            if type(scores[score_idx][0]) == float:
                self.matrix[score_idx][score_idx] = [Node(scores[score_idx], None, None)]
            else:
                if self.greedy == 'sentence':
                    scores[score_idx] = self.top_k(self.reduce_to, scores[score_idx], 
                                                   (x+(len(self.matrix)-1)-y), len(self.matrix)-1)
                self.matrix[score_idx][score_idx] = scores[score_idx]
        # Iterate over all cells to be filled
        for x_element in range(0, len(scores), 1):
            for y_element in range(x_element-1, -1, -1):
                self.calculate_cell(x_element, y_element)
                
    def calculate_cell(self, x, y):
        for x_idx, x_combinations in enumerate(range(x-abs(x-y),x,1)):
            y_combinations = y+x_idx+1
            x_cell = self.matrix[x_combinations][y]
            y_cell = self.matrix[x][y_combinations] 
            for x_node in x_cell:
                for y_node in y_cell:
                    if self.nuc_calc_function == 'ternary':
                        new_node = Node(None, x_node, y_node)
                        new_node.calc_value(self.mode, self.nuc_calc_function, self.real_out, 'Nucleus', 'Nucleus')
                        self.matrix[x][y].append(new_node)
                        new_node = Node(None, x_node, y_node)
                        new_node.calc_value(self.mode, self.nuc_calc_function, self.real_out)
                        self.matrix[x][y].append(new_node)
                    else:
                        new_node = Node(None, x_node, y_node)
                        new_node.calc_value(self.mode, self.nuc_calc_function, self.real_out)
                        self.matrix[x][y].append(new_node)
        if self.greedy == 'merge':
            self.matrix[x][y] = self.top_k(self.reduce_to, self.matrix[x][y], (x+(len(self.matrix)-1)-y), len(self.matrix)-1)
                    
    def get_final_candidates(self):
        return self.matrix[-1][0]
    
    def top_k(self, k, node_list, dist, max_dist):
        node_diff_values = []

        if not self.stoch:
            for node in node_list:
                node_diff_values.append(abs(node.value[0]-self.real_out))
            indexes = sorted(range(len(node_diff_values)), key=lambda i: node_diff_values[i], reverse=False)[:k]
            top_nodes = []
            for index in indexes:
                top_nodes.append(node_list[index])
            return top_nodes
        
        elif self.stoch:
            for node in node_list:
                node_diff_values.append(abs(node.value[0]-self.real_out))
            
            lvl_coeff = 1/(dist+1)
            probabilities = self.softmax(node_diff_values, lvl_coeff)
            if k > (len(node_list) - probabilities.count(0)):
                k = (len(node_list) - probabilities.count(0))
            if k < len(node_list):
                top_nodes = np.random.choice(node_list, k, p=probabilities, replace=False)
            else:
                top_nodes = node_list
            return top_nodes
    
    def softmax(self, x, lvl_coeff):
        x = [1/x_i * lvl_coeff for x_i in x]
        e_x = np.exp(x - np.max(x))
        return list(e_x / e_x.sum())
    
    def get_best(self, gold_label, doc_id, data):
        best_node = None
        min_val = 9999
        best_node_val = 0
        for node in self.matrix[-1][0]:
            if abs(node.value[0] - self.real_out) < min_val:
                min_val = abs(node.value[0] - self.real_out)
                best_node_val = node.value[0]
                best_node = node
        return best_node, min_val, best_node_val, len(self.matrix[-1][0])
    
    def get_number_trees(self):
        matrix_length = len(self.matrix) # as the matrix is always a square, this is the length in both directions
        nr_trees = 0
        
        for matrix_row in range(0, matrix_length):
            for matrix_column in range(0, matrix_length):
                nr_trees += len(self.matrix[matrix_row][matrix_column])
        return nr_trees

    def tree_recursion_dis(self, node, nuclearity, span, lvl):
        if node == None:
            return ('', span)

        nuclearity_child1, nuclearity_child2 = None, None
        if node.get_childs()[0] is not None:
            nuclearity_child1 = node.child_nuclearity[0]
        if node.get_childs()[1] is not None:
            nuclearity_child2 = node.child_nuclearity[1]
        (output1, span1) = self.tree_recursion_dis(node.get_childs()[0], nuclearity_child1, span, lvl+1)
        (output2, span2) = self.tree_recursion_dis(node.get_childs()[1], nuclearity_child2, span1, lvl+1)

        output = output1 + output2

        # Leaf node
        if node.get_childs()[0] == None and node.get_childs()[1] == None:
            if nuclearity == ' Root ':
                n = ' Root ' 
                relation = ''
            else:
                n = ' '+nuclearity+' '
                relation = '(rel2par span)'
            output = (output + '\n'+(' '*(lvl*2))+'('+n+'(leaf '+str(max(span1,span2))+') ' 
                      + relation + ' (text _!'+node.value[2]+'_!) )')
            return (output, span2+1)
        else:
            if nuclearity == ' Root ':
                n = ' Root ' 
                relation = ''
            else:
                n = ' '+nuclearity+' '
                relation = '(rel2par span)'
            output = ('\n'+(' '*(lvl*2))+'('+n+'(span '+str(span)+' '+str(span2-1)+') ' 
                      + relation + output + ' \n'+(' '*(lvl*2))+')')
            return (output, span2)


def create_dp_trees(save_dir, data_ext, model_ext, out_ext,
                    cpu_workers, cuda_no, batch_size, gru_hidden_size,
                    classes, cky_calc, cky_samples, cky_doc_len, reduce_to, 
                    stoch, greedy, nuc_calc_function, allowed_mem_percentage, 
                    overwrite):

    create_cky_data_input(save_dir, data_ext, model_ext,
                          cpu_workers, cuda_no, batch_size,
                          gru_hidden_size, classes, overwrite)
    errors_to_stop = 10000
    mem = psutil.virtual_memory()
    resource.setrlimit(resource.RLIMIT_AS,
                       (mem.total*allowed_mem_percentage,
                        mem.total*allowed_mem_percentage))
    resource.setrlimit(resource.RLIMIT_DATA,
                       (mem.total*allowed_mem_percentage,
                        mem.total*allowed_mem_percentage))

    # Set up the HDF5 file
    datasets = [['/edu_level_annotations_test.h5', "test"],
                ['/edu_level_annotations_dev.h5', "dev"],
                ['/edu_level_annotations_train.h5', "train"]
                ]
    for dataset in datasets:
        hdf5_file_name = save_dir+model_ext+dataset[0]
        hdf5_file = h5py.File(hdf5_file_name, 'r', libver='latest', swmr=True)
        nb_documents = len(hdf5_file.keys())
        print("nb_documents", nb_documents)
        print('Starting discourse structure generation with', nb_documents, 'documents on', dataset[1])
        samples_saved = 0
        nb_errors = 0
        if not os.path.exists(save_dir+out_ext+"/"+dataset[1]):
            os.mkdir(save_dir+out_ext+"/"+dataset[1])
        if not os.path.exists(save_dir+"/tmp_memLog"):
            with open(save_dir+"/tmp_memLog", 'w') as f:
                d = dict()
                json.dump(d, f)
        for element in hdf5_file.keys():
            try:
                files = {}
                # Break and continue conditions
                if cky_samples is not None:
                    if samples_saved >= cky_samples:
                        break
                    try:
                        with open(save_dir+"/tmp_memLog") as f:
                            files = json.load(f)
                            if str(element) in files:
                                if files[str(element)] == 1:
                                    samples_saved += 1
                                continue
                    except:
                        print('Cannot read tmp_memLog')
                        os.remove(save_dir+"/tmp_memLog")
                        with open(save_dir+"/tmp_memLog", 'w') as f: f.write("")
                if hdf5_file.get(str(element)) is None:
                    continue
                data = list(hdf5_file.get(str(element)))
                doc_id = str(data[4].decode("UTF-8")).zfill(6)
                if (os.path.exists(save_dir+out_ext+"/"+dataset[1]+"/discourse/"+ doc_id + ".out.dis") and
                    os.path.exists(save_dir+out_ext+"/"+dataset[1]+"/sentences/"+ doc_id + ".out") and 
                    os.path.exists(save_dir+out_ext+"/"+dataset[1]+"/edus/"+ doc_id + ".out.edus") and 
                    os.path.exists(save_dir+out_ext+"/"+dataset[1]+"/info/"+ doc_id + ".out.info")):
                    continue
                gold_label = float(data[1])
                predicted_label = float(data[3])
                length = int(data[5])
                empty_edu = False
                edu_text, edu_sent, edu_attn = [], [], []
                for idx, edu in enumerate(data[6:]):
                    if idx%3 == 0:
                        edu_cleaned = re.sub(r'\.(\s*\.)*','.',edu.decode('utf-8'))
                        if (len(edu_cleaned) == 1 and edu_cleaned in string.punctuation) or (len(edu_cleaned) == 0):
                            empty_edu = True
                        else:
                            edu_text.append(edu_cleaned)
                    elif (idx-1)%3 == 0:
                        if not empty_edu:
                            edu_sent.append(float(edu))
                    elif (idx-2)%3 == 0:
                        if not empty_edu:
                            edu_attn.append(float(edu))
                        else:
                            empty_edu = False
                if len(edu_text) > cky_doc_len:
                    continue
                                        
                with open(save_dir + "/tmp_memLog", 'w') as f:
                    files[str(element)] = 0
                    json.dump(files, f)
                cky_data = list(zip(edu_sent, edu_attn, edu_text))
                input_sentence, cky_sentence_parse = [], []
                
                for idx in range(len(cky_data)):
                    # Add edu to the sentence
                    input_sentence.append(list(cky_data[idx]))
                    # Sentence boundary found
                    if (idx == len(cky_data)-1 or len(nltk.sent_tokenize(
                         list(cky_data[idx])[2] + ' ' + list(cky_data[idx+1])[2])) > 1):
                        if len(input_sentence) > 0:
                            cky = CKY(reduce_to=reduce_to, stoch=stoch, mode=cky_calc, real_out=gold_label, 
                                      greedy=greedy, nuc_calc_function=nuc_calc_function)
                            cky.parse(input_sentence)
                            cky_sentence_parse.append(cky.get_final_candidates())
                            input_sentence = []
                cky = CKY(reduce_to=reduce_to, stoch=stoch, mode=cky_calc, real_out=gold_label, 
                          greedy=greedy, nuc_calc_function=nuc_calc_function)
                cky.parse(cky_sentence_parse)
                best_tree = cky.get_best(gold_label, doc_id, data)
                (out, _) = cky.tree_recursion_dis(node=best_tree[0], nuclearity=' Root ', span=1, lvl=0)

                # Save data as .out, .out.edus, .out.info and .out.dis files
                if not os.path.exists(save_dir+out_ext+"/"+dataset[1]+"/discourse"):
                    os.mkdir(save_dir+out_ext+"/"+dataset[1]+"/discourse")
                if not os.path.exists(save_dir+out_ext+"/"+dataset[1]+"/sentences"):
                    os.mkdir(save_dir+out_ext+"/"+dataset[1]+"/sentences")
                if not os.path.exists(save_dir+out_ext+"/"+dataset[1]+"/edus"):
                    os.mkdir(save_dir+out_ext+"/"+dataset[1]+"/edus")
                if not os.path.exists(save_dir+out_ext+"/"+dataset[1]+"/info"):
                    os.mkdir(save_dir+out_ext+"/"+dataset[1]+"/info")
                with open(save_dir+out_ext+"/"+dataset[1]+"/discourse/"+ doc_id + ".out.dis", "w") as file:
                    file.write(out)
                with open(save_dir+out_ext+"/"+dataset[1]+"/sentences/"+ doc_id + ".out", "w") as file:
                    file.write('\n'.join(nltk.sent_tokenize(' '.join(edu_text))))
                with open(save_dir+out_ext+"/"+dataset[1]+"/edus/"+ doc_id + ".out.edus", "w") as file:
                    file.write('\n'.join(edu_text))
                with open(save_dir+out_ext+"/"+dataset[1]+"/info/"+ doc_id + ".out.info", "w") as file:
                    json.dump({'length': length, 'gold_label': gold_label, 'predicted_label': predicted_label}, file)
                with open(save_dir + "/tmp_memLog", 'w') as f:
                    files[str(element)] = 1
                    json.dump(files, f)
                samples_saved += 1
            except:
                nb_errors += 1
                cky = []
                input_sentence = []
                print('Sample', element, 'exhausted the memory. This was the', nb_errors, 'time for that to happen.')
                if nb_errors == errors_to_stop:
                    raise Exception('Out of Memory')
                pass
        os.remove(save_dir+"/tmp_memLog")
