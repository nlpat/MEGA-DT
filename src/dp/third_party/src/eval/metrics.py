#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Yizhong
# created_at: 10/27/2016 下午8:34

import numpy


class Performance(object):
    def __init__(self, percision, recall, hit_num):
        self.percision = percision
        self.recall = recall
        self.hit_num = hit_num
        
def convert_to_orig_parseval(rst_parseval):
    orig_parseval = []
    for constituent in rst_parseval:
        # Not a leaf -> Leaves are removed in original parseval
        # Constituent are in format [[BEGIN_IDX, END_IDX], NUCLEARITY, RELATION]
        if not constituent[0][0] == constituent[0][1]:
             orig_parseval.append([[constituent[0][0], constituent[0][1]], None, "rel2par:span"])
    
    # Add root node
    max_edu = max([edu[0][1] for edu in rst_parseval])
    orig_parseval.append([[1, max_edu], None, "rel2par:span"])
    
    # Add Nuclearity
    # For each of the original parseval nodes left, assign them a nuclearity based on the children
    # e.g for [[1,15], TBD, X] --> if [[1,10], Nucleus, X] and [[11,15], Satellite, X] --> [[1,15], N-S, X]
    for orig_idx, orig_constituent in enumerate(orig_parseval):
        # Iterate over all RST parseval constituents 
        # and break once the current orig parseval constituent is reached,
        # as there is nothing more left to do
        for rst_constituent in rst_parseval:
            # Break if the rst-parseval node equals the orig parseval node
            if (rst_constituent[0][0] == orig_constituent[0][0] and
                rst_constituent[0][1] == orig_constituent[0][1]):
                break
            # Continue to overwrite 'left' with the largest child-constituent (due to post-order-traversal)
            elif (rst_constituent[0][0] == orig_constituent[0][0]):
                left = rst_constituent[1][0]
            # Continue to overwrite 'right' with the largest child-constituent (due to post-order-traversal)
            elif (rst_constituent[0][1] == orig_constituent[0][1]):
                right = rst_constituent[1][0]
        # Assign the parent node with the combination of the child node nuclearities
        orig_parseval[orig_idx][1] = left+"-"+right
    return orig_parseval

class Metrics(object):
    def __init__(self, levels=['span', 'nuclearity', 'relation']):
        """ Initialization

        :type levels: list of string
        :param levels: eval levels, the possible values are only
                       'span','nuclearity','relation'
        """
        self.levels = levels
        self.span_perf = Performance([], [], 0)
        self.nuc_perf = Performance([], [], 0)
        self.rela_perf = Performance([], [], 0)
        self.span_num = 0
        self.hit_num_each_relation = {}
        self.pred_num_each_relation = {}
        self.gold_num_each_relation = {}

    def eval(self, goldtree, predtree, measure):
        """ Evaluation performance on one pair of RST trees

        :type goldtree: RSTTree class
        :param goldtree: gold RST tree

        :type predtree: RSTTree class
        :param predtree: RST tree from the parsing algorithm
        """
        goldbrackets = goldtree.bracketing()
        predbrackets = predtree.bracketing()
        
        # Transform RST-parseval trees to original-parseval trees
        if measure == "orig":
            goldbrackets = convert_to_orig_parseval(goldbrackets)
            predbrackets = convert_to_orig_parseval(predbrackets)
        
        self.span_num += len(goldbrackets)
        for level in self.levels:
            if level == 'span':
                self._eval(goldbrackets, predbrackets, idx=1)
            elif level == 'nuclearity':
                self._eval(goldbrackets, predbrackets, idx=2)
            elif level == 'relation':
                self._eval(goldbrackets, predbrackets, idx=3)
            else:
                raise ValueError("Unrecognized eval level: {}".format(level))

    def _eval(self, goldbrackets, predbrackets, idx):
        """ Evaluation on each discourse span
        """
        # goldspan = [item[:idx] for item in goldbrackets]
        # predspan = [item[:idx] for item in predbrackets]
        if idx == 1 or idx == 2:
            goldspan = [item[:idx] for item in goldbrackets]
            predspan = [item[:idx] for item in predbrackets]
        elif idx == 3:
            goldspan = [(item[0], item[2]) for item in goldbrackets]
            predspan = [(item[0], item[2]) for item in predbrackets]
        else:
            raise ValueError('Undefined idx for evaluation')
        hitspan = [span for span in goldspan if span in predspan]
        if len(goldspan) == 0:
            print('Goldspan len is 0')
        elif len(goldspan) > 0:
            p, r = 0.0, 0.0
            for span in hitspan:
                if span in goldspan:
                    p += 1.0
                if span in predspan:
                    r += 1.0
            if idx == 1:
                self.span_perf.hit_num += p
            elif idx == 2:
                self.nuc_perf.hit_num += p
            elif idx == 3:
                self.rela_perf.hit_num += p
            p /= len(goldspan)
            r /= len(predspan)
            if idx == 1:
                self.span_perf.percision.append(p)
                self.span_perf.recall.append(r)
            elif idx == 2:
                self.nuc_perf.percision.append(p)
                self.nuc_perf.recall.append(r)
            elif idx == 3:
                self.rela_perf.percision.append(p)
                self.rela_perf.recall.append(r)
            if idx == 3:
                for span in hitspan:
                    relation = span[-1]
                    if relation in self.hit_num_each_relation:
                        self.hit_num_each_relation[relation] += 1
                    else:
                        self.hit_num_each_relation[relation] = 1
                for span in goldspan:
                    relation = span[-1]
                    if relation in self.gold_num_each_relation:
                        self.gold_num_each_relation[relation] += 1
                    else:
                        self.gold_num_each_relation[relation] = 1
                for span in predspan:
                    relation = span[-1]
                    if relation in self.pred_num_each_relation:
                        self.pred_num_each_relation[relation] += 1
                    else:
                        self.pred_num_each_relation[relation] = 1

    def report(self):
        """ Compute the F1 score for different eval levels
            and print it out
        """
        for level in self.levels:
            if 'span' == level:
                p = numpy.array(self.span_perf.percision).mean()
                r = numpy.array(self.span_perf.recall).mean()
                f1 = (2 * p * r) / (p + r)
                print("\t\t Structure:")
                print('\t\t\t Macro precision {0:.4f}'.format(p))
                #print('Recall on span level is {0:.4f}'.format(r))
                #print('F1 score on span level is {0:.4f}'.format(f1))
                print('\t\t\t Micro precision {0:.4f}'.format(self.span_perf.hit_num / self.span_num))
            elif 'nuclearity' == level:
                print("\t\t Nuclearity:")
                p = numpy.array(self.nuc_perf.percision).mean()
                r = numpy.array(self.nuc_perf.recall).mean()
                f1 = (2 * p * r) / (p + r)
                print('\t\t\t Macro precision {0:.4f}'.format(p))
                #print('Recall on nuclearity level is {0:.4f}'.format(r))
                #print('F1 score on nuclearity level is {0:.4f}'.format(f1))
                print('\t\t\t Micro precision {0:.4f}'.format(self.nuc_perf.hit_num / self.span_num))
                print("\n")
            elif 'relation' == level:
                p = numpy.array(self.rela_perf.percision).mean()
                r = numpy.array(self.rela_perf.recall).mean()
                f1 = (2 * p * r) / (p + r)
                #print('Average precision on relation level is {0:.4f}'.format(p))
                #print('Recall on relation level is {0:.4f}'.format(r))
                #print('F1 score on relation level is {0:.4f}'.format(f1))
                #print('Global precision on relation level is {0:.4f}'.format(self.rela_perf.hit_num / self.span_num))
            else:
                raise ValueError("Unrecognized eval level")
        # sorted_relations = sorted(self.gold_num_each_relation.keys(), key=lambda x: self.gold_num_each_relation[x])
        sorted_relations = sorted(self.gold_num_each_relation.keys())
        for relation in sorted_relations:
            hit_num = self.hit_num_each_relation[relation] if relation in self.hit_num_each_relation else 0
            gold_num = self.gold_num_each_relation[relation]
            pred_num = self.pred_num_each_relation[relation] if relation in self.pred_num_each_relation else 0
            precision = hit_num / pred_num if pred_num > 0 else 0
            recall = hit_num / gold_num
            try:
                f1 = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                f1 = 0
