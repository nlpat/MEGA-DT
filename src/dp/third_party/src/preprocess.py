import argparse
import os
import random
import copy
from multiprocessing import Pool
from .utils.xmlreader import reader, writer, combine
from pycorenlp import StanfordCoreNLP
import signal
import threading
import time
import psutil


class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def join_edus(fedu, ftxt):
    with open(fedu, 'r') as fin, open(ftxt, 'w') as fout:
        lines = [l.strip() for l in fin if l.strip()]
        fout.write(' '.join(lines))


def extract(fxml, fconll):
    #print(fxml)
    sent_list, const_list = reader(fxml)
    if not sent_list is None and not const_list is None:
        sent_list = combine(sent_list, const_list)
        writer(sent_list, fconll)


def merge(fedu, fsentence, ftxt, fxml, fconll, fmerge):
    with open(fconll, 'r') as fin1, open(fedu, 'r') as fin2, open(fsentence, 'r') as fin3, open(fmerge, 'w') as fout:
        edus = [l.strip() for l in fin2 if l.strip()]
        paras = []
        para_cache = ''
        for line in fin3:
            if line.strip():
                para_cache += line.strip() + ' '
            else:
                paras.append(para_cache.strip())
                para_cache = ''
        if para_cache:
            paras.append(para_cache)
        edu_idx = 0
        para_idx = 0
        cur_edu_offset = len(edus[edu_idx]) - 1 + 1  # plus 1 for one blank space
        edu_cache = ''
        for line in fin1:
            if not line.strip():
                continue
            line_info = line.strip().split()
            token_end_offset = int(line_info[-1])
            fout.write('%s\t%s\t%s\n' % ('\t'.join(line.strip().split('\t')[:-2]), edu_idx + 1, para_idx + 1))
            if token_end_offset == cur_edu_offset:
                edu_cache += edus[edu_idx] + ' '
                if len(edu_cache) == len(paras[para_idx]) + 1:
                    edu_cache = ''
                    para_idx += 1
                edu_idx += 1
                if edu_idx < len(edus):
                    cur_edu_offset += len(edus[edu_idx]) + 1
            elif token_end_offset > cur_edu_offset:
                print("Error while merging token \"{}\" in file {} with edu : {}.".format(line_info[2], fconll,
                                                                                          edus[edu_idx]))
                edu_idx += 1
                if edu_idx < len(edus):
                    cur_edu_offset += len(edus[edu_idx]) + 1
                    

def coreNLP_annotate(txt_list, xml_dir, corenlp_dir):
    # Start the CoreNLP restart deamon
    corenlp_thread = CoreNLP_Thread(corenlp_dir)
    corenlp_thread.start()
    
    # Start the XML generation processes
    processes = 8
    chunk_size = int(len(txt_list)/processes)+1
    txt_list_parallel = copy.copy(txt_list)
    random.shuffle(txt_list_parallel)
    pool = Pool(processes)
    tasks = [(txt_list_parallel[i:min(i+chunk_size,len(txt_list_parallel))], xml_dir) 
             for i in range(0,len(txt_list_parallel),chunk_size)]
    pool.starmap(annotation_process, tasks)
    
    nlp = StanfordCoreNLP('http://localhost:9000')
    for text_file in txt_list:
        if not os.path.isfile(os.path.join(xml_dir, os.path.basename(text_file).replace(".text", ".text.xml"))):
            with open(text_file, "r") as f:
                try:
                    with timeout(seconds=(1000)):
                        res = nlp.annotate(f.read(), properties={'annotators': 'tokenize,ssplit,pos,lemma,ner,parse',
                                       'outputFormat': 'xml', 'timeout': 100000})
                    with open(os.path.join(xml_dir, os.path.basename(text_file).replace(".text", ".text.xml")), "w") as f:
                        print("Saving to", os.path.join(xml_dir, os.path.basename(text_file).replace(".text", ".text.xml")))
                        f.write(res)
                except:
                    # Don't restart CoreNLP here as this will instead be done by the deamon, give it some time to restart...
                    time.sleep(10)
                    pass

    corenlp_thread.stop()
    corenlp_thread.join()

def find_corenlp_process(user):
    for process in psutil.process_iter():
        with process.oneshot():
            pid = process.pid
            cmd = " ".join(process.cmdline())
            username = process.username()
            try : cores = len(process.cpu_affinity())
            except psutil.AccessDenied : cores = 0
            cpu_usage = process.cpu_percent()
            if username == user:
                if "java" in cmd and "StanfordCoreNLPServer" in cmd:
                    return True, cpu_usage
    return False, None


class CoreNLP_Thread(threading.Thread):
    def __init__(self, corenlp_dir, name='CoreNLP_Thread'):
        self.stopped = False
        self.corenlp_dir = corenlp_dir
        threading.Thread.__init__(self, name=name)

    def run(self):
        inactivity_count = 0
        while not self.stopped:
            found_process, cpu_usage = find_corenlp_process(os.getlogin())
            if found_process:
                if cpu_usage < 1.0:
                    inactivity_count += 1
                else: 
                    inactivity_count = 0
            elif not found_process:
                print("No CoreNLP service found. Starting now...")
                os.system("java -mx5g -cp '"+self.corenlp_dir+"*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 100000&")
                
            if inactivity_count > 4:
                print("CoreNLP service unresponsive. Restarting now...")
                os.system("ps -ef | grep edu.stanford.nlp.pipeline.StanfordCoreNLPServer | grep -v grep | awk '{print $2}' | xargs kill")
                os.system("java -mx5g -cp '"+self.corenlp_dir+"*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 100000&")
            time.sleep(1.0)

    def stop(self):
        self.stopped = True
        

def annotation_process(files, xml_dir):
    nlp = StanfordCoreNLP('http://localhost:9000')
    for text_file in files:
        if not os.path.isfile(os.path.join(xml_dir, os.path.basename(text_file).replace(".text", ".text.xml"))):
            with open(text_file, "r") as f:
                try:
                    with timeout(seconds=(100)):
                        # timeout here is in milliseconds
                        res = nlp.annotate(f.read(), properties={'annotators': 'tokenize,ssplit,pos,lemma,ner,parse',
                                       'outputFormat': 'xml', 'timeout': 100000})
                    with open(os.path.join(xml_dir, os.path.basename(text_file).replace(".text", ".text.xml")), "w") as f:
                        print("Saving to", os.path.join(xml_dir, os.path.basename(text_file).replace(".text", ".text.xml")))
                        f.write(res)
                except:
                    # Don't restart CoreNLP here as this will trigger all parallel processed to kill and restart CoreNLP.
                    # Instead the deamon takes care of that, give it some time to restart...
                    time.sleep(10)
                    pass

            
def preprocess(edu_dir, sentences_dir, text_dir, xml_dir, conll_dir, merge_dir, txcm, corenlp_dir):
    print('Starting Preprocessing...')
    edu_files = [os.path.join(edu_dir, fname) for fname in os.listdir(edu_dir)]
    print("Collected all", len(edu_files), 'edu files...')
    
    if 't' in txcm:
        print("Generating text files...")
        for fedu in edu_files:
            ftxt = os.path.join(text_dir, os.path.basename(fedu).replace('.edus', '.text'))
            join_edus(fedu, ftxt)

    if 'x' in txcm:
        print("Generating XML files...")
        #try: 
        #    nlp = StanfordCoreNLP('http://localhost:9000')
        #except:
        #    print("""Could not find the coreNLP webservice at http://localhost:9000.
        #             Please ensure to have a CoreNLP server running by entering:
        #             java -mx5g -cp '/PATH/TO/CoreNLP/stanford-corenlp-full-2018-10-05/*' \
        #             edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 60000&""")
        text_files = [os.path.join(text_dir, fname) for fname in os.listdir(text_dir)]
        coreNLP_annotate(text_files, xml_dir, corenlp_dir)
    
    if 'c' in txcm:
        print("Generating conll files...")
        for fedu in edu_files:
            fxml = os.path.join(xml_dir, os.path.basename(fedu).replace('.edus', '.text.xml'))
            fconll = os.path.join(conll_dir, os.path.basename(fedu).replace('.edus', '.conll'))
            if os.path.exists(fxml):
                extract(fxml, fconll)
    
    if 'm' in txcm:
        print("Generating merge files...")
        for fedu in edu_files:
            fsentence = os.path.join(sentences_dir, os.path.basename(fedu).replace('.edus', ''))
            ftxt = os.path.join(text_dir, os.path.basename(fedu).replace('.edus', '.text'))
            fxml = os.path.join(xml_dir, os.path.basename(fedu).replace('.edus', '.text.xml'))
            fconll = os.path.join(conll_dir, os.path.basename(fedu).replace('.edus', '.conll'))
            fmerge = os.path.join(merge_dir, os.path.basename(fedu).replace('.edus', '.merge'))
            if os.path.exists(fsentence) and os.path.exists(ftxt) and os.path.exists(fxml) and os.path.exists(fconll):
                merge(fedu, fsentence, ftxt, fxml, fconll, fmerge)
