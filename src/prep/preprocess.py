import os
import h5py
import pickle
import torch
from .data_format import DataFormat


# Read EDU segmented documents from text file
def _read_data(path):
    with open(path) as file:
        curr_doc, doc_border = [], True
        for line in file:
            if doc_border:
                header = line.replace('\n', '').split(' ')
                sentiment = header[0].encode('utf8')
                doc_id = header[1].encode('utf8')
                doc_border = False
            else:
                if line != '\n':
                    curr_doc.append(line.replace('\n', '').
                                    replace('<s>', '').
                                    replace('.', '. ').
                                    replace('/', ' ').
                                    encode('utf8'))
                elif line == '\n':
                    yield [sentiment]+[doc_id]+curr_doc
                    curr_doc = []
                    doc_border = True

                    
# Combine the data from text files into a single HDF5 file to be read by the data loader
def _create_hdf_file(data, overwrite, path, prefix):
    if overwrite:
        if os.path.exists(path + '/' + prefix + '_doc.h5'):
            os.remove(path+'/'+prefix+'_doc.h5')
    else:
        if os.path.exists(path+'/'+prefix+'_doc.h5'):
            print("The hdf5 document data file already exists and "
                  + "will not be created again. If you want "
                  + "to generate the file again, please delete "
                  + "the existing file or use the --overwrite parameter.")
            return
    
    hdf5_file_name = path+'/'+prefix+'_doc.h5'
    idx2doc_file_name = path+'/'+prefix+'_idx2doc.dict'
    # If an .h5 file exists, replace it with a newly generated file
    if os.path.exists(hdf5_file_name):
        os.remove(hdf5_file_name)
    # Open the hdf5 file
    hdf5_file = h5py.File(hdf5_file_name, 'w', libver='latest', swmr=True)
    # Create a mapping of index to document name
    idx2doc_id = {}
    # Iterate over the dataset in .txt format and aggregate by document
    for idx, document in enumerate(_read_data(data)):
        sentiment, doc_id, edus = document[0], document[1], document[2:]
        hdf5_file.create_dataset(doc_id, data=[sentiment]+edus)
        # Save the index to document_id relation
        idx2doc_id[int(idx)] = doc_id.decode('utf8')
    # Save the hdf5 file
    hdf5_file.close()
    with open(idx2doc_file_name, 'wb') as file:
        pickle.dump(idx2doc_id, file)

# Prepare the GloVe embeddings so they
# can be used within the pytorch embedding layer
def _prepare_embeddings(data_path, overwrite, save_path, pad, vector_len=300):
    if overwrite:
        if os.path.exists(save_path+'/glove_vectors.tensor'):
            os.remove(save_path+'/glove_vectors.tensor')
    else:
        if os.path.exists(save_path+'/glove_vectors.tensor'):
            print("The glove vector file already exists and "
                  + "will not be created again. If you want "
                  + "to generate the file again, please delete "
                  + "the existing file or use the --overwrite parameter.")
            return
    idx, words, word2idx, idx2word = 1, [pad], {pad: 0}, {0: pad}
    vectors = [torch.zeros([vector_len])]
    with open(data_path, 'rb') as file:
        for line in file:
            line = line.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx2word[idx] = word
            vector = torch.tensor([float(i) for i in line[1:]], dtype=torch.float)
            vectors.append(vector)
            idx += 1
    vectors = torch.stack(vectors)
    pickle.dump(words, open(save_path+'/glove_words.list', 'wb'))
    pickle.dump(word2idx, open(save_path+'/glove_word2idx.dict', 'wb'))
    pickle.dump(idx2word, open(save_path+'/glove_idx2word.dict', 'wb'))
    pickle.dump(vectors, open(save_path+'/glove_vectors.tensor', 'wb'))

# Pad documents and EDUs
def _pad(data, max_doc_size, max_edu_size):
    doc_size, edu_size = [], []
    pad_data = torch.zeros([len(data), max_doc_size, max_edu_size], dtype=torch.long)
    for doc_id, document in enumerate(data):
        tmp_doc_size = min(len(document), max_doc_size)
        doc_size.append(tmp_doc_size)
        for edu_id, edu in enumerate(document[:tmp_doc_size]):
            tmp_edu_size = min(len(edu), max_edu_size)
            edu_size.append(tmp_edu_size)
            pad_data[doc_id][edu_id][:tmp_edu_size] = torch.tensor(edu[:tmp_edu_size],dtype=torch.long)
    doc_size = torch.tensor(doc_size)
    edu_size = torch.tensor(edu_size)
    return (pad_data, doc_size, edu_size)


class Batch_Creation():

    def __init__(self, max_doc_size, max_edu_size):
        self.max_doc_size = max_doc_size
        self.max_edu_size = max_edu_size

    # Prepare batch data in data loader
    def _create_batch(self, data):
        # Seperate source and target sequences
        doc_id, X, y, edus = zip(*data)
        # Convert tupels to lists
        doc_id, X, y, edus = list(doc_id), list(X), list(y), list(edus)
        # Pad sequences with 0
        X, doc_len, edu_len = _pad(X, self.max_doc_size, self.max_edu_size)
        edus, _, _ = _pad(edus, self.max_doc_size, self.max_edu_size)
        # For SPOT data
        if isinstance(y[0], list):
            for idx, document in enumerate(y):
                y[idx] = document + [0]*(self.max_doc_size-len(document))
        y = torch.tensor(y)
        doc_id = torch.tensor([int(d) for d in doc_id])
        return (doc_id, X, doc_len, edu_len, y, edus)

def _preprocess_data(save_path, prefix, index_file, data_file,
                     word2idx_file, overwrite, batch_size, batch_prep,
                     cpu_workers, edu_lvl_supervision, rm_stopwords,
                     lemmatize, shuffle=True, drop_last=False):
    if overwrite:
        if os.path.exists(save_path+'/'+prefix+'_data.h5'):
            os.remove(save_path+'/'+prefix+'_data.h5')
    else:
        if os.path.exists(save_path+'/'+prefix+'_data.h5'):
            print("The hdf5 model input file already exists and "
                  + "will not be created again. If you want "
                  + "to generate the file again, please delete "
                  + "the existing file or use the --overwrite parameter.")
            return

    dataset = DataFormat(index_file, data_file,
                         word2idx_file, edu_lvl_supervision,
                         rm_stopwords, lemmatize)
    # data loader for the custom dataset
    dl = torch.utils.data.DataLoader(
                    dataset=dataset,  batch_size=batch_size,
                    shuffle=shuffle, collate_fn=batch_prep,
                    num_workers=cpu_workers, drop_last=drop_last)
    hdf5_file_name = save_path+'/'+prefix+'_data.h5'
    idx2batch_id_file_name = save_path+'/'+prefix+'_idx2data.dict'
    # If an .h5 file exists, replace it with a newly generated file
    if os.path.exists(hdf5_file_name):
        os.remove(hdf5_file_name)
    # Open the hdf5 file
    hdf5_file = h5py.File(hdf5_file_name, 'w', libver='latest', swmr=True)
    # Create a mapping of index to dataset name for the data loader
    idx2batch_id = {}
    for batch_idx, (doc_id, X, doc_len, edu_len, y, edus) in enumerate(dl):
        # Save the batch as a dataset within the hdf5 file
        hdf5_file.create_dataset(str(batch_idx)+'_x', data=X)
        hdf5_file.create_dataset(str(batch_idx)+'_edu', data=edus)
        hdf5_file.create_dataset(str(batch_idx)+'_doc_len', data=doc_len)
        hdf5_file.create_dataset(str(batch_idx)+'_edu_len', data=edu_len)
        hdf5_file.create_dataset(str(batch_idx)+'_y', data=y)
        hdf5_file.create_dataset(str(batch_idx)+'_doc_id', data=doc_id)
        # Save the index to batch_id relation
        idx2batch_id[int(batch_idx)] = {'data': str(batch_idx)+'_x',
                                        'edu': str(batch_idx)+'_edu',
                                        'doc_length': str(batch_idx)+'_doc_len',
                                        'edu_length': str(batch_idx)+'_edu_len',
                                        'labels': str(batch_idx)+'_y',
                                        'doc_id': str(batch_idx)+'_doc_id'}
    # Save the hdf5 file
    hdf5_file.close()
    with open(idx2batch_id_file_name, 'wb') as file:
        pickle.dump(idx2batch_id, file)
    # Remove intermediate documents
    if os.path.exists(index_file):
        os.remove(index_file)
    if os.path.exists(data_file):
        os.remove(data_file)

def _create_directory_structure(save_dir, model_ext, data_ext, out_ext, dp_ext):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(save_dir+model_ext):
        os.makedirs(save_dir+model_ext)
    if not os.path.exists(save_dir+data_ext):
        os.makedirs(save_dir+data_ext)
    if not os.path.exists(save_dir+out_ext):
        os.makedirs(save_dir+out_ext)
    if not os.path.exists(save_dir+dp_ext):
        os.makedirs(save_dir+dp_ext)

def run_preprocessing(save_dir, data_ext, model_ext, out_ext, dp_ext,
                      train_data, dev_data, eval_data,
                      edu_eval_data, glove_data, lemmatize, rm_stopwords,
                      pad_token, max_doc_len, max_edu_len, batch_size,
                      cpu_workers, overwrite):
    # Create directory structure if not existent
    print('Running preprocessing...')
    print('Initializing directory structure...')
    _create_directory_structure(save_dir, model_ext, data_ext, out_ext, dp_ext)
    print('Generating temporary hdf5 files...')
    _create_hdf_file(train_data, overwrite, save_dir+data_ext, prefix='train')
    _create_hdf_file(dev_data, overwrite, save_dir+data_ext, prefix='dev')
    _create_hdf_file(eval_data, overwrite, save_dir+data_ext, prefix='test')
    _create_hdf_file(edu_eval_data, overwrite, save_dir+data_ext, prefix='edu_test')
    print('Preparing GloVe embeddings...')
    _prepare_embeddings(glove_data, overwrite, save_dir+data_ext, pad_token, vector_len=300)
    print('Creating data files...')
    batch = Batch_Creation(max_doc_len, max_edu_len)
    _preprocess_data(save_dir+data_ext, 'train',
                          save_dir+data_ext+'/train_idx2doc.dict',
                          save_dir+data_ext+'/train_doc.h5',
                          save_dir+data_ext+'/glove_word2idx.dict',
                          overwrite,
                          batch_size,
                          batch_prep=batch._create_batch,
                          cpu_workers=cpu_workers,
                          edu_lvl_supervision=False,
                          rm_stopwords=rm_stopwords,
                          lemmatize=lemmatize,
                          shuffle=True,
                          drop_last=True)
    _preprocess_data(save_dir+data_ext, 'dev',
                          save_dir+data_ext+'/dev_idx2doc.dict',
                          save_dir+data_ext+'/dev_doc.h5',
                          save_dir+data_ext+'/glove_word2idx.dict',
                          overwrite,
                          batch_size,
                          batch_prep=batch._create_batch,
                          cpu_workers=cpu_workers,
                          edu_lvl_supervision=False,
                          rm_stopwords=rm_stopwords,
                          lemmatize=lemmatize,
                          shuffle=True,
                          drop_last=True)
    _preprocess_data(save_dir+data_ext, 'test',
                          save_dir+data_ext+'/test_idx2doc.dict',
                          save_dir+data_ext+'/test_doc.h5',
                          save_dir+data_ext+'/glove_word2idx.dict',
                          overwrite,
                          batch_size,
                          batch_prep=batch._create_batch,
                          cpu_workers=cpu_workers,
                          edu_lvl_supervision=False,
                          rm_stopwords=rm_stopwords,
                          lemmatize=lemmatize,
                          shuffle=True,
                          drop_last=True)
    _preprocess_data(save_dir+data_ext, 'edu_test',
                          save_dir+data_ext+'/edu_test_idx2doc.dict',
                          save_dir+data_ext+'/edu_test_doc.h5',
                          save_dir+data_ext+'/glove_word2idx.dict',
                          overwrite,
                          batch_size,
                          batch_prep=batch._create_batch,
                          cpu_workers=cpu_workers,
                          edu_lvl_supervision=True,
                          rm_stopwords=rm_stopwords,
                          lemmatize=lemmatize,
                          shuffle=True,
                          drop_last=False)
    print('Finishing preprocessing...')
