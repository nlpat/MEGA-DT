from torch.utils import data as t_data
import pickle
import h5py
import copy
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')


class DataFormat(t_data.Dataset):
    def __init__(self, i2d_path, hdf5_path, w2i_path,
                 edu_lvl_supervision, rm_stopwords, lemmatize):
        self.edu_lvl_supervision = edu_lvl_supervision
        self.hdf5_path = hdf5_path
        self.rm_stopwords = rm_stopwords
        self.lemmatize = lemmatize
        self.hdf5_file = None
        self.idx2doc_id = pickle.load(open(i2d_path, 'rb'))
        self.word2index = pickle.load(open(w2i_path, 'rb'))
        self.numerize_ratings = {'+': 3, '0': 2, '-': 1}
        self.missing_words = 0

    def __len__(self):
        return len(self.idx2doc_id)

    def __getitem__(self, index):
        self.hdf5_file = h5py.File(self.hdf5_path, 'r', libver='latest', swmr=True)
        doc_id = self.idx2doc_id[index].encode('utf8')
        if not self.edu_lvl_supervision:
            # First value in the list is the target sentiment
            y = int(list(self.hdf5_file.get(doc_id))[0])
            # The rest of the list contains the edus
            X = list(self.hdf5_file.get(doc_id))[1:]
            # Turn byte strings to regular strings
            X = [i.decode('utf8') for i in X]
        elif self.edu_lvl_supervision:
            X, y = [], []
            # Document level label is not needed with edu_lvl_supervision
            data = list(self.hdf5_file.get(doc_id))[1:]
            for edu in data:
                edu = edu.decode('utf8').split('\t')
                y.append(self.numerize_ratings[edu[0]])
                X.append(edu[1])
        edus = copy.deepcopy(X)
        # Tokenize the EDUs on word level
        X = self._word_tokenize(X)
        edus = self._word_tokenize(edus)
        # If selected, remove stopwords
        if self.rm_stopwords:
            X = self._rm_stopwords(X)
        # If selected, lemmatize the words
        if self.lemmatize:
            X = self._lemmatize(X)
        # Transform words to indexes
        X = self._enumerize_words(X, self.word2index)
        edus = self._enumerize_words(edus, self.word2index)
        doc_id = doc_id.decode('utf8')
        return doc_id, X, y, edus

    # Word tokenize
    # Use nltk word tokenizer to tokenize the words within the EDUs
    def _word_tokenize(self, edu_lvl_seg_data):
        word_lvl_seg_data = copy.deepcopy(edu_lvl_seg_data)
        for edu_id, edu in enumerate(edu_lvl_seg_data):
            word_lvl_seg_data[edu_id] = nltk.word_tokenize(edu.lower())
        return word_lvl_seg_data

    # Remove stop-words
    def _rm_stopwords(self, data):
        rm_stopword_data = []
        for edu in data:
            rm_stopword_data.append([])
            rm_stopword_data[-1] = [word for word in edu
                                    if word not in stopwords.words('english')]
        return rm_stopword_data

    # Lemmatize (stemming not used in original paper)
    def _lemmatize(self, data):
        tmp_data = copy.deepcopy(data)
        lemmatizer = WordNetLemmatizer()
        for e_id, edu in enumerate(tmp_data):
            for w_id, word in enumerate(edu):
                tmp_data[e_id][w_id] = lemmatizer.lemmatize(word)
        return tmp_data

    def _enumerize_words(self, data, w2i):
        enum_data = []
        for edu in data:
            enum_data.append([])
            for word in edu:
                try:
                    enum_data[-1].append(w2i[word])
                except Exception:
                    continue
        return enum_data
