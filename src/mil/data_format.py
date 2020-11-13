from torch.utils import data as t_data
import pickle
import h5py
import torch


class DataFormat(t_data.Dataset):
    def __init__(self, idx2batch_path, hdf5_path):
        self.hdf5_path = hdf5_path
        self.hdf5_file = None
        self.idx2batch_id = pickle.load(open(idx2batch_path, 'rb'))

    def __len__(self):
        return len(self.idx2batch_id)

    def __getitem__(self, index):
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r',
                                       libver='latest', swmr=True)
        element = self.idx2batch_id[index]
        try:
            X = torch.tensor(self.hdf5_file.get(element['data']))
            edus = torch.tensor(self.hdf5_file.get(element['edu']))
            doc_len = torch.tensor(self.hdf5_file.get(element['doc_length']))
            edu_len = torch.tensor(self.hdf5_file.get(element['edu_length']))
            y = torch.tensor(self.hdf5_file.get(element['labels']))
            doc_id = torch.tensor(self.hdf5_file.get(element['doc_id']))
            return doc_id, X, doc_len, edu_len, y, edus
        except:
            print("Can't find datapoint, moving on...")
