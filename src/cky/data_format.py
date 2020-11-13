from mil.training import _data_loader, _batch_prep, _evaluate_network
from mil.network import AttentionWordRNN, AttentionSentRNN
import torch
import torch.nn.functional as F
import pickle
import time
import os
import numpy as np
import h5py
import re

def create_cky_data_input(save_dir, data_ext, model_ext,
                          cpu_workers, cuda_no, batch_size,
                          gru_hidden_size, classes, overwrite):

    train_data = _data_loader(
                   save_dir+data_ext+'/train_idx2data.dict',
                   save_dir+data_ext+'/train_data.h5',
                   cpu_workers=cpu_workers,
                   collate=_batch_prep,
                   shuffle=True)
    
    dev_data = _data_loader(
                   save_dir+data_ext+'/dev_idx2data.dict',
                   save_dir+data_ext+'/dev_data.h5',
                   cpu_workers=cpu_workers,
                   collate=_batch_prep,
                   shuffle=True)
    
    test_data = _data_loader(
                   save_dir+data_ext+'/test_idx2data.dict',
                   save_dir+data_ext+'/test_data.h5',
                   cpu_workers=cpu_workers,
                   collate=_batch_prep,
                   shuffle=True)
    
    datasets = [
        [train_data, 'train'],
        [dev_data, 'dev'], 
        [test_data, 'test']]

    # Define CPU/GPU execution
    device = torch.device("cuda:"+str(cuda_no)
                          if torch.cuda.is_available() else "cpu")

    pretrained_emb_weights = pickle.load(open(
                        save_dir+data_ext+'/glove_vectors.tensor', 'rb'))
    word_attn = AttentionWordRNN(batch_size=batch_size,
                                 emb_weights=pretrained_emb_weights,
                                 embed_size=300,
                                 word_gru_hidden=gru_hidden_size,
                                 device=device).to(device)
    word_attn.load_state_dict(torch.load(
                    save_dir+model_ext+'/wordNet_states.pt'))

    sent_attn = AttentionSentRNN(batch_size=batch_size,
                                 sent_gru_hidden=gru_hidden_size,
                                 word_gru_hidden=gru_hidden_size,
                                 n_classes=classes,
                                 device=device).to(device)
    sent_attn.load_state_dict(torch.load(
                    save_dir+model_ext+'/eduNet_states.pt'))
    word_attn.eval()
    sent_attn.eval()
    word_attn.to(device)
    sent_attn.to(device)
    print("Model loaded...")
    class_weight_vector = [-1]
    for i in range(1, classes):
        class_weight_vector.append((2/(classes-1))+class_weight_vector[i-1])
    index2word = pickle.load(open(
                        save_dir+data_ext+'/glove_idx2word.dict', 'rb'))
    for dataset in datasets:
        if os.path.exists(save_dir+model_ext+'/edu_level_annotations_'+dataset[1]+'.h5') and not overwrite:
            print("The CKY data file edu_level_annotations_"+dataset[1]+".h5 already exists "
                  "and will not be created again. If you want "
                  "to generate the file again, please delete "
                  "the existing file.")
            continue
        elif os.path.exists(save_dir+model_ext+'/edu_level_annotations_'+dataset[1]+'.h5') and overwrite:
            os.remove(save_dir+model_ext+'/edu_level_annotations_'+dataset[1]+'.h5')
            print("Overwriting existing HDF5 file for", dataset[1])
        print("Processing edu level annotations for", dataset[1])
        hdf5_file_name = save_dir+model_ext+'/edu_level_annotations_'+dataset[1]+'.h5'
        hdf5_file = h5py.File(hdf5_file_name, 'w', libver='latest', swmr=True)
        for batch_idx, (doc_id, X, doc_len, _, y, edus) in enumerate(dataset[0]):
            X = X.transpose(0, 1)
            doc_sent, edu_attn, edu_sent = _evaluate_network(X, word_attn, sent_attn, device)
            doc_sent = doc_sent.squeeze(-1).tolist()
            doc_len = doc_len.tolist()
            doc_ids = doc_id.tolist()
            edu_attn = edu_attn.squeeze(-1).transpose(0, 1).tolist()
            edu_sent = edu_sent.tolist()
            edus = edus.tolist()
            y = y.tolist()
            for doc_idx in range(len(edu_attn)):
                # Remove documents with only a single EDU, 
                # as those don't help the prediction
                if doc_len[doc_idx] < 2:
                    continue
                document = [str(y[doc_idx]), str(class_weight_vector[y[doc_idx]]),
                            str(doc_sent[doc_idx]), str(class_weight_vector[doc_sent[doc_idx]]),
                            str(doc_ids[doc_idx]), 
                            str(doc_len[doc_idx])]
                for edu_idx in range(doc_len[doc_idx]):
                    tmp_edu = F.softmax(torch.tensor(
                                        edu_sent[doc_idx][edu_idx]), dim=-1).tolist()
                    polarity = 0
                    for idx, value in enumerate(tmp_edu):
                        polarity += class_weight_vector[idx]*value
                    tmp_text = ' '.join(
                                        [index2word[x] for x in
                                         edus[doc_idx][edu_idx] if x != 0])
                    tmp_text = re.sub(r'\.(\s*\.)*', '.', tmp_text)
                    tmp_text = tmp_text.strip()
                    if not tmp_text == '.':
                        if not tmp_text == '':
                            document.extend([str(tmp_text).encode('utf8'), 
                                             str(polarity).encode('utf8'), 
                                             str(edu_attn[doc_idx][edu_idx]).encode('utf8')])
                for idx, d in enumerate(document):
                    if type(d) == str:
                        document[idx] = d.encode('UTF-8')
                hdf5_file.create_dataset(str(doc_ids[doc_idx]).encode('UTF-8'), data=document)
        hdf5_file.close()
