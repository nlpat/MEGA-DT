from .training import _data_loader, _batch_prep, _evaluate_network
import torch
import pickle
from .network import AttentionWordRNN, AttentionSentRNN
from sklearn.metrics import f1_score, accuracy_score
import json
import numpy as np
import torch.nn.functional as F


def _evaluate_ternary_classes(data_pred, data_label, samples):
    numerize_ratings = {'+': 3, '0': 2, '-': 1}
    threshold_1, threshold_2 = min(data_pred), max(data_pred)
    step_size = (threshold_2-threshold_1)/samples
    best_t1, best_t2 = threshold_1, threshold_1+1
    best_f1 = 0
    if threshold_1 == threshold_2:
        discrete_data_pred = []
        for datapoint in data_pred:
            if datapoint <= threshold_1:
                discrete_data_pred.append(numerize_ratings['-'])
            elif datapoint >= threshold_2:
                discrete_data_pred.append(numerize_ratings['+'])
            else:
                discrete_data_pred.append(numerize_ratings['0'])
        spot_f1 = f1_score(
            data_label, [float(i) for i in discrete_data_pred], average='macro')
        best_f1 = spot_f1
        best_t1 = threshold_1
        best_t2 = threshold_2
    else:
        for t1 in np.arange(threshold_1, threshold_2, step_size):
            for t2 in np.arange(t1, threshold_2, step_size):
                discrete_data_pred = []
                for datapoint in data_pred:
                    if datapoint <= t1:
                        discrete_data_pred.append(numerize_ratings['-'])
                    elif datapoint >= t2:
                        discrete_data_pred.append(numerize_ratings['+'])
                    else:
                        discrete_data_pred.append(numerize_ratings['0'])
                spot_f1 = f1_score(
                            data_label, discrete_data_pred, average='macro')
                if spot_f1 > best_f1:
                    best_f1 = spot_f1
                    best_t1 = t1
                    best_t2 = t2
    return best_f1, best_t1, best_t2

def test_doc_lvl(save_dir, data_ext, model_ext, cpu_workers,
                 cuda_no, batch_size, gru_hidden_size, classes):
    print('Evaluating model on document level...')
    test_data = _data_loader(
                       save_dir+data_ext+'/test_idx2data.dict',
                       save_dir+data_ext+'/test_data.h5',
                       cpu_workers=cpu_workers,
                       collate=_batch_prep,
                       shuffle=True)

    # Define CPU/GPU execution
    device = torch.device("cuda:"+str(cuda_no) if torch.cuda.is_available() else "cpu")
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

    # Run evaluation
    test_predictions, test_labels = [], []
    for test_batch_idx, (doc_id, X, doc_len, _, y, _) in enumerate(test_data):
        X = X.transpose(0, 1)
        tmp_pred,_,_ = _evaluate_network(
                                        X, word_attn, sent_attn, device)
        test_predictions.extend([int(i) for i in tmp_pred])
        test_labels.extend([int(i) for i in y])
    test_acc = accuracy_score(test_labels, test_predictions)
    print('Average test accuracy-score:', round(test_acc,2)*100, '%')

def test_edu_lvl(save_dir, data_ext, model_ext, cpu_workers,
                 cuda_no, batch_size, gru_hidden_size, classes):
    print('Evaluating model on EDU level...')
    spot_data = _data_loader(
                       save_dir+data_ext+'/edu_test_idx2data.dict',
                       save_dir+data_ext+'/edu_test_data.h5',
                       cpu_workers=cpu_workers,
                       collate=_batch_prep,
                       shuffle=True)

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

    # Run evaluation
    spot_prediction_doc_level, spot_prediction_edu = [], []
    spot_attention, spot_labels, doc_lengths = [], [], []
    for spot_batch_idx, (doc_id, X, doc_len, _, y, _) in enumerate(spot_data):
        X = X.transpose(0, 1)
        doc_class, attn, sent = _evaluate_network(
                                    X, word_attn, sent_attn, device)
        attn = attn.transpose(0, 1)
        spot_prediction_doc_level.extend(doc_class)
        spot_prediction_edu.extend(sent)
        spot_attention.extend(attn)
        spot_labels.extend(y)
        doc_lengths.extend(doc_len)
    _show_results(spot_prediction_doc_level, spot_prediction_edu,
                       spot_attention, spot_labels, doc_lengths, classes)

def _show_results(spot_prediction_doc_level, spot_prediction_edu,
                  spot_attention, spot_labels, doc_lengths, classes):

    pred, labels = [], []
    for idx, length in enumerate(doc_lengths):
        pred.extend([spot_prediction_doc_level[idx]]*int(length))
        labels.extend([int(i) for i in spot_labels[idx][:length]])
    pred = [np.float(i) for i in pred]
    labels = [np.float(i) for i in labels]
    f1_score_result, thres1, thres2 = _evaluate_ternary_classes(
                                                pred, labels, samples=100)
    print('F1-score using document-level sentiment only:',
          round(f1_score_result, 4)*100, '%')

    pred, labels = [], []
    for idx, length in enumerate(doc_lengths):
        tmp = spot_prediction_edu[idx][:length]
        class_weight_vector = [-1]
        for i in range(1, classes):
            class_weight_vector.append((2/(classes-1))+class_weight_vector[i-1])
        tmp_new = []
        for edu in tmp:
            weight_sum = 0
            for idx2, element in enumerate(edu):
                weight_sum += class_weight_vector[idx2]*element
            tmp_new.append(weight_sum)
        pred.extend(tmp_new)
        labels.extend([int(i) for i in spot_labels[idx][:length]])
    pred = [np.float(i) for i in pred]
    labels = [np.float(i) for i in labels]
    f1_score_result, thres1, thres2 = _evaluate_ternary_classes(
                                                pred, labels, samples=100)
    print('F1-score using edu level sentiment:', round(f1_score_result, 4)*100,
          '%, between', round(thres1, 2),
          'and', round(thres2, 2))

    pred, labels = [], []
    for idx, length in enumerate(doc_lengths):
        tmp = spot_prediction_edu[idx][:length]
        tmp_attn = spot_attention[idx]
        class_weight_vector = [-1]
        for i in range(1, classes):
            class_weight_vector.append((2/(classes-1))+class_weight_vector[i-1])
        tmp_new = []
        for idx_attn, edu in enumerate(tmp):
            weight_sum = 0
            for idx2, element in enumerate(edu):
                weight_sum += class_weight_vector[idx2]*element
            tmp_new.append(weight_sum*tmp_attn[idx_attn])
        pred.extend(tmp_new)
        labels.extend([int(i) for i in spot_labels[idx][:length]])
    pred = [np.float(i) for i in pred]
    labels = [np.float(i) for i in labels]
    f1_score_result, thres1, thres2 = _evaluate_ternary_classes(
                                        pred, labels, samples=100)
    print('F1-score using edu level sentiment with attention:', round(f1_score_result, 4)*100,
          '%, between', round(thres1, 5), 'and', round(thres2, 5))

    pred, labels = [], []
    for idx, length in enumerate(doc_lengths):
        tmp = spot_prediction_edu[idx][:length]
        tmp = F.softmax(tmp, dim=-1)
        tmp_attn = spot_attention[idx]
        class_weight_vector = [-1]
        for i in range(1, classes):
            class_weight_vector.append((2/(classes-1))+class_weight_vector[i-1])
        tmp_new = []
        for idx_attn, edu in enumerate(tmp):
            weight_sum = 0
            for idx2, element in enumerate(edu):
                weight_sum += class_weight_vector[idx2]*element
            tmp_new.append(weight_sum*tmp_attn[idx_attn])
        pred.extend(tmp_new)
        labels.extend([int(i) for i in spot_labels[idx][:length]])
    pred = [np.float(i) for i in pred]
    labels = [np.float(i) for i in labels]

    f1_score_result, thres1, thres2 = _evaluate_ternary_classes(
                                        pred, labels, samples=100)
    print('F1-score using softmax(edu) sentiment with attention:',
          round(f1_score_result, 4)*100, '%, between',
          round(thres1, 5), 'and', round(thres2, 5))
