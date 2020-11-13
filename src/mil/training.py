from .data_format import DataFormat
import torch
import torch.nn.functional as F
import pickle
import os
from .network import AttentionWordRNN, AttentionSentRNN
import json
import time
from sklearn.metrics import f1_score, accuracy_score


def _data_loader(index_file, data_file,
                 cpu_workers, collate, shuffle=True):
    dataset = DataFormat(index_file, data_file)
    # Data already contains batches, therefore batch_size = 1
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1,
        shuffle=shuffle, collate_fn=collate,
        num_workers=cpu_workers)
    return data_loader


def _batch_prep(data):
    doc_id, X, doc_len, edu_len, y, edus = zip(*data)
    X = X[0].clone().detach()
    edus = edus[0].clone().detach()
    doc_len = doc_len[0].clone().detach()
    edu_len = edu_len[0].clone().detach()
    y = y[0].clone().detach()
    doc_id = doc_id[0].clone().detach()
    return (doc_id, X, doc_len, edu_len, y, edus)


def _train_network(mini_batch, targets, word_attn_model,
                   sent_attn_model, word_optimizer, sent_optimizer,
                   criterion, device):
    state_word = word_attn_model.init_hidden()
    state_sent = sent_attn_model.init_hidden()
    targets = targets.to(device)
    max_sents, batch_size, max_tokens = mini_batch.size()
    word_optimizer.zero_grad()
    sent_optimizer.zero_grad()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(
            mini_batch[i, :, :].transpose(0, 1),
            state_word)
        if (s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    y_pred, state_sent, _, _ = sent_attn_model(s, state_sent)
    loss = criterion(y_pred, targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(word_attn_model.parameters(), 0.2)
    torch.nn.utils.clip_grad_norm_(sent_attn_model.parameters(), 0.2)
    word_optimizer.step()
    sent_optimizer.step()
    # Apply L2 normalization
    word_attn_model.word_lin_layer.weight.data = F.normalize(
        word_attn_model.word_lin_layer.weight.data,
        p=2, dim=1, eps=1e-12, out=None)
    sent_attn_model.edu_class_lin_layer.weight.data = F.normalize(
        sent_attn_model.edu_class_lin_layer.weight.data,
        p=2, dim=1, eps=1e-12, out=None)
    sent_attn_model.sent_lin_layer.weight.data = F.normalize(
        sent_attn_model.sent_lin_layer.weight.data,
        p=2, dim=1, eps=1e-12, out=None)
    return loss


def _evaluate_network(mini_batch, word_attn_model, sent_attn_model,
                      device):
    state_word = word_attn_model.init_hidden()
    state_sent = sent_attn_model.init_hidden()
    max_sents, batch_size, max_tokens = mini_batch.size()
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn_model(
            mini_batch[i, :, :].transpose(0, 1),
            state_word)
        if (s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    y_pred, state_sent, attn, sent = sent_attn_model(s, state_sent)
    _, doc_sent_class = y_pred.topk(1, dim=1)
    return doc_sent_class, attn, sent


def train(save_dir, model_ext, data_ext,
          cpu_workers, cuda_no, batch_size, gru_hidden_size,
          classes, epochs, eval_freq, overwrite):
    if overwrite:
        if os.path.exists(save_dir + model_ext + '/wordNet_states.pt'):
            os.remove(save_dir + model_ext + '/wordNet_states.pt')
        if os.path.exists(save_dir + model_ext + '/eduNet_states.pt'):
            os.remove(save_dir + model_ext + '/eduNet_states.pt')
        if os.path.exists(save_dir + model_ext + '/wordNet_model.pt'):
            os.remove(save_dir + model_ext + '/wordNet_model.pt')
        if os.path.exists(save_dir + model_ext + '/eduNet_model.pt'):
            os.remove(save_dir + model_ext + '/eduNet_model.pt')
    else:
        if os.path.exists(save_dir + model_ext + '/wordNet_states.pt'):
            print("The model file already exists and "
                  + "will not be created again. If you want "
                  + "to generate the file again, please delete "
                  + "the existing file or use the --overwrite parameter.")
            return

    train_data = _data_loader(
        save_dir + data_ext + '/train_idx2data.dict',
        save_dir + data_ext + '/train_data.h5',
        cpu_workers=cpu_workers,
        collate=_batch_prep,
        shuffle=True)
    dev_data = _data_loader(
        save_dir + data_ext + '/dev_idx2data.dict',
        save_dir + data_ext + '/dev_data.h5',
        cpu_workers=cpu_workers,
        collate=_batch_prep,
        shuffle=True)
    # Loading preprocessed embedding weights from GloVe
    print('Loading GloVe embeddings...')
    pretrained_emb_weights = pickle.load(open(
        save_dir + data_ext + '/glove_vectors.tensor', 'rb'))
    # Define CPU/GPU execution
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(cuda_no) if cuda else "cpu")
    # Define neural network
    print('Instantiating neural network...')
    word_attn = AttentionWordRNN(
        batch_size=batch_size,
        emb_weights=pretrained_emb_weights,
        embed_size=300,
        word_gru_hidden=gru_hidden_size,
        device=device).to(device)

    sent_attn = AttentionSentRNN(
        batch_size=batch_size,
        sent_gru_hidden=gru_hidden_size,
        word_gru_hidden=gru_hidden_size,
        n_classes=classes,
        device=device).to(device)
    word_optimizer = torch.optim.Adadelta(word_attn.parameters())
    sent_optimizer = torch.optim.Adadelta(sent_attn.parameters())
    loss_fn = torch.nn.NLLLoss()

    model_f1_scores, model_acc_scores, model_loss = [], [], []
    # Save the hyper-parameters of the model, so it can be reinstanciated
    model_parameters = {'NUM_EPOCHS': epochs,
                        'BATCH_SIZE': batch_size,
                        'RNN_HIDDEN_SIZE': gru_hidden_size,
                        'SENTIMENT_CLASSES': classes}
    with open(save_dir + model_ext + '/model_parameters.json',
              'w') as file:
        json.dump(model_parameters, file)

    # TRAINING
    print('Starting training...')
    for epoch in range(1, epochs + 1):
        avg_epoch_loss = 0
        epoch_dev_predictions = []
        epoch_dev_labels = []
        nb_batches = train_data.__len__()

        # TRAIN
        # Set network to train mode
        word_attn.train()
        sent_attn.train()
        for batch_idx, (doc_id, X, doc_len, _, y, _) in enumerate(train_data):
            X = X.transpose(0, 1)
            # Transform input tensor into a long tensor
            X = X.clone().detach()
            # Execute one forward and backward pass
            avg_epoch_loss += _train_network(
                X, y, word_attn, sent_attn, word_optimizer,
                sent_optimizer, loss_fn, device).item()
            torch.cuda.empty_cache()
        avg_epoch_loss = avg_epoch_loss / nb_batches
        model_loss.append(avg_epoch_loss)
        with open(save_dir + model_ext + '/model_loss.json', 'w') as file:
            json.dump(model_loss, file)
        print('Average epoch loss in epoch', epoch, ':', avg_epoch_loss)

        # EVALUATE
        # If current epoch is a validation_epoch
        # Run forward pass on unseen dev data
        if (epoch % eval_freq) == 0:
            # Set network to evaluation mode
            word_attn.eval()
            sent_attn.eval()
            print('Evaluating on development set')
            for dev_batch_idx, (doc_id, X, doc_len, _, y, _) in enumerate(dev_data):
                X = X.transpose(0, 1)
                tmp_pred, _, _ = _evaluate_network(X, word_attn, sent_attn, device)
                epoch_dev_predictions.extend([int(i) for i in tmp_pred])
                epoch_dev_labels.extend([int(i) for i in y])
            # Calculate f1-score for the complete dev dataset
            epoch_f1 = f1_score(epoch_dev_labels,  epoch_dev_predictions, average='macro')
            print('Average epoch f1-score on dev in epoch', epoch, ':', epoch_f1)
            # Calculate accuracy-score for the complete dev dataset
            epoch_acc = accuracy_score(epoch_dev_labels, epoch_dev_predictions)
            print('Average epoch accuracy-score on dev in epoch', epoch, ':', epoch_acc)
            if (len(model_f1_scores) == 0 or epoch_f1 > max(model_f1_scores)):
                # Only keep model with highest F1 score
                torch.save(word_attn.state_dict(), save_dir + model_ext + '/wordNet_states.pt')
                torch.save(sent_attn.state_dict(), save_dir + model_ext + '/eduNet_states.pt')
                torch.save(word_attn, save_dir + model_ext + '/wordNet_model.pt')
                torch.save(sent_attn, save_dir + model_ext + '/eduNet_model.pt')
            model_f1_scores.append(epoch_f1)
            model_acc_scores.append(epoch_acc)
            with open(save_dir + model_ext + '/model_dev_scores.json', 'w') as file:
                json.dump(list(zip(odel_acc_scores, model_f1_scores)), file)
    print('Finishing training...')
