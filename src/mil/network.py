import torch
import torch.nn.functional as F
from torch.autograd import Variable


class AttentionWordRNN(torch.nn.Module):
    def __init__(self, batch_size, emb_weights, embed_size, word_gru_hidden, device):
        super(AttentionWordRNN, self).__init__()
        self.batch_size = batch_size
        self.emb_weights = emb_weights
        self.embed_size = embed_size
        self.word_gru_hidden = word_gru_hidden
        self.device = device
        self.lookup = torch.nn.Embedding.from_pretrained(emb_weights, freeze=False)
        self.word_gru = torch.nn.GRU(embed_size, word_gru_hidden, bidirectional=True)
        self.word_lin_layer = torch.nn.Linear(2*word_gru_hidden, 2*word_gru_hidden)
        self.word_attn_lin = torch.nn.Linear(2*word_gru_hidden, 1, bias=False)

    def forward(self, embed, state_word):
        embedded = self.lookup(embed.to(self.device))
        output_word, state_word = self.word_gru(embedded, state_word)
        word_squish = self.word_lin_layer(output_word)
        word_squish = torch.sigmoid(word_squish)
        word_attn = self.word_attn_lin(word_squish)
        word_attn = torch.sigmoid(word_attn)
        word_attn_vectors = torch.sum(output_word * word_attn.expand_as(output_word), 0).unsqueeze(0)
        return word_attn_vectors, state_word, word_attn

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden)).to(self.device)


class AttentionSentRNN(torch.nn.Module):
    def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden, n_classes, device):
        super(AttentionSentRNN, self).__init__()
        self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.dropout = torch.nn.Dropout(p=.2)
        self.device = device

        self.sent_gru = torch.nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional=True)
        self.sent_lin_layer = torch.nn.Linear(2*word_gru_hidden, 2*word_gru_hidden)
        self.edu_class_lin_layer = torch.nn.Linear(2*word_gru_hidden, n_classes)
        self.sent_attn_lin = torch.nn.Linear(2*word_gru_hidden, 1, bias=False)

    def forward(self, word_attention_vectors, state_sent):
        output_sent, state_sent = self.sent_gru(word_attention_vectors, state_sent)
        sent_squish = self.sent_lin_layer(output_sent)
        sent_squish = torch.sigmoid(sent_squish)
        sent_attn = self.sent_attn_lin(sent_squish)
        sent_attn = torch.sigmoid(sent_attn)
        sentiment_values = torch.sigmoid(self.edu_class_lin_layer(self.dropout(output_sent.transpose(0, 1))))
        sent_attn_vectors = torch.sum(sentiment_values.transpose(0,1) 
                                      * sent_attn.expand_as(sentiment_values.transpose(0,1)), 0).unsqueeze(0)
        return F.log_softmax(sent_attn_vectors.squeeze(0), dim=1), state_sent, sent_attn, sentiment_values

    def init_hidden(self):
        return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden)).to(self.device)
