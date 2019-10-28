
from torch.autograd import Variable
from torch.nn import functional as F
from gensim.models import FastText
from gensim.test.utils import datapath
import torch

import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


'''BI-LSTM with Softmax loss'''


class LSTMTagger_softmax(nn.Module):

    def __init__(self, config):
        super(LSTMTagger_softmax, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.tagset_size = config.tagset_size
        self.use_fasttext = config.use_fasttext
        if self.use_fasttext:
            self.embed_model = config.embedding_model
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.num_layers = config.num_layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, num_layers=self.num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)

    def forward(self, sentence):
        if self.use_fasttext:
            embeds = self.get_embeddings(sentence)
        else:
            embeds = self.word_embeddings(sentence)
        #print(embeds.view(len(sentence), 1, -1).shape)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        #print(tag_scores.shape)
        return tag_scores

    def get_embeddings(self, sentence):
        return torch.tensor(self.embed_model[sentence], dtype=torch.float)


'''BILSTM CRF'''


class LSTMTagger_CRF(nn.Module):

    def __init__(self, config):
        super(LSTMTagger_CRF, self).__init__()
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.tagset_size = config.tagset_size
        self.use_fasttext = config.use_fasttext
        if self.use_fasttext:
            self.embed_model = config.embedding_model
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = config.num_layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, bidirectional=True, num_layers=self.num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def forward(self, sentence, tags):
        if self.use_fasttext:
            embeds = self.get_embeddings(sentence)
        else:
            embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out)
        tag_scores = self.crf(tag_space, tags)
        return -1 * (tag_scores)      # returns negative loglikelihood


    def get_lstm_features(self, sentence):
        if self.use_fasttext:
            embeds = self.get_embeddings(sentence)
        else:
            embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out)

        return tag_space

    def crf_decode(self, feats):

        return self.crf.decode(feats)

    def get_embeddings(self, sentence):

        return torch.tensor(self.embed_model[sentence], dtype=torch.float)

'''CNN'''


class textCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.embedding_dim
        self.n_class = config.n_class
        self.embedding_model = config.embedding_model

        kernels = [3, 4, 5]
        kernel_number = [100, 100, 100]

        self.convs = nn.ModuleList([nn.Conv2d(1, number, (size, self.dim), padding=(size - 1, 0)) for (size, number) in
                                    zip(kernels, kernel_number)])
        self.dropout = nn.Dropout()
        self.out = nn.Linear(sum(kernel_number), self.n_class)

    def forward(self, x):
        x = self.get_embeddings(x).unsqueeze(0)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.out(x)
        tag_scores = F.log_softmax(x, dim=1)

        return tag_scores

    def get_embeddings(self, data):
        return torch.tensor(self.embedding_model[data])


'''Attention'''
'''From - https://mlwhiz.com/blog/2019/03/09/deeplearning_architectures_text_classification/'''

class Attention_layer(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention_layer, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class Attention_model(nn.Module):

    def __init__(self, config):
        super(Attention_model, self).__init__()
        drp = 0.1
        self.embed_model = config.embedding_model
        self.embed_size = config.embedding_dim
        self.maxlen = config.max_length
        self.num_layers = config.num_layer
        self.tagset_size = config.tagset_size

        self.lstm = nn.LSTM(self.embed_size, 10, bidirectional=True,num_layers=self.num_layers)
        #self.lstm2 = nn.GRU(128 * 2, 64, bidirectional=True, batch_first=True)

        self.attention_layer = Attention_layer(10*2, self.maxlen)

        self.linear = nn.Linear(10* 2, self.tagset_size)
        #self.relu = nn.ReLU()
        #self.out = nn.Linear(10, self.tagset_size)

    def forward(self, x):
        h_embedding = self.get_embeddings(x)
        h_embedding = torch.unsqueeze(h_embedding, 1)
        h_lstm, _ = self.lstm(h_embedding)    #add pack and unpack operations
        h_lstm_atten = self.attention_layer(h_lstm)
        conc = self.linear(h_lstm_atten)
        tag_scores = F.log_softmax(conc, dim=1)
        return tag_scores


    def get_embeddings(self,sentence):       
        return torch.tensor(self.embed_model[sentence],dtype=torch.float)
