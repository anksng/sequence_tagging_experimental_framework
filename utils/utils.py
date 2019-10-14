import torch
from gensim.models import FastText
from gensim.test.utils import datapath
from torch.nn.utils.rnn import pad_sequence
import pickle

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def load_embedding_model(path_to_bin_file):   # This bin file is stored in gensim site package
    cap_path = datapath(path_to_bin_file)
    emd_model = FastText.load_fasttext_format(cap_path)
    return emd_model

def load_pickle(file):
    with open(file,'rb') as f:
        data=pickle.load(f)
    return data

def invert_mapping(dict):
    return {v: k for k, v in dict.items()}

def word_to_index(word,word_to_ix):
    return word_to_ix[word]

def index_to_word(index,index_to_word):
    return index_to_word[index]


def return_padded_data_dict(data_dict_):
    '''This data pre processing for attention experiments'''
    try:
        data = data_dict_['data']

        word_to_ix = data_dict_['word_to_ix']
        # print(len(word_to_ix))
        word_to_ix['<pad>'] = len(word_to_ix) + 1
        # print(word_to_ix['<pad>'])
        tag_to_ix = data_dict_['tag_to_ix']
        tag_to_ix['<pad>'] = len(tag_to_ix) + 1

        sequences = []
        seq_lengths = []  # for packing and unpacking in lstm
        for i in data:
            seq = []
            seq_lengths.append(len(i[0]))
            for j in i[0]:
                seq.append(word_to_index(j, word_to_ix))
            sequences.append(torch.tensor(seq))
        tags = []
        for i in data:
            tag = []
            for j in i[1]:
                tag.append(word_to_index(j, tag_to_ix))
            tags.append(torch.tensor(tag))

        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=len(word_to_ix))

        padded_tags = pad_sequence(tags, batch_first=True, padding_value=len(tag_to_ix))
        '''invert mapping'''
        index2word = invert_mapping(word_to_ix)
        index2tag = invert_mapping(tag_to_ix)
        padded_sequences_words = []
        for i in padded_sequences:
            seq = []

            for j in i:
                seq.append(index_to_word(int(j), index2word))
            padded_sequences_words.append(seq)
        padded_tags_words = []
        for i in padded_tags:
            seq = []

            for j in i:
                seq.append(index_to_word(int(j), index2tag))
            padded_tags_words.append(seq)

        data = []
        for i in range(len(padded_sequences_words)):
            data.append([padded_sequences_words[i], padded_tags_words[i]])

        data_dict_to_return = {'data': data,
                               'word_to_ix': word_to_ix,
                               'tag_to_ix': tag_to_ix,
                               'sequence_lengths': seq_lengths}
    except KeyError:
        print('Reload data from data loader before calling return_padded_data_dict')

    return data_dict_to_return