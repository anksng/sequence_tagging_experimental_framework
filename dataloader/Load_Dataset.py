
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
import numpy as np
import itertools
import scipy
import matplotlib.pyplot as plt
import pagexml
from pagexmltools.process import page_region_with_ordered_textlines
import numpy as np
import os
import re
import nltk
import sys
import pagexmltools
import statistics
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from collections import Counter
from scipy.spatial import distance_matrix
import seaborn as sns; sns.set()
from torch import nn
import glob
import os
import csv
import sys
import pickle
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from nlp_classification.nlp_classification.pre_processor import gaussianize
from sklearn.cluster import KMeans






def read_page_xml_extract_text(xml_file_path, page_num):
    """Read page xml and extract text, coords


    Arguments:
        xml_file_path {String} -- Path of the file
    """

    pagexml = process_pdf_pagexml(xml_file_path)
    words, coords, labels = get_words_and_coords(pagexml, page_num)

    return words, coords, labels


def process_pdf_pagexml(input):
    try:
        """Processing of non-ground truth pdf page xmls.

        - For each page in a page xml
        * Reorder all TextLines in each page ignoring parent TextRegion.
        * Create a full page TextRegion.
        * Move all TextLines in new order to page TextRegion.
        - Relabel line IDs to ease preservation of their order.

        Args:
            input (str): The input page xml file path.

        Returns:
            PageXML object.
        """

        pxml = pagexml.PageXML()
        pxml.loadXml(input)
        pagexmltools.process.page_region_with_ordered_textlines(pxml, fake_baseline=True)
        return pxml
    except Exception as e:
        print(e)
        raise e


def get_words_and_coords(pxml, page_num):
    xpath_page = '//_:Page[%d]' % page_num
    try:
        words = []
        coords = []
        entity_label = []

        pg = pxml.selectNth(xpath_page)
        tokens = pxml.select('.//_:Word', pg)
        for token in tokens:
            # page_num=pxml.
            word = pxml.getTextEquiv(token)
            points = pxml.getPoints(token)
            label = pxml.getPropertyValue(token, 'entity')
            rectangle = np.array([points[0].x, points[0].y, points[2].x, points[2].y])
            rectangle = rectangle.astype(int)
            words.append(word)
            coords.append(rectangle)
            entity_label.append(label)

        return words, coords, entity_label
    except Exception as e:
        print(e)
        raise e





def read_data(uuid_path):
    result_df = pd.read_csv(uuid_path)
    result_df.columns = ['uuid']
    result_df['ocr_file_path'] = result_df.uuid.apply(lambda x: x + '/' + 'page.xml')
    result_df['text'] = result_df.ocr_file_path.apply(get_text)
    result_df = result_df.loc[result_df.text != '']
    return result_df


def read_uuids(path_to_merged_xml):
    '''Returns a dataframe with uuids given file path. Change the path in OPEN() to locate the merged_file_list'''
    with open(path_to_merged_xml,'r') as f:
        uuid=f.readlines()
    uuids=[]
    for i in uuid:
        uuids.append(i.strip())
    df_uuid=pd.DataFrame(data=uuids,columns=['uuids'],index=None)
    return df_uuid

def read_pagewise_data(path,page_num):
    word,coord,label=read_page_xml_extract_text(path,page_num)
    data= []
    for i in range(len(word)):
        data.append([word[i],coord[i],label[i]])
    return pd.DataFrame(data,columns=['word','coord','label'])

def abs_pos(x1,x2,y1,y2):
    x=(x1+x2)/2
    y = (y1+y2)/2
    return (x,y)

def abs_pos_dataframe(data):
    pos=[]
    for i in range(len(data)):
        df=data.coord[i][0],data.coord[i][1],data.coord[i][2],data.coord[i][3]
        pos_x=(abs_pos(df[0],df[2],df[1],df[3]))
        pos.append(pos_x)
    df_x=pd.DataFrame({'position':[pos]})
    return pd.DataFrame(df_x.position[0],columns=['x','y'])

def add_absolute_positions_main(df):
    for i in range(len(df)):
        df[i]=df[i].join(abs_pos_dataframe(df[i]))


def scatter_plot(data):
    x_coordinate = []
    y_coordinate = []
    tag_x = []
    tag_y = []
    for i in range(len(data)):
        if data.label[i] != 'O':
            tag_x.append(data.x[i])
            tag_y.append(data.y[i])
        else:
            x_coordinate.append(data.x[i])
            y_coordinate.append(data.y[i])
    plt.figure(figsize=(20, 13))
    #     plt.scatter(x_coordinate,y_coordinate)
    plt.scatter(x_coordinate, y_coordinate)

    plt.scatter(tag_x, tag_y)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def scatter_plot_annot(data):
    x_coordinate = []
    y_coordinate = []

    for i in range(len(data)):
        x_coordinate.append(data.x[i])
        y_coordinate.append(data.y[i])
    plt.figure(figsize=(20, 13))
    plt.scatter(x_coordinate, y_coordinate)

    for i, txt in enumerate(data.label):
        plt.annotate(txt, (x_coordinate[i], y_coordinate[i]))

    plt.show()

# Function to Normalizing the x and y :
def _normalize(x):
    return normalize(x[:,np.newaxis], axis=0).ravel()
def _normalize_main(df):
    for i in range(len(df)):
            try:
                df[i].x = _normalize(df[i].x)
                df[i].y = _normalize(df[i].y)
            except ValueError:
                pass


def get_sequences(train):
    seq = []
    buffer = []
    x_coord_buffer = []
    x_coord = []
    buffer_tag = []
    tags = []
    for i in range(len(train) - 1):
        if train.x[i + 1] > train.x[i]:
            buffer.append(train.word[i])
            x_coord_buffer.append(train.x[i])
            buffer_tag.append(train.label[i])

        else:
            x_coord_buffer.append(train.x[i])
            buffer.append(train.word[i])
            buffer_tag.append(train.label[i])
            x_coord.append(x_coord_buffer)
            seq.append(buffer)
            tags.append(buffer_tag)
            buffer = []
            buffer_tag = []
            x_coord_buffer = []

    return seq, tags, x_coord

def get_line_wise_main(df):
    all_data=[]
    for i in df :
        seq,tag,coords=get_sequences(i)
        df_ = pd.DataFrame({'sequences':seq,'tags':tag,'x_coord':coords})
        all_data.append(df_)
    return all_data


'''For Ngram sequences'''


def get_Ngrams(data, N):
    data_ = []

    for i in range(len(data)):
        seq = [j[0] for j in data[i:i + N]]
        seq_ = list(np.concatenate([i for i in seq]))
        tags = [j[1] for j in data[i:i + N]]
        tags_ = list(np.concatenate([i for i in tags]))
        assert len(seq_) == len(tags_)
        data_.append([seq_, tags_])
    return data_


def tuple_of_seq_tag(df):
    data = []
    for i in df:
        for j in range(len(i)):
            data.append([i.sequences.iloc[j], i.tags.iloc[j]])
    return data


'''main'''


def get_ngrams_main(df, N):
    data = tuple_of_seq_tag(df)
    data_grams = get_Ngrams(data, N)
    return data_grams


def mean(arr):
    return np.mean(arr)


def append_cluster_center(df):
    for j in df:
        list_means = [mean(i) for i in j.x_coord]
        j['cluster_centers'] = list_means
    return df


'''assigning y values to cluster centers'''


def get_ycoord(length):
    y_coord = np.arange(0, 1, 1 / length)
    return y_coord


def append_y_coord(df):
    for j in df:
        y_coord = get_ycoord(len(j))
        if len(y_coord) == len(j):
            j['y_coord'] = y_coord
        else:
            j['y_coord'] = y_coord[:-1]
    return df


def fit_kmeans(page, n_clusters, plot=False):
    '''
    Fits kmeans on the given page dataframe
    x = cluster_center (row-wise)
    y = np.arange(0,1,1/length)
    '''
    arr = np.array([page.x_y[i] for i in range(len(page))])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(arr)
    if plot == True:
        plt.figure(figsize=(15, 10))
        plt.scatter(arr[:, 0], arr[:, -1])

        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x')

        plt.title('Data points and cluster centroids')
        plt.show()
    return kmeans.labels_




def get_kmeans_sequnces_main(df, n_clusters):
    df = append_cluster_center(df)
    df = append_y_coord(df)
    for j in df:
        col = []
        for i in range(len(j)):
            col.append([j.cluster_centers[i], j.y_coord[i]])
        j['x_y'] = col

    labels_list = []
    for i in df:
        if len(i) < n_clusters:
            labels_list.append(fit_kmeans(i, n_clusters=len(i)))
        else:
            labels_list.append(fit_kmeans(i, n_clusters=n_clusters))

    sequences_ = []
    for k in range(len(df)):
        seq = []
        buff_seq = []
        for i in range(0, len(labels_list[k]) - 1):
            if labels_list[k][i + 1] == labels_list[k][i]:
                buff_seq.append(df[k].sequences[i])
            else:
                buff_seq.append(df[k].sequences[i + 1])
                seq.append(buff_seq)
                buff_seq = []
        length = 0
        for i in seq:
            length += len(i)
        seq.append(df[k].sequences[length:])
        sequences_.append(seq)

    sequences = []
    for k in sequences_:
        sequen = []

        for i in k:
            seq_ = []
            for j in i:
                for l in j:
                    seq_.append(l)
            sequen.append(seq_)
        sequences.append(sequen)

    tags_ = []
    for k in range(len(df)):
        seq = []
        buff_seq = []
        for i in range(0, len(labels_list[k]) - 1):
            if labels_list[k][i + 1] == labels_list[k][i]:
                buff_seq.append(df[k].tags[i])
            else:
                buff_seq.append(df[k].tags[i + 1])
                seq.append(buff_seq)
                buff_seq = []
        length = 0
        for i in seq:
            length += len(i)
        seq.append(df[k].tags[length:])
        tags_.append(seq)

    tags = []
    for k in tags_:
        sequen = []

        for i in k:
            seq_ = []
            for j in i:
                for l in j:
                    seq_.append(l)
            sequen.append(seq_)
        tags.append(sequen)

    train_seq = []
    for i in sequences:
        for j in i:
            train_seq.append(j)
    train_tags = []
    for i in tags:
        for j in i:
            train_tags.append(j)
    train_data = []
    for i in range(len(train_seq)):
        train_data.append((train_seq[i], train_tags[i]))
    return train_data


def get_benchmark(data):
    data_ = []
    for i in data:
        data_.append([list(i.word), list(i.label)])
    return data_

def basic_preprocessing(data):
    data_ = []
    for i in range(len(data)):
        data_.append([data[i][0],data[i][1]])

    sequences = [(i[0]) for i in data]
    '''Strip characters'''
    merged_sequences =  [' '.join(i.strip('/.:*°)^(><\|«■•»► ♦').lower() for i in j) for j in sequences]
    sequences = [i.split(' ') for i in merged_sequences]
    '''Replace any emty line with UNK'''
    for i in range(len(sequences)):
        for j in range(len(sequences[i])):
            if sequences[i][j]=='':
                sequences[i][j]='UNK'
    for i in range(len(data_)):
        data_[i][0] = sequences[i]
    return data_

def get_word_to_ix(data):
    word_to_ix = {}

    for sent, tags in data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print('Total unique words in dataset',len(word_to_ix))
    return word_to_ix


def get_tag2idx(data):
    tag2idx = {}

    for sent, tags in data:
        for word in tags:
            if word not in tag2idx:
                tag2idx[word] = len(tag2idx)
    print('Total unique Tags in dataset', len(tag2idx))
    return tag2idx


def dump_train_fast_text(file_path, data):
    with open(file_path, 'w') as f:
        for i in data:
            f.write(' '.join(j for j in i[0]) + ' ')


''''Main'''


def load_dataset(config, path_to_merged_xml):
    '''
       options : azp-booking-confirmation
                 azp-death-certificate
                 azp-proof-of-cancellation
    '''
    page_nums = np.arange(1, 15, 1)

    dfs_page = []
    df = read_uuids(path_to_merged_xml)
    for i in df.uuids:
        for j in page_nums:
            dfs_page.append(read_pagewise_data(i, page_num=j))
    indices = np.arange(13, len(dfs_page), 14)
    df_ = [dfs_page[i] for i in indices]  # use this for all but regions

    if config.benchmark_data:
        data_benchmark = get_benchmark(df_)
        preprocessed_df_ = basic_preprocessing(data_benchmark)
        word_to_ix = get_word_to_ix(preprocessed_df_)
        tag2idx = get_tag2idx(preprocessed_df_)
        data_dict = {'data': preprocessed_df_,
                     'word_to_ix': word_to_ix,
                     'tag_to_ix': tag2idx}
        dump_train_fast_text(config.dump_fasttext_file, preprocessed_df_)
        print('Fasttext training data at ->', config.dump_fasttext_file)


    else:
        add_absolute_positions_main(df_)
        _normalize_main(df_)
        df_ = get_line_wise_main(df_)

        if config.Ngrams:
            data_for_ngrams = get_ngrams_main(df_, N=config.N_GRAMS)
            preprocessed_df_ = basic_preprocessing(data_for_ngrams)
            word_to_ix = get_word_to_ix(preprocessed_df_)
            tag2idx = get_tag2idx(preprocessed_df_)
            data_dict = {'data': preprocessed_df_,
                         'word_to_ix': word_to_ix,
                         'tag_to_ix': tag2idx}
            dump_train_fast_text(config.dump_fasttext_file, preprocessed_df_)
            print('Fasttext training data at ->', config.dump_fasttext_file)

        if config.kmeans:
            data_kmeans = get_kmeans_sequnces_main(df_, n_clusters=config.N_CLUSTERS)
            preprocessed_df_ = basic_preprocessing(data_kmeans)
            word_to_ix = get_word_to_ix(preprocessed_df_)
            tag2idx = get_tag2idx(preprocessed_df_)
            data_dict = {'data': preprocessed_df_,
                         'word_to_ix': word_to_ix,
                         'tag_to_ix': tag2idx}
            dump_train_fast_text(config.dump_fasttext_file, preprocessed_df_)
            print('Fasttext training data at ->', config.dump_fasttext_file)

    return data_dict

