
import jsonargparse
import torch
from dataloader import Load_Dataset
from models import models
from predict import predict
from models.models import Attention_layer,Attention_model
from utils import utils
from train import train
from speak import speak
import pickle


def load_test_data(config):
    data_dict__test = Load_Dataset.load_dataset(config, config.path_to_merged_xml_TEST)

    return data_dict__test

def run_predictions(model, config, test_data, tag_to_ix,file_name):
    p__, r__, f__ = predict.predict_for_all_epochs(model, config, test_data,
                                                   tag_to_ix)  # it will pick model from path in config
    prf = [p__,r__,f__]
    with open(config.path_to_saved_model + file_name, 'wb') as f:
        pickle.dump(prf, f)


def run_training(config):
    '''TRAIN'''
    data_dict_ = Load_Dataset.load_dataset(config,config.path_to_merged_xml_TRAIN)
    word_to_ix_train = data_dict_['word_to_ix']
    tag_to_ix_train = data_dict_['tag_to_ix']
    data = data_dict_['data']
    '''TEST'''
    data_dict_test = Load_Dataset.load_dataset(config, config.path_to_merged_xml_TEST)
    #word_to_ix_test = data_dict_test['word_to_ix']
    #tag_to_ix_test = data_dict_test['tag_to_ix']

    test_data = data_dict_test['data']
    #print('max_length test=',max())
    '''Load embedding model'''
    embedding_model = utils.load_embedding_model('model_complete.bin')


    if config.BILSTM_SOFTMAX:
        #print('running_Softmax')
        vocab_size = len(word_to_ix_train)  # fix
        tagset_size = len(tag_to_ix_train)
        #max_length = len(data[0][0])
        model = models.LSTMTagger_softmax(config,vocab_size,tagset_size,embedding_model)
        if config.train:

            train.train_loop(config, model, data, test_data , tag_to_ix_train, use_attention=False)    #starts training
            pass
        if config.predict:
            run_predictions(model,config,data,tag_to_ix_train,file_name='train_prf')
            run_predictions(model,config,test_data,tag_to_ix_train,file_name='test_prf')

    if config.BILSTM_CRF:
        vocab_size = len(word_to_ix_train)  # fix
        tagset_size = len(tag_to_ix_train)
        # max_length = len(data[0][0])
        model = models.LSTMTagger_CRF(config, vocab_size, tagset_size,embedding_model)
        if config.train:
            train.train_loop(config, model, data, test_data,  tag_to_ix_train, use_attention=False)  # starts training
            pass
        if config.predict:

            run_predictions(model, config, data, tag_to_ix_train, file_name='train_prf')
            run_predictions(model, config, test_data, tag_to_ix_train, file_name='test_prf')

    if config.ATTENTION:
        data_for_attention = utils.return_padded_data_dict(data_dict_)
        #word_to_ix_train = data_for_attention['word_to_ix']
        tag_to_ix_train = data_for_attention['tag_to_ix']
        tagset_size = len(tag_to_ix_train)
        data = data_for_attention['data']
        max_length = len(data[0][0])
        model = models.Attention_model(config,  tagset_size,max_length,embedding_model)
        train.train_loop(config, model, data, test_data, tag_to_ix_train, use_attention=True)  # starts training
        pass



if __name__ == '__main__':
  
  # yaml config 
    parser = jsonargparse.ArgumentParser()
    # Model
    '''Data type'''
    parser.add_argument('--benchmark_data', type=bool, help='Loads benchmark data, where one document is returned as one sequence')
    parser.add_argument('--ngrams', type=bool, help='Loads Ngrams from document, each document is read line wise and Ngrams of LINES are returned')
    parser.add_argument('--kmeans',  type=bool, help='Runs a Kmeans clustering over a document and returns clusters = N_CLUSTERS')
    parser.add_argument('--document_to_matrix', type=bool, help='Returns a list of documents as matrices and corresponding tag matrices')
    parser.add_argument('--linewise',type=bool)


    parser.add_argument('--path_to_merged_xml_TRAIN', help='Path to file containing paths to merged xml')
    parser.add_argument('--path_to_merged_xml_TEST', help='Path to file containing paths to merged xml')




    '''Parameters for data type'''
    parser.add_argument('--N_GRAMS', default=5, type=int, help='N-lines per Ngram')
    parser.add_argument('--N_CLUSTERS', default=15, type=int, help='Number of clusters per document')
    parser.add_argument('--dump_fasttext_file', default='/home/ankit/', help='Path to dump fast text data text file. Use wordsvectorcreation module to train fasttext data then save the model.bin in gensim site-package')


    '''LSTM'''
    parser.add_argument('--dropout',default=0.2, type=float, help='Dropout probablity for embedding and lstm layers')
    parser.add_argument('--embedding_dim', default=100,  help='Embedding dim of fasttext vectors')
    parser.add_argument('--hidden_dim', default=100,
                        help='The number of LSTM units for each direction in the Bi-LSTM.')
    parser.add_argument('--num_layer', default=1, type=int, help='Number of Bi-LSTM layers')
    parser.add_argument('--use_fasttext', type=bool, default=True)

    '''Training'''
    parser.add_argument('--num_epochs', default=100,type=int, help='Number of epochs to run the training')
    parser.add_argument('--learning_rate', default=0.001, type =float, help='Learning rate')

    '''LOSS'''
    parser.add_argument('--use_crf', type=bool,   help='Whether to use CRF loss, compatible with model = LSTMtagger_crf ')
    parser.add_argument('--use_softmax',  type=bool, help='Whether to use softmax over the outputs, compatible with model = LSTMTagger_softmax and Attention_model')

    '''Paths to save model'''
    parser.add_argument('--path_to_saved_model', help='Path to save the model checkpoints and training loss, precision recall and f1 score pickles')

    '''CNN specific'''
    parser.add_argument('--n_class', help='number of classes when model = CNN')

    '''Model'''
    parser.add_argument('--BILSTM_SOFTMAX',  type=bool, help = 'model options - BILSTM_softmax, BILSTM_crf, BILSTM_attention')    #add 2D CNN experiment
    parser.add_argument('--BILSTM_CRF',  type=bool, help = 'model options - BILSTM_softmax, BILSTM_crf, BILSTM_attention')
    parser.add_argument('--ATTENTION',  type=bool, help = 'model options - BILSTM_softmax, BILSTM_crf, BILSTM_attention')
    '''Train or predict'''
    parser.add_argument('--train', type=bool)
    parser.add_argument('--predict', type=bool)


    config = parser.parse_args()

    #speak.speak('starting experiments')

    run_training(config)



