
#from utils import utils
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import torch, torchcrf
from torchcrf import CRF
def get_tags(tag_scores):
    tags=[]
    for i in tag_scores:
        tags.append(np.argmax(i))
    return np.array(tags)

def flatten_array(list_arr):
    arr=[]
    for i in list_arr:
        for j in i:
            arr.append(j)
    return arr

def load_model(model,path_to_state_dict):
    model.load_state_dict(torch.load(path_to_state_dict))
    model.eval()
    return model


def write_predictions(model,data,tag2idx,config):
    predictions=[]
    true=[]
    if config.use_softmax:
        for sentence,tags in data:
            with torch.no_grad():
                inputs =sentence
                tag_scores = model(inputs)
                pred = [i for i in get_tags(tag_scores)]
                predictions.append(pred)
                tru_tags=[tag2idx.get(i) for i in tags]
                true.append(tru_tags)
    if config.use_crf:
        for sentence,tags in data:
            with torch.no_grad():
                inputs = sentence
                lstm_feats= model.get_lstm_features(inputs)
                pred = model.crf_decode(lstm_feats)
                predictions.append(pred[0])
                tru_tags = [tag2idx.get(i) for i in tags]
                true.append(tru_tags)

    '''Flatten'''
    true=flatten_array(true)
    pred=flatten_array(predictions)
    assert len(true)==len(pred)
    return true,pred


def predict_metrics(true,
                    pred,
                    predict_for,
                    average='macro'):
    results = precision_recall_fscore_support(true, pred, average='macro', labels=predict_for)
    precision, recall, fscore, support = results
    return precision, recall, fscore, support


def predict(model, config, data,epoch, tag2idx):
    model_ = load_model(model, config.path_to_saved_model + 'model' + str(epoch) + '.pt')
    true, pred = write_predictions(model_, data,tag2idx,config)

    tag_list = list(tag2idx.values())[1:]  # should except 0
    assert list(tag2idx.values())[0] == 0
    precision, recall, fscore, support = predict_metrics(true, pred, predict_for=tag_list)
    return precision, recall, fscore, support


def predict_for_all_epochs(model, config, data, tag2idx):
    precision_ = []
    recall_ = []
    fscore_ = []
    epochs = np.arange(0, config.num_epochs, 1)

    for i in epochs:
        precision, recall, fscore, support = predict(model, config, i, data, tag2idx)
        precision_.append(precision)
        recall_.append(recall)
        fscore_.append(fscore)
    return precision_, recall_, fscore_