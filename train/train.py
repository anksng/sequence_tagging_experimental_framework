'''train loop'''
from utils import utils
import torch.nn as nn
import torch.optim as optim
import torch
import pickle
from predict import predict

#from models.models import LSTMTagger_softmax,LSTMTagger_CRF,Attention_layer,Attention_model


def train_loop(config, model, data, tag_to_ix,use_attention=False):
    if use_attention:
        loss_function = nn.NLLLoss(size_average=True,ignore_index=len(tag_to_ix))
    else:
        loss_function = nn.NLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    a = 0
    loss_values = []
    for epoch in range(config.num_epochs):
        running_loss = 0.0
        a += 1
        print('Running epoch',a)
        loss = []

        for sentence, tags in data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            # '''run this when not using fasttexxt'''
            # sentence_in = prepare_sequence(sentence,word_to_ix)
            sentence_in = sentence
            targets = utils.prepare_sequence(tags, tag_to_ix)
            if config.use_softmax:
                tag_scores = model(sentence_in)
                loss = loss_function(tag_scores, targets)
                loss.backward()
                running_loss = + loss.item()
                optimizer.step()

                torch.save(model.state_dict(), config.path_to_saved_model + '/model{}.pt'.format(epoch))
            loss_values.append(running_loss / len(data))
            if config.use_crf:
                targets = targets.view(targets.shape[0], 1)
                loss = model(sentence_in, targets)
                loss.backward()
                running_loss = + loss.item()
                optimizer.step()

                torch.save(model.state_dict(), config.path_to_saved_model + '/model{}.pt'.format(epoch))

            print('Precision, recall, F1score->',predict.predict(model, config, data, epoch,tag_to_ix))

            loss_values.append(running_loss / len(data))

    print('Saving models at->', config.path_to_saved_model)
    '''Save training_loss'''
    with open(config.path_to_saved_model + 'training_loss.pkl', 'wb') as f:
        pickle.dump(loss_values, f)
