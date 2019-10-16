# sequence_tagging_experimental_framework
An experimental framework for sequence tagging task. Consists of BILSTM-softmax, BILSTM-CRF, BILSTM-CNN-CRF and attention model to empirically compare performance and analyse results

# Dependencies
gensim==3.8.0 //
nltk==3.4.5 //
torch==1.2.0 //
Keras==2.2.4 //
python-crfsuite==0.9.6 //
pytorch-crf==0.7.2 //
jsonargparse==2.2.1 //


# Code structure

* **dataloader** - Loads annotated documents from xml files
* **models**  - Contains Deep learning models namely viz. BILSTM with softmax activation, BILSTM CRF, Attention, CNN for context feature extraction
* **train**  - Run training by selecting one of the models
* **predict** -  Run predictions using saved checkpoints at PATH_TO/model_checkpoints 
* **saved_models**  - Placeholder for saving models
* **train_fasttext_data**  - Placeholder for fast text training data
* **utils**  - Utility functions 




# Usage example

The framework functions mentioned above has the following options - 
* dataloader - returns full sequences (one document = one sequence)-> `--BENCHMARK`, Kmeans clustered sequences on document level -> `--KMEANS` (default K = 10), Ngram sequences -> `--NGRAMS` and padded sequences for attention experiment -> `--ATTENTION_DATA` .

* Models - model names-> `--BILSTM_SOFTAMAX`, `--BILSTM_CRF`, `--CNN`, `--ATTENTION`

## Usage in terminal
To run an experiment use the following command while in the root folder of the project ->
### Training
To train using default parameters, the following command can be used:
```
python main.py --MODEL_NAME --DATATYPE --PATH_TO_DOCUMENT_XMLS /path/to/documents --TRAIN TRUE
```
For e.g. to start a BILSTM_CRF experiment with Kmeans sequences, when the documents are stored at /home/user/folder/list_of_file_paths.txt use the following:
```
python main.py --BILSTM_CRF --KMEANS --PATH_TO_DOCUMENT_XMLS /home/user/folder/list_of_file_paths.txt  --TRAIN TRUE
```

To specify all the parameters, following use the following options:
```
python main.py --MODEL_NAME --DATATYPE --PATH_TO_DOCUMENT_XMLS /path/to/documents --EMBEDDING DIM 100 --HIDDEN_DIM 10 --NUM_LAYERS 1 --EMBEDDING_MODEL model_name.bin --use_fattext TRUE --num_epochs 100 --learning_rate 0.001 --PATH_TO_SAVED_MODEL /saved_models --TRAIN TRUE
```
## Predict - Dumps predictions and metrics at -> --PATH_TO_SAVED_MODEL

```
python main.py --MODEL_NAME --DATATYPE --PATH_TO_DOCUMENT_XMLS_TEST_SET /path/to/documents --PATH_TO_SAVED_MODEL /saved_models --PREDICT TRUE
```
