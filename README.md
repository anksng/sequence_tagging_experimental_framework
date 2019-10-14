# sequence_tagging_experimental_framework
An experimental framework for sequence tagging task. Consists of BILSTM-softmax, BILSTM-CRF, BILSTM-CNN-CRF and attention model to empirically compare performance and analyse results

# Code structure

* **dataloader** - Loads annotated documents from xml files
* **models**  - Contains Deep learning models for experiments
* **train**  - Run training by selecting one of the models  
* **predict** -  Run predictions using saved PATH_TO/model checkpoints
* **saved_models**  - Placeholder for saving models
* **train_fasttext_data**  - Placeholder for fast text training data
* **utils**  - Utility functions 




# Usage example

The framework funcitons mentioned above has the following options - 
* dataloader - returns full sequences (one document = one sequence)-> *--BENCHMARK*, Kmeans clustered sequences on document level -> *--KMEANS* (default K = 10), Ngram sequences -> *--NGRAMS* and padded sequences for attention experiment -> *--ATTENTION_DATA* .

* Models - model names-> *--BILSTM_SOFTAMAX, --BILSTM_CRF, --CNN, --ATTENTION*

## Usage in terminal
To run an experiment use the following command while in the root folder of the project ->
### Training
```
python main.py --MODEL_NAME --DATATYPE --PATH_TO_DOCUMENT_XMLS --EMBEDDING DIM 100 --HIDDEN_DIM 10 --NUM_LAYERS 1 --EMBEDDING_MODEL model_name.bin --use_fattext TRUE --num_epochs 100 --learning_rate 0.001 --PATH_TO_SAVED_MODEL /saved_models --TRAIN TRUE
```
## Predict - Dumps predictions and metrics at -> --PATH_TO_SAVED_MODEL
```
python main.py --MODEL_NAME --DATATYPE --PATH_TO_DOCUMENT_XMLS_TEST_SET --EMBEDDING DIM 100 --HIDDEN_DIM 10 --NUM_LAYERS 1 --EMBEDDING_MODEL model_name.bin --use_fattext TRUE --num_epochs 100 --learning_rate 0.001 --PATH_TO_SAVED_MODEL /saved_models --PREDICT TRUE
```
