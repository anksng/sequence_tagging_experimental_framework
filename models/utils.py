
from gensim.models import FastText
from gensim.test.utils import datapath




'''Utils'''
def load_fasttext_model(path_to_bin_file='model_complete.bin'):   #model.bin - for complete sequences # This bin file is stored in gensim site package
    cap_path = datapath(path_to_bin_file)
    emd_model = FastText.load_fasttext_format(cap_path)
    return emd_model