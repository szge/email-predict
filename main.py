from preproc import *
from create_npy_inputs import *
from classifiers import *
from bert_finetune import *

if __name__ == "__main__":
    preproc()
    create_npy_inputs()
    run_classifiers()
    finetune_bert_model()
    
