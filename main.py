from preproc import *
from create_npy_inputs import *
from classifiers import *

if __name__ == "__main__":
    preproc(mode=Mode.PARTIAL)
    create_npy_inputs()
    run_classifiers()
