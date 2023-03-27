from preproc import *
from create_npy_inputs import *
from order_events import *
from classifiers import *

if __name__ == "__main__":
    # preproc(mode=Mode.PARTIAL)
    # preproc(mode=Mode.FULL)
    # order_events()
    create_npy_inputs()
    # run_classifiers()
