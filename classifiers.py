from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB

import numpy as np
import numpy.typing as npt
import time

import matplotlib.pyplot as plt

from helper import *

np.random.seed(420)


def accuracy(c: npt.NDArray[np.float64]) -> float:
    """Compute accuracy given Numpy array confusion matrix c. Returns a floating point value"""
    return sum([c[i][i] for i in range(len(c))]) / sum([c[i][j] for i in range(len(c)) for j in range(len(c[0]))])


def recall(c: npt.NDArray[np.float64]) -> list[float]:
    """Compute recall given Numpy array confusion matrix C. Returns a list of floating point values"""
    return [c[k][k] / sum([c[k][j] for j in range(len(c[0]))]) for k in range(len(c))]


def precision(c: npt.NDArray[np.float64]) -> list[float]:
    """Compute precision given Numpy array confusion matrix C. Returns a list of floating point values"""
    return [c[k][k] / sum([c[i][k] for i in range(len(c[0]))]) for k in range(len(c))]


def find_best_classifier(x_train, x_test, y_train, y_test) -> int:
    # Find the best classifier for the data
    print("Finding best classifier...")

    i_best = 0
    acc_best = 0

    classifiers = [
        GaussianNB,
        ComplementNB,
        RandomForestClassifier,
        AdaBoostClassifier,
        BaggingClassifier,
        # KNeighborsClassifier,
    ]

    # clear results file
    with open("e_output/results.txt", "w") as outf:
        outf.write("")

    for idx, _class in enumerate(classifiers):
        start_time = time.time()
        if _class == RandomForestClassifier:
            clf = RandomForestClassifier(max_depth=5, n_estimators=10)
        elif _class == MLPClassifier:
            clf = MLPClassifier(alpha=0.05)
        else:
            clf = _class()

        print(f"Training {_class.__name__}...")
        clf.fit(x_train, y_train)
        print(f"Testing {_class.__name__}...")
        y_pred = clf.predict(x_test)
        cm = confusion_matrix(y_test, y_pred, normalize="all")
        acc = accuracy(cm)

        if acc > acc_best:
            i_best = idx
            acc_best = acc

        end_time = time.time()
        print(f"{_class.__name__} accuracy: {round(acc, 4)} time: {round(end_time - start_time, 2)}")
        save_results(_class.__name__, cm)

    print(f"Best classifier: {classifiers[i_best].__name__} with accuracy {round(acc_best, 2)}")

    return i_best


def save_results(model_name: str, cm: any):
    acc = accuracy(cm)
    class_acc = cm.diagonal()/cm.sum(axis=1)
    with open("e_output/results.txt", "a") as outf:
        outf.write(f"{model_name} accuracy: {round(acc, 4)}\n")
        for i in range(len(class_acc)):
            outf.write(f"{evt_names[i]} accuracy: {round(class_acc[i], 4)}\n")
        outf.write("\n")
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=evt_names)
    disp.plot()
    # remove values from confusion matrix
    for i in range(len(evt_codes)):
        for j in range(len(evt_codes)):
            disp.text_[i, j].set_text("")
    # increase size of confusion matrix
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    # make x axis labels vertical
    plt.xticks(rotation=90)
    # padding
    plt.tight_layout()
    plt.savefig(f"e_output/{model_name}.png")


def run_classifiers():
    data = np.load("d_npy/features.npy")
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=420)

    # Run classifiers on the data
    i_best = find_best_classifier(x_train, x_test, y_train, y_test)
