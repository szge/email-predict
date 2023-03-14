from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier

import numpy as np
import numpy.typing as npt
import time

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


def find_best_classifier(output_dir, x_train, x_test, y_train, y_test) -> int:
    # Find the best classifier for the data
    print("Finding best classifier...")

    i_best = 0
    acc_best = 0

    classifiers = [
        GaussianNB,
        RandomForestClassifier,
        AdaBoostClassifier,
        # SGDClassifier,
        # MLPClassifier
    ]

    with open(f"{output_dir}/results.txt", "w") as outf:
        for idx, _class in enumerate(classifiers):
            print(f"Testing {_class.__name__}...")
            start_time = time.time()
            if _class == RandomForestClassifier:
                clf = RandomForestClassifier(max_depth=5, n_estimators=10)
            elif _class == MLPClassifier:
                clf = MLPClassifier(alpha=0.05)
            else:
                clf = _class()

            clf.fit(x_train, y_train)
            acc = clf.score(x_test, y_test)

            if acc > acc_best:
                i_best = idx
                acc_best = acc

            end_time = time.time()
            outf.write(f"{_class.__name__} accuracy: {acc}\n")
            print(f"{_class.__name__} accuracy: {acc} time: {round(end_time - start_time, 2)}")

    print(f"Best classifier: {classifiers[i_best].__name__} with accuracy {acc_best}")

    return i_best


def run_classifiers():
    data = np.load("npy/features.npy")
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=420)

    # Run classifiers on the data
    i_best = find_best_classifier("output", x_train, x_test, y_train, y_test)
