from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.inspection import PartialDependenceDisplay

import numpy as np
import numpy.typing as npt
import time

import matplotlib.pyplot as plt

from helper import *
from create_npy_inputs import feature_names

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


def find_best_classifier(x_train, x_test, y_train, y_test) -> None:
    # Find the best classifier for the data
    print("Finding best classifier...")

    i_best = 0
    acc_best = 0

    classifiers = [
        GaussianNB,
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
            clf = RandomForestClassifier(max_depth=10, n_estimators=10)
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


def rfc_hyperparameter_tuning(x_train, x_test, y_train, y_test) -> dict:
    print("RandomForestClassifier hyperparameter tuning...")
    # hyperparameter tuning
    best_hyperparameters = {
        "max_depth": 0,
        "n_estimators": 0,
        "accuracy": 0
    }
    for i in [1, 5, 10]:
        for j in [1, 5, 10]:
            clf = RandomForestClassifier(max_depth=i, n_estimators=j)
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            cm = confusion_matrix(y_test, y_pred, normalize="all")
            acc = accuracy(cm)
            print(f"max_depth: {i} n_estimators: {j} accuracy: {round(acc, 4)}")
            save_results(f"RandomForestClassifier_max_depth_{i}_n_estimators_{j}", cm, draw_cm=False)
            if acc > best_hyperparameters["accuracy"]:
                best_hyperparameters["max_depth"] = i
                best_hyperparameters["n_estimators"] = j
                best_hyperparameters["accuracy"] = acc

    return best_hyperparameters


def save_results(model_name: str, cm: any, draw_cm: bool = True):
    acc = accuracy(cm)
    class_acc = cm.diagonal()/cm.sum(axis=1)
    with open("e_output/results.txt", "a") as outf:
        outf.write(f"{model_name} accuracy: {round(acc, 4)}\n")
        for i in range(len(class_acc)):
            outf.write(f"{class_labels[i]} accuracy: {round(class_acc[i], 4)}\n")
        outf.write("\n")

    if draw_cm:
        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
        disp.plot()
        # remove values from confusion matrix
        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                disp.text_[i, j].set_text("")
        # increase size of confusion matrix
        fig = plt.gcf()
        fig.set_size_inches(10, 10)
        # make x axis labels vertical
        plt.xticks(rotation=90)
        # padding
        plt.tight_layout()
        plt.savefig(f"e_output/{model_name}.png")
        plt.close()


def run_classifiers():
    data = np.load("d_npy/features.npy")

    # split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.7, random_state=420)

    # split testing data into validation and testing
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=420)
    # 70% training, 15% validation, 15% testing

    # Run classifiers on the data
    find_best_classifier(x_train, x_val, y_train, y_val)
    
    #subset to first 12 features since we are not using embeddings for this part
    x_train_subset = x_train[:,:12]
    x_val_subset = x_val[:,:12]
    x_test_subset = x_test[:,:12]
    
    # hyperparameter tuning
    best_hyperparameters = rfc_hyperparameter_tuning(x_train_subset, x_val_subset, y_train, y_val)
    # run the best hyperparameters on testing data
    clf = RandomForestClassifier(max_depth=best_hyperparameters["max_depth"], n_estimators=best_hyperparameters["n_estimators"])
    clf.fit(x_train_subset, y_train)
    y_pred = clf.predict(x_test_subset)
    cm = confusion_matrix(y_test, y_pred, normalize="all")
    acc = accuracy(cm)
    print(f"RandomForestClassifier best hyperparameters accuracy on test data: {round(acc, 4)}")
    save_results(f"RandomForestClassifier_best_hyperparameters", cm)

    #feature importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    #Print the feature ranking 
    print("Feature ranking:")

    with open("e_output/results.txt", "a") as outf:
        for f in range(12):
            feature_string = f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]})"
            print(feature_string)
            outf.write(feature_string + "\n")

    # plot_partial_dependence(clf, x_train_subset, indices[:5], feature_names=feature_names, n_jobs=4)
    display = PartialDependenceDisplay.from_estimator(clf, x_train_subset, indices[:5], feature_names=feature_names)
    display.plot()
    plt.savefig(f"e_output/feature_importance.png")
    plt.close()
