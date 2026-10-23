import numpy as np


def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == label) & (y_pred == label))
    fp = np.sum((y_true != label) & (y_pred == label))

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def recall_score(y_true, y_pred, label):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = np.sum((y_true == label) & (y_pred == label))
    fn = np.sum((y_true == label) & (y_pred != label))

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def f1_score(y_true, y_pred, label):
    precision = precision_score(y_true, y_pred, label)
    recall = recall_score(y_true, y_pred, label)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def calculate_all_metrics(y_true, y_pred, num_classes=10):
    accuracy = accuracy_score(y_true, y_pred)

    precision = []
    recall = []
    f1 = []

    for label in range(num_classes):
        precision.append(
            precision_score(y_true, y_pred, label)
        )

        recall.append(
            recall_score(y_true, y_pred, label)
        )

        f1.append(
            f1_score(y_true, y_pred, label)
        )

    precision_macro = np.mean(precision)
    recall_macro = np.mean(recall)
    f1_macro = np.mean(f1)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }