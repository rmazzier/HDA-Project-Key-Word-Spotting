import numpy as np
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def one_class_accuracy(target_class_int, y_pred, y_test):
    class_indexes_pred = np.where(y_pred==target_class_int)
    class_indexes_test = np.where(y_test==target_class_int)

    one_class_pred = np.zeros(len(y_pred))
    one_class_pred[class_indexes_pred] = 1

    one_class_test = np.zeros(len(y_test))
    one_class_test[class_indexes_test] = 1

    test_acc_one_class = sum(one_class_pred == one_class_test) / len(one_class_test)
    return test_acc_one_class

def get_fpr_fnr_aucs(y_true, y_scores, output_classes):
    binarized = label_binarize(y_true,classes=range(len(output_classes)))
    # SEPARATE ROC CURVE FOR EACH CLASS
    # in my case, i want false negative rate instead of true positive rate!
    # FNR = 1 - TPR

    fpr = dict()
    fnr = dict()
    roc_auc = dict()
    for i in range(len(output_classes)):
        fpr[i], tpr, _ = roc_curve(binarized[:, i], y_scores[:, i])
        fnr[i] = 1. - tpr
        roc_auc[i] = auc(fpr[i], tpr)

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr, _ = roc_curve(binarized.ravel(), y_scores.ravel())
    fnr["micro"] = 1. - tpr
    roc_auc["micro"] = auc(fpr["micro"], tpr)

    return fpr, fnr, roc_auc