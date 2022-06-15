import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc


# Predictions on the test set
def anomaly_detection(x_test, y_test, batch_size, discriminator):
    nr_batches_test = np.ceil(x_test.shape[0] // batch_size).astype(np.int32)

    results = []


    for t in range(nr_batches_test + 1):
        ran_from = t * batch_size
        ran_to = (t + 1) * batch_size
        image_batch = x_test[ran_from:ran_to]
        tmp_rslt = discriminator.predict(
            x=image_batch, batch_size=128, verbose=0)
        results = np.append(results, tmp_rslt)

    pd.options.display.float_format = '{:20,.7f}'.format
    results_df = pd.concat(
        [pd.DataFrame(results), pd.DataFrame(y_test)], axis=1)
    results_df.columns = ['results', 'y_test']

    # Obtaining the lowest 3% score
    per = np.percentile(results, 3)
    y_pred = results.copy()
    y_pred = np.array(y_pred)

    # Thresholding based on the score
    inds = (y_pred > per)
    inds_comp = (y_pred <= per)
    y_pred[inds] = 0
    y_pred[inds_comp] = 1

    return y_pred, results_df


# Predictions on the test set
def anomaly_detection_test(x_test, batch_size, discriminator):
    nr_batches_test = np.ceil(x_test.shape[0] // batch_size).astype(np.int32)

    results = []

    for t in range(nr_batches_test + 1):
        ran_from = t * batch_size
        ran_to = (t + 1) * batch_size
        image_batch = x_test[ran_from:ran_to]
        tmp_rslt = discriminator.predict(
            x=image_batch, batch_size=128, verbose=0)
        results = np.append(results, tmp_rslt)

    pd.options.display.float_format = '{:20,.7f}'.format

    # Obtaining the lowest 3% score
    per = np.percentile(results, 3)
    y_pred = results.copy()
    y_pred = np.array(y_pred)

    # Thresholding based on the score
    inds = (y_pred > per)
    inds_comp = (y_pred <= per)
    y_pred[inds] = 0
    y_pred[inds_comp] = 1

    return y_pred


# Evaluates the performance of the model
def evaluation(y_test, y_pred, print_results=False):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary')
    acc_score = accuracy_score(y_test, y_pred)
    if print_results:
        print('Accuracy Score :', acc_score)
        print('Precision :', precision)
        print('Recall :', recall)
        print('F1 :', f1)
    return acc_score, precision, recall, f1


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    target_names = ['normal', 'anomaly']
    # plt.figure(figsize=(10,10),)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()

    width, height = cm.shape

    for x in range(width):
        for y in range(height):
            plt.annotate(str(cm[x][y]), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_roc_curve(y_test, y_pred):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    # plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras,
             label='Keras (area = {:.2f})'.format(auc_keras))

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()


# Performs only the anomaly detection given a test set a batch size and a discriminator
def test(x_test, batch_size, discriminator):
    y_pred = anomaly_detection_test(x_test, batch_size, discriminator)
    return y_pred
