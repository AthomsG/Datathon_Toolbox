# METRICS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

'''
ytest           - List of test set target values;

preds           - List of models predictions (if BALANCING_NAMES=None);
                - Dictionary of predictions, where each entry corresponds to a models, trained on BALANCING_NAME dataset, predictions;

BALANCING_NAMES - List of balanced dataset names. SHOULD BE EQUAL TO preds.keys()

OUTPUT ORDER - Accuracy, Precision, Recall, F1 score.
'''
def get_dataset_metrics(ytest, preds, BALANCING_NAMES=None):
    dataset_metrics=dict()
    if (BALANCING_NAMES):

        for BALANCING_NAME in BALANCING_NAMES:
            dataset_metrics[BALANCING_NAME]=(accuracy_score(ytest, preds[BALANCING_NAME]),
                                             precision_score(ytest, preds[BALANCING_NAME]),
                                             recall_score(ytest, preds[BALANCING_NAME]),
                                             f1_score(ytest, preds[BALANCING_NAME]))
    else:
        dataset_metrics['Best Dataset']=(accuracy_score(ytest, preds),
                                             precision_score(ytest, preds),
                                             recall_score(ytest, preds),
                                             f1_score(ytest, preds))
    return dataset_metrics


def plot_metrics(dataset_metrics, titles=('Accuracy', 'Precision', 'Recall', 'F1 score')):
    x = np.arange(len(titles))  # the label locations
    width = 0.20  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(constrained_layout=True)

    for attribute, measurement in dataset_metrics.items():
        offset = width * multiplier - 0.1
        rects = ax.bar(x + offset, measurement, width, label=attribute, edgecolor='white')
        ax.bar_label(rects, labels=tuple(map(lambda x: round(x, 2), measurement)), fontsize=6)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_title('Binary Classification Metrics')
    ax.set_xticks(x + width, titles)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 1)

    plt.show()
