

from dataset import Dataset
from baseline import Baseline
from scorer import report_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("Spa_LC_SVM_300")



def execute(language):
    language = language
    data = Dataset(language)
    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    baseline = Baseline(language)
    estimator = SVC(gamma=300)
    title = 'Spanish Learning Curves (SVM, γ=300)'
    X, y = baseline.train(data.trainset)
    plot_learning_curve(estimator, title, X, y, ylim=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5))

    predictions = baseline.test(data.devset)

    gold_labels = [sent['gold_label'] for sent in data.devset]

    target_words = [sent['target_word'] for sent in data.devset]
    prediction = []
    for i in predictions:
        prediction.append(i)
    df= pd.DataFrame(columns=['target_word', 'prediction'])
    df["target_word"] = target_words
    df['gold_label'] = gold_labels
    df['prediction'] = prediction
    df.to_csv('out_s2.csv')
    report_score(gold_labels, predictions)






if __name__ == '__main__':
    # execute('english')
    execute('spanish')




