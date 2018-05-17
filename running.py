

from dataset import Dataset
from baseline import Baseline
from scorer import report_score
import pandas as pd

def execute_demo(language):
    data = Dataset(language)

    print("{}: {} training - {} dev".format(language, len(data.trainset), len(data.devset)))

    # for sent in data.trainset:
    #    print(sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)

    baseline.train(data.trainset)

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
    execute_demo('english')
    execute_demo('spanish')


