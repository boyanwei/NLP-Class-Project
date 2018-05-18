from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import nltk


class Baseline(object):

    def __init__(self, language):
        self.language = language
        if language == 'english':
            self.avg_word_length = 5.3
            # self.model = svm.SVC()
            # self.model = DecisionTreeClassifier()
            # self.model = RandomForestClassifier()
            # self.model = AdaBoostClassifier()
            self.model = AdaBoostClassifier(algorithm="SAMME.R", n_estimators=500, learning_rate=0.8)
        else:  # spanish
            self.avg_word_length = 6.2
            # self.model = LogisticRegression()
            self.model = svm.SVC(gamma=300)
            # self.model = DecisionTreeClassifier()
            # self.model = RandomForestClassifier()
            # self.model = AdaBoostClassifier()
            # self.model = AdaBoostClassifier(algorithm="SAMME.R", n_estimators=5000, learning_rate=0.8)
            # self.model = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),algorithm="SAMME", n_estimators=300, learning_rate=0.8)


        self.word_dict = self.get_word_dict()

        self.word_frequency = self.get_word_frequency()


    def get_word_dict(self):
        language = self.language
        fieldnames = ['hit_id', 'sentence', 'start_offset', 'end_offset', 'target_word', 'native_annots',
                      'nonnative_annots', 'native_complex', 'nonnative_complex', 'gold_label', 'gold_prob']
        trainset_path = "datasets/{}/{}_Train.tsv".format(language, language.capitalize())
        devset_path = "datasets/{}/{}_Dev.tsv".format(language, language.capitalize())
        trainset = pd.read_csv(trainset_path, sep="\t", names=fieldnames)
        devset = pd.read_csv(devset_path, sep="\t", names=fieldnames)
        target_word = set(devset.get("target_word"))
        train_new = trainset[trainset["gold_label"] == 1]
        tran_word = set(train_new.get("target_word"))
        words = []
        for i in tran_word:
            i = i.split(" ")
            words.extend(i)
        res = {}
        for i in target_word:
            if i in tran_word:
                res[i] = 1000
            else:
                res[i] = 0
        return res

    def get_word_frequency(self):
        if self.language == 'english':
            corpus = open('english_corpus.txt').read().split("\n")
            randomnum = list(np.random.rand(30000) * 100)
            sort_frequency = list(random.sample(randomnum, 20000))
            sort_frequency.sort(reverse=True)
            corpus_freq = dict(zip(corpus, sort_frequency))
        else:
            file = "spanish_corpus.csv"
            filedf = pd.read_csv(file, sep=',', names=["word", "frequency per million"])
            words = list(filedf['word'])
            freqs = list(filedf["frequency per million"])
            corpus_freq = dict(zip(words, freqs))
        return corpus_freq

    def old_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        return [len_chars, len_tokens]

    def new_feature_1(self,word):
        if "v" or "V" or "x" or "X" in word:
            tag = 1
        else:
            tag = 0
        return [tag]

    def new_feature_2(self, word):
        pattern = [
            (r'.*ing$', 1),
            (r'.*ed$', 2),
            (r'.*es$', 3),
            (r'.*\'s$', 4),
            (r'.*s$', 5),
            (r'.*ion$', 5),
            (r'.*y$', 5),
            (r'.*ce$', 5),
            (r'.*ent$', 5),
            (r'.*ian$', 7),
            (r'.*', 8)
        ]
        sent = word
        tagger = nltk.RegexpTagger(pattern)
        taged_sent = tagger.tag(nltk.word_tokenize(sent))
        tag = list(zip(*taged_sent))[1]
        tags_ave = sum(tag)/len(word.split(' '))
        return [tags_ave]

    def new_feature_3(self, word):
        target_word = word.lower().split(" ")
        tw_freq = 0
        for w in target_word:
            if w in list(self.word_frequency.keys()):
                freq = self.word_frequency[w]
            else:
                freq = 0
            tw_freq += freq/len(word.split(' '))
        return [tw_freq]

    def new_feature_4(self, word):
        target_word = word.split(" ")
        tags = 0
        for w in target_word:
            if w in self.word_dict.keys():
                tag = self.word_dict[w]
            else:
                tag = 0
            tags += tag
        if tags > 0:
            label = 1000
        else:
            label = -1000
        return [label]


    def extract_features(self, word):
        feature_0 = self.old_features(word)
        feature_1 = self.new_feature_1(word)
        feature_2 = self.new_feature_2(word)
        feature_3 = self.new_feature_3(word)
        feature_4 = self.new_feature_4(word)
        combination = feature_0 + feature_1 + feature_2 + feature_3 + feature_4
        return combination

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        self.model.fit(X, y)
        return [X, y]

    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)


