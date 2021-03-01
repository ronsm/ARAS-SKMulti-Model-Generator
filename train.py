import pandas as pd
import numpy as np
from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.meta import OnlineRUSBoostClassifier 
from skmultiflow.bayes import NaiveBayes
import colorama
from colorama import Fore, Style
import pickle
import sys

class ScikitMultiflowTest(object):
    def __init__(self):
        self.startup_msg()
        self.id = 'scikit_multiflow_test'

        # set the dataset to a or b
        self.dataset = 'b'

        if self.dataset == 'a':
            self.training_file = 'datasets/aras/a/train.csv'
        elif self.dataset == 'b':
            self.training_file = 'datasets/aras/b/train.csv'
        else:
            self.log('Invalid dataset. Check configuration.')

        self.training_limit = 5000000

    # Dataset

    def load_and_test_dataset(self, file):
        msg = 'Loading file: ' + file
        self.log(msg)

        stream = FileStream(file)
        
        self.num_samples = stream.n_remaining_samples()

        self.log('Test sample:')
        print(stream.next_sample())
        self.log('Remaining samples:')
        print(stream.n_remaining_samples())

        return stream

    # Naive Bayes

    def train_naive_bayes(self, stream):
        self.log('Training Naive Bayes classifier...')

        nb = NaiveBayes()

        n_samples = 0
        correct_cnt = 0

        while (n_samples < self.num_samples and stream.has_more_samples()) and (n_samples < self.training_limit):
            X, y = stream.next_sample()
            y_pred = nb.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt = correct_cnt + 1
            nb.partial_fit(X, y)
            n_samples = n_samples + 1
            if (n_samples % 1000) == 0:
                print('Naive Bayes accuracy: {}'.format(correct_cnt / n_samples), n_samples)

        print('{} samples analyzed.'.format(n_samples))
        print('Naive Bayes accuracy: {}'.format(correct_cnt / n_samples), n_samples)

        return nb

    # Hoeffding Tree

    def train_hoeffding_tree(self, stream):
        self.log('Training Hoeffding Tree classifier...')

        ht = HoeffdingTreeClassifier()

        y_pred = np.zeros(self.num_samples)
        y_true = np.zeros(self.num_samples)

        n_samples = 0
        correct_cnt = 0

        while (n_samples < self.num_samples and stream.has_more_samples()) and (n_samples < self.training_limit):
            X, y = stream.next_sample()
            y_pred = ht.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt = correct_cnt + 1
            ht = ht.partial_fit(X, y, classes=stream.target_values)
            n_samples = n_samples + 1
            if (n_samples % 1000) == 0:
                print('Hoeffding Tree accuracy: {}'.format(correct_cnt / n_samples), n_samples)

        print('{} samples analyzed.'.format(n_samples))
        print('Hoeffding Tree accuracy: {}'.format(correct_cnt / n_samples))

        return ht

    # RUS Boost

    def train_rus_boost(self, stream):
        self.log('Training RUSBoost...')

        rb = OnlineRUSBoostClassifier()

        n_samples = 0
        correct_cnt = 0

        while (n_samples < self.num_samples and stream.has_more_samples()) and (n_samples < self.training_limit):
            X, y = stream.next_sample()
            y_pred = rb.predict(X)
            if y[0] == y_pred[0]:
                correct_cnt = correct_cnt + 1
            rb.partial_fit(X, y, classes=stream.target_values)
            n_samples = n_samples + 1
            if (n_samples % 1000) == 0:
                print('RUSBoost accuracy: {}'.format(correct_cnt / n_samples), n_samples)

        print('{} samples analyzed'.format(n_samples))
        print('Online RUSBoost performance: {}'.format(correct_cnt / n_samples))

        return rb

    # Utilities 

    def save_model(self, model, name):
        self.log('Saving model...')
        save_file = name + '.p'
        pickle.dump(model, open(save_file, "wb"))

    def set_train_limit(self, train):
        if train == 0:
            self.training_limit = 5000000
        else:
            self.training_limit = train

    def startup_msg(self):
        print(Fore.YELLOW + '* * * * * * * * * * * * * * * * * *')
        print()
        print(Style.BRIGHT + 'Untitled Project' + Style.RESET_ALL + Fore.YELLOW)
        print()
        print(' Developer: Ronnie Smith')
        print(' Email:     ronnie.smith@ed.ac.uk')
        print(' GitHub:    @ronsm')
        print()
        print('* * * * * * * * * * * * * * * * * *')

    def log(self, msg):
        tag = '[' + self.id + ']'
        print(Fore.CYAN + tag, Fore.RESET + msg)

if __name__ == "__main__":
    smt = ScikitMultiflowTest()

    stream = smt.load_and_test_dataset(smt.training_file)

    train_limit = int(sys.argv[1])

    smt.set_train_limit(train_limit)

    rb = smt.train_rus_boost(stream)
    smt.save_model(rb, 'RUSBoost')

    stream = smt.load_and_test_dataset(smt.training_file)

    ht = smt.train_hoeffding_tree(stream)
    smt.save_model(ht, 'HoeffdingTree')

    stream = smt.load_and_test_dataset(smt.training_file)
    
    nb = smt.train_naive_bayes(stream)
    smt.save_model(nb, 'NaiveBayes')