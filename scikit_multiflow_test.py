import pandas as pd
import numpy as np
from skmultiflow.data.file_stream import FileStream
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.bayes import NaiveBayes
import colorama
from colorama import Fore, Style
import pickle

class ScikitMultiflowTest(object):
    def __init__(self):
        self.startup_msg()
        self.id = 'scikit_multiflow_test'

        self.training_file = 'datasets/aras/b/train.csv'
        self.test_file = 'datasets/aras/b/test.csv'

        self.training_limit = 40000
        self.testing_limit = 500000

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

    def predict_naive_bayes(self, nb, stream):
        self.log('Making predictions with Naive Bayes...')

        n_samples = 0

        predictions = []

        while (n_samples < self.num_samples and stream.has_more_samples()) and (n_samples < self.testing_limit):
            X, y = stream.next_sample()
            y_pred = nb.predict(X)
            predictions.append(y_pred[0])
            n_samples = n_samples + 1
            if (n_samples % 1000) == 0:
                print('Number of samples processed: ', n_samples)

        return predictions

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
            ht = ht.partial_fit(X, y)
            n_samples = n_samples + 1
            if (n_samples % 1000) == 0:
                print('Hoeffding Tree accuracy: {}'.format(correct_cnt / n_samples), n_samples)

        print('{} samples analyzed.'.format(n_samples))
        print('Hoeffding Tree accuracy: {}'.format(correct_cnt / n_samples))

        return ht

    def predict_hoeffding_tree(self, ht, stream):
        self.log('Making predictions with Hoeffding Tree...')

        n_samples = 0

        predictions = []

        while (n_samples < self.num_samples and stream.has_more_samples()) and (n_samples < self.testing_limit):
            X, y = stream.next_sample()
            y_pred = ht.predict(X)
            predictions.append(y_pred[0])
            n_samples = n_samples + 1
            if (n_samples % 1000) == 0:
                print('Number of samples processed: ', n_samples)

        return predictions

    # Assessment

    def assess_performance(self, predictions):
        self.log('Assessing performance...')
        df = pd.read_csv(self.test_file)

        y_true = df['R1'].to_list()

        correct_cnt = 0
        for i in range(0, len(predictions)):
            if predictions[i] == y_true[i]:
                correct_cnt = correct_cnt + 1

        score = correct_cnt / len(predictions)
        msg = 'Percent correct (accuracy): ' + str(score)
        self.log(msg)

    # Utilities 

    def save_model(self, ht):
        self.log('Saving model...')
        pickle.dump(ht, open("save.p", "wb"))

    def load_model(self):
        self.log('Loading model...')
        ht = pickle.load(open("save.p", "rb"))

        return ht

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

    ht = smt.train_hoeffding_tree(stream)
    smt.save_model(ht)
    ht = smt.load_model()
    stream = smt.load_and_test_dataset(smt.test_file)
    predictions = smt.predict_hoeffding_tree(ht, stream)

    # nb = smt.train_naive_bayes(stream)
    # smt.save_model(nb)
    # nb = smt.load_model()
    # stream = smt.load_and_test_dataset(smt.test_file)
    # predictions = smt.predict_naive_bayes(nb, stream)

    smt.assess_performance(predictions)