import os
import pandas as pd
import colorama
from colorama import Fore, Style
from sklearn.model_selection import train_test_split

class DatasetPreprocessing(object):
    def __init__(self):
        self.startup_msg()
        self.id = "dataset_preprocessing"

        # set the dataset to a or b
        self.dataset = 'b'

        if self.dataset == 'a':
            self.dataset_root = 'datasets/aras/a/'
            self.column_headers = ['Ph1', 'Ph2', 'Ir1', 'Fo1', 'Fo2', 'Di3', 'Di4', 'Ph3', 'Ph4', 'Ph5', 'Ph6', 'Co1', 'Co2', 'Co3', 'So1', 'So2', 'Di1', 'Di2', 'Te1', 'Fo3', 'R1', 'R2', 'DAY']
            self.log('Selected dataset: a')
        elif self.dataset == 'b':
            self.dataset_root = 'datasets/aras/b/'
            self.column_headers = ['Co1', 'Co2', 'Co3', 'Co4', 'Co5', 'Co6', 'di2', 'fo1', 'fo2', 'fo3', 'ph1', 'ph2', 'pr1', 'pr2', 'pr3', 'pr4', 'pr5', 'so1', 'so2', 'so3', 'R1', 'R2', 'DAY']
            self.log('Selected dataset: b')
        else:
            self.log('Invalid dataset. Check configuration.')

        self.dataset_days = self.dataset_root + 'days/'
        self.output_file = self.dataset_root + 'all.csv'
        self.output_file_train = self.dataset_root + 'train.csv'
        self.output_file_test = self.dataset_root + 'test.csv'
        self.output_file_validation = self.dataset_root + 'validation.csv'

    def get_paths(self):
        self.log('Getting paths...')

        paths = []
        for path in os.listdir(self.dataset_days):
            full_path = os.path.join(self.dataset_days, path)
            if os.path.isfile(full_path):
                paths.append(full_path)

        self.log('Got paths:')
        print(paths)

        return paths

    def merge_and_format_files(self, paths):
        self.log('Reading file: 0')
        df = pd.read_csv(paths[0], header=None, delim_whitespace=True, dtype=int)

        df['DAY'] = paths[0]

        for i in range(1, len(paths)):
            msg  = 'Reading file: ' + str(i)
            self.log(msg)
            
            df_append = pd.read_csv(paths[i], header=None, delim_whitespace=True, dtype=int)
            df_append['DAY'] = paths[i]
            df = pd.concat([df, df_append])

        df.drop

        self.log('Writing CSV...')
        df.to_csv(self.output_file, header=False, index=False)

    def add_headers(self):
        self.log('Adding column headers to CSV...')
        df = pd.read_csv(self.output_file, header=None)
        df.to_csv(self.output_file, header = self.column_headers, index=False)

    def split_files(self):
        self.log('Creating train/test/validate files...')
        df = pd.read_csv(self.output_file)

        train, test = train_test_split(df, train_size=432000, shuffle=False)

        train = train.drop(["DAY", "R2"], axis=1)
        test = test.drop(["DAY", "R2"], axis=1)

        test, validation = train_test_split(test, test_size=0.5, shuffle=False)

        train.to_csv(self.output_file_train, index=False)
        test.to_csv(self.output_file_test, index=False)
        validation.to_csv(self.output_file_validation, index=False)

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
    dp = DatasetPreprocessing()
    paths = dp.get_paths()
    dp.merge_and_format_files(paths)
    dp.add_headers()
    dp.split_files()