import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import colorama
from colorama import Fore, Style
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding

class ModelTraining(object):
    def __init__(self):
        self.startup_msg()
        self.id = "model_training"

        self.R1 = 21
        self.R2 = 22

        self.dataset = 'a'

        if self.dataset == 'a':
            self.dataset_dir = 'datasets/aras/a/'
            self.dataset_unified = 'datasets/aras/a/all.csv'
            self.dataset_train = 'datasets/aras/a/all_train.csv'
            self.dataset_test = 'datasets/aras/a/all_test.csv'
            self.log('Selected dataset: a')
        elif self.dataset == 'b':
            self.dataset_dir = 'datasets/aras/b'
            self.dataset_unified = 'datasets/aras/b/all.csv'
            self.dataset_train = 'datasets/aras/b/all_train.csv'
            self.dataset_test = 'datasets/aras/b/all_test.csv'
            self.log('Selected dataset: b')
        else:
            self.log('Invalid dataset. Check configuration.')

    def load_dataset(self):
        self.log('Loading dataset...')

        dataframe = pd.read_csv(self.dataset_unified)

        # print(dataframe.head())
        dataframe.drop(['Index', 'R2', 'DAY'], axis=1)

        test_dataframe = dataframe.sample(frac=0.2, random_state=1337)
        train_dataframe = dataframe.drop(test_dataframe.index)

        train_ds = self.dataframe_to_dataset(train_dataframe)
        test_ds = self.dataframe_to_dataset(test_dataframe)

        train_ds = train_ds.batch(32)
        test_ds = test_ds.batch(32)

        self.log('DONE')

        return train_ds, test_ds

    def dataframe_to_dataset(self, dataframe):
        dataframe = dataframe.copy()
        labels = dataframe.pop("R1")
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        ds = ds.shuffle(buffer_size=len(dataframe))

        return ds

    def encode_integer_categorical_feature(self, feature, name, dataset):
        # Create a CategoryEncoding for our integer indices
        encoder = CategoryEncoding(output_mode="binary")

        # Prepare a Dataset that only yields our feature
        feature_ds = dataset.map(lambda x, y: x[name])
        feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

        # Learn the space of possible indices
        encoder.adapt(feature_ds)

        # Apply one-hot encoding to our indices
        encoded_feature = encoder(feature)
        return encoded_feature

    def compile_model_1(self, train_ds, test_ds):
        self.log('Training model 1...')

        Ph1 = keras.Input(shape=(1,), name="Ph1", dtype="int64")
        Ph2 = keras.Input(shape=(1,), name="Ph2", dtype="int64")
        Ir1 = keras.Input(shape=(1,), name="Ir1", dtype="int64")
        Fo1 = keras.Input(shape=(1,), name="Fo1", dtype="int64")
        Fo2 = keras.Input(shape=(1,), name="Fo2", dtype="int64")
        Di3 = keras.Input(shape=(1,), name="Di3", dtype="int64")
        Di4 = keras.Input(shape=(1,), name="Di4", dtype="int64")
        Ph3 = keras.Input(shape=(1,), name="Ph3", dtype="int64")
        Ph4 = keras.Input(shape=(1,), name="Ph4", dtype="int64")
        Ph5 = keras.Input(shape=(1,), name="Ph5", dtype="int64")
        Ph6 = keras.Input(shape=(1,), name="Ph6", dtype="int64")
        Co1 = keras.Input(shape=(1,), name="Co1", dtype="int64")
        Co2 = keras.Input(shape=(1,), name="Co2", dtype="int64")
        Co3 = keras.Input(shape=(1,), name="Co3", dtype="int64")
        So1 = keras.Input(shape=(1,), name="So1", dtype="int64")
        So2 = keras.Input(shape=(1,), name="So2", dtype="int64")
        Di1 = keras.Input(shape=(1,), name="Di1", dtype="int64")
        Di2 = keras.Input(shape=(1,), name="Di2", dtype="int64")
        Te1 = keras.Input(shape=(1,), name="Te1", dtype="int64")
        Fo3 = keras.Input(shape=(1,), name="Fo3", dtype="int64")

        all_inputs = [
            Ph1,
            Ph2,
            Ir1,
            Fo1,
            Fo2,
            Di3,
            Di4,
            Ph3,
            Ph4,
            Ph5,
            Ph6,
            Co1,
            Co2,
            Co3,
            So1,
            So2,
            Di1,
            Di2,
            Te1,
            Fo3,
        ]

        Ph1_encoded = self.encode_integer_categorical_feature(Ph1, "Ph1", train_ds)
        Ph2_encoded = self.encode_integer_categorical_feature(Ph2, "Ph2", train_ds)
        Ir1_encoded = self.encode_integer_categorical_feature(Ir1, "Ir1", train_ds)
        Fo1_encoded = self.encode_integer_categorical_feature(Fo1, "Fo1", train_ds)
        Fo2_encoded = self.encode_integer_categorical_feature(Fo2, "Fo2", train_ds)
        Di3_encoded = self.encode_integer_categorical_feature(Di3, "Di3", train_ds)
        Di4_encoded = self.encode_integer_categorical_feature(Di4, "Di4", train_ds)
        Ph3_encoded = self.encode_integer_categorical_feature(Ph3, "Ph3", train_ds)
        Ph4_encoded = self.encode_integer_categorical_feature(Ph4, "Ph4", train_ds)
        Ph5_encoded = self.encode_integer_categorical_feature(Ph5, "Ph5", train_ds)
        Ph6_encoded = self.encode_integer_categorical_feature(Ph6, "Ph6", train_ds)
        Co1_encoded = self.encode_integer_categorical_feature(Co1, "Co1", train_ds)
        Co2_encoded = self.encode_integer_categorical_feature(Co2, "Co2", train_ds)
        Co3_encoded = self.encode_integer_categorical_feature(Co3, "Co3", train_ds)
        So1_encoded = self.encode_integer_categorical_feature(So1, "So1", train_ds)
        So2_encoded = self.encode_integer_categorical_feature(So2, "So2", train_ds)
        Di1_encoded = self.encode_integer_categorical_feature(Di1, "Di1", train_ds)
        Di2_encoded = self.encode_integer_categorical_feature(Di2, "Di2", train_ds)
        Te1_encoded = self.encode_integer_categorical_feature(Te1, "Te1", train_ds)
        Fo3_encoded = self.encode_integer_categorical_feature(Fo3, "Fo3", train_ds)

        all_features = layers.concatenate(
            [
                Ph1_encoded,
                Ph2_encoded,
                Ir1_encoded,
                Fo1_encoded,
                Fo2_encoded,
                Di3_encoded,
                Di4_encoded,
                Ph3_encoded,
                Ph4_encoded,
                Ph5_encoded,
                Ph6_encoded,
                Co1_encoded,
                Co2_encoded,
                Co3_encoded,
                So1_encoded,
                So2_encoded,
                Di1_encoded,
                Di2_encoded,
                Te1_encoded,
                Fo3_encoded,
            ]
        )

        x = layers.Dense(32, activation="relu")(all_features)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1, activation="sigmoid")(x)
        model = keras.Model(all_inputs, output)
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])

        # keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

        model.fit(train_ds, epochs=50, validation_data=test_ds, verbose=1)

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
    mt = ModelTraining()

    train_ds, test_ds = mt.load_dataset()
    mt.compile_model_1(train_ds, test_ds)
