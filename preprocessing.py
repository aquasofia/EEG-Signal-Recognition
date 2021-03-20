
import os
import data_split
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def read_data(file):

    df = pd.read_csv(file,
                     sep='\t',
                     usecols=[4, 6],
                     header=None,
                     names=['number', 'signal'])

    return df


def process_data(df):

    signals = []

    # Split by comma into arrays
    for index, row in df.iterrows():
        signals.append(row['signal'].split(','))

    # Calculate short fast fourier transform
    for i, sig in enumerate(signals):
        signals[i] = fft(sig)

    df['signal_array'] = signals

    return df


def features_classes(df):

    # Initialize a null list
    labels_unique = {}
    num = 0
    for i, l in enumerate(df['number']):
        if l not in labels_unique:
            labels_unique.update({l: num})
            num += 1

    labels_enumerated = []
    # Enumerate classes
    for index, label in enumerate(df['number']):
        labels_enumerated.append(labels_unique[label])

    numbers_df = pd.DataFrame(labels_enumerated, columns=['number'])

    # Create df output for data
    df_sub = pd.concat([numbers_df['number'], df['signal_array']], axis=1)

    return df_sub


def split_data(df):

    X = df['signal_array']
    y = df['number']

    return train_test_split(X, y, test_size=0.33, random_state=42)


def to_input_df(X, y):

    frame = {'label': y, 'features': X}

    return pd.DataFrame(frame)


def save_data(file, data):

    # Check if file exists
    path = '/'.join(file.split('/')[:-1])
    if not Path(path).exists():
        os.mkdir(path)

    # Save as pickle on disk
    data.to_pickle(file)


def main():

    data_split.main()
    directory = 'data/raw'

    print('Reading in data')
    for file in os.listdir(directory):

        with open(os.path.join(directory, file)):

            file_index = file.split('.')[0]
            file_path = os.path.join(directory, file)
            print(file_path)
            df = read_data(file_path)

            print('Processing the data')
            df_pro = process_data(df)
            df = None

            print('Extracting features and classes')
            df_sub = features_classes(df_pro)
            df_pro = None

            X_tr, X_te, y_tr, y_te = split_data(df_sub)
            df_sub = None

            print('Splitting the data to train and test sets')
            input_train = to_input_df(X_tr, y_tr)
            input_test = to_input_df(X_te, y_te)

            X_tr = None
            X_te = None
            y_tr = None
            y_te = None

            train_data = 'data/train/train_' + str(file_index)
            test_data = 'data/test/test_' + str(file_index)

            print('Saving the data to disk')
            print(train_data)
            save_data(train_data, input_train)
            print(test_data)
            save_data(test_data, input_test)


if __name__ == '__main__':
    main()

