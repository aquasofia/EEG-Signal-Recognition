
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.fftpack import fft


def main():

    signals = []
    file = 'data/raw/50000.txt'
    with open(file):
        df = pd.read_csv(file,
                         sep='\t',
                         usecols=[4, 6],
                         header=None,
                         names=['number', 'signal'])

    sample = df['signal'][0]

    # Split by comma into arrays
    for index, row in df.iterrows():
        if index == 0:
            signals.append(row['signal'].split(','))

    sig = signals[0].T
    plt.plot(sig)
    plt.show()


if __name__ == '__main__':
    main()