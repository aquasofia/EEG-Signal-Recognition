from torch.utils.data import Dataset
from torch import tensor, zeros, cat
import pandas as pd
import torch
import numpy as np


class EEG(Dataset):

    def __init__(self,
                 file: str,
                 data_dir: str,
                 features='features',
                 label='label',
                 ):

        super().__init__()

        data = pd.read_pickle(data_dir + file)

        self.key_features = data[features].to_numpy()
        self.key_class = data[label].to_numpy()

    def __len__(self) \
            -> int:
        """
        Returns the length of the dataset.

        :return: Length of the dataset.
        :rtype: int
        """

        return len(self.key_features)

    def __getitem__(self,
                    item: int):

        """Returns an item from the dataset.
        :param item: Index of the item.
        :type item: int
        """

        # Pytorch expects input as shape: [N x C x L]
        # N = Number of samples in a batch: Batch 5-10
        # C = Number of channels: 1
        # L = Length of the signal sequence (one EEG signal) : 260

        features = torch.DoubleTensor(np.real(self.key_features[item])).unsqueeze(0)

        return features, self.key_class[item]

