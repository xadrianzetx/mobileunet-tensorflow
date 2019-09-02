import os
import numpy as np
import pandas as pd
from utils.decoding import mask_from_splines


class ImageDataGenerator:

    def __init__(self, path, batch_size, lookup_name):
        self._path = path
        self._batch_size = batch_size
        self._lookup_name = lookup_name
        self._lookup = self._read_lookup()
        self._max_idx = self._lookup.shape[0]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        """

        :return:
        """
        if self.idx + self._batch_size > self._max_idx:
            # reset index and shuffle
            self.idx = 0
            self._lookup = self._lookup.sample(frac=1).reset_index(drop=True)

        batch = self._lookup.loc[self.idx:self.idx + self._batch_size - 1]
        self.idx += self._batch_size
        x = []

        for idx, row in batch.iterrows():
            img_path = row['img_path'].replace('/', '\\')
            x.append(os.path.join(self._path, img_path))

        return x

    def _read_lookup(self):
        abs_path = os.path.join(self._path, self._lookup_name)

        if not os.path.isfile(abs_path):
            raise FileNotFoundError('')

        lookup = pd.read_csv(abs_path, header=None)
        lookup.columns = ['img_path']

        return lookup.sample(frac=1).reset_index(drop=True)

    def create_mask(self):
        pass
