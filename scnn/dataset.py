import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from utils.decoding import mask_from_splines


class ImageDataGenerator:

    def __init__(self, path, batch_size, lookup_name, augment=False):
        self._path = path
        self._batch_size = batch_size
        self._lookup_name = lookup_name
        self._augment = augment
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
        batch_x, batch_y = [], []

        for idx, row in batch.iterrows():
            # load image and switch color channels
            img_path = row['img_path'].replace('/', os.path.sep)
            img = Image.open(os.path.join(self._path, img_path[1:]))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape

            # create empty mask
            mask_path = os.path.splitext(row['img_path'])[0]
            mask_path += '.lines.txt'
            mask_path = mask_path.replace('/', os.path.sep)
            mask = np.zeros((height, width), dtype=int)

            with open(os.path.join(self._path, mask_path[1:]), 'r') as file:
                # add lines to mask one by one
                for line in file:
                    mask = mask_from_splines(line=line, mask=mask)

            if self._augment:
                # TODO
                self._augment_image_mask(img, mask)

            batch_x.append(img)
            batch_y.append(mask)

        return np.array(batch_x), np.array(batch_y)

    def _read_lookup(self):
        abs_path = os.path.join(self._path, self._lookup_name)

        if not os.path.isfile(abs_path):
            raise FileNotFoundError('')

        lookup = pd.read_csv(abs_path, header=None)
        lookup.columns = ['img_path']

        return lookup.sample(frac=1).reset_index(drop=True)

    def _create_mask(self, size, splines_path):
        pass

    def _augment_image_mask(self, img, mask):
        pass
