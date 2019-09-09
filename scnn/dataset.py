import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from utils.decoding import mask_from_splines


class CULaneImage:

    def __init__(self, path, batch_size, lookup_name, **kwargs):
        self._path = path
        self._batch_size = batch_size
        self._lookup_name = lookup_name
        self._augment = kwargs.get('augment', False)
        self._augmentations = kwargs.get('augmentations', ('scale', 'flip', 'brightness'))
        self._size = kwargs.get('size', (1080, 720))
        self._lookup = self._read_lookup()

    @staticmethod
    def _create_mask(size, splines_path):
        # initialize empty mask
        mask = np.zeros(size, dtype=np.uint8)

        with open(splines_path, 'r') as file:
            for line in file:
                # build mask by encoding traffic line pixels line by line
                mask = mask_from_splines(line=line, mask=mask)

        return mask

    def _read_lookup(self):
        abs_path = os.path.join(self._path, self._lookup_name)

        if not os.path.isfile(abs_path):
            raise FileNotFoundError('Could not find lookup file')

        # read lookup and shuffle before first epoch
        lookup = pd.read_csv(abs_path, header=None)
        lookup.columns = ['img_path']
        lookup = lookup.sample(frac=1).reset_index(drop=True)

        return lookup
    
    def _get_batch(self, metadata):
        batch_x, batch_y = [], []

        for idx, row in metadata.iterrows():
            # load image and switch color channels
            img_path = row['img_path'].replace('/', os.path.sep)
            img = Image.open(os.path.join(self._path, img_path[1:]))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape

            # create empty mask and rebuild lines from splines
            mask_path = os.path.splitext(row['img_path'])[0]
            mask_path += '.lines.txt'
            mask_path = os.path.join(self._path, mask_path[1:]).replace('/', os.path.sep)
            mask = self._create_mask(size=(height, width), splines_path=mask_path)

            if self._augment:
                img, mask = self._augment_image_mask(img, mask)

            batch_x.append(img)
            batch_y.append(mask)
        
        return batch_x, batch_y

    def _augment_image_mask(self, img, mask):
        pair = [img, mask]
        pair = [cv2.resize(x, self._size) for x in pair]

        if 'flip' in self._augmentations and np.random.choice([True, False]):
            # horizontal flip
            pair = [cv2.flip(x, 1) for x in pair]

        return pair[0], pair[1]


class CULaneImageIterator(CULaneImage):

    def __init__(self, path, batch_size, lookup_name, augment=False):
        CULaneImage.__init__(self, path, batch_size, lookup_name, augment=augment)
        self._max_idx = self._lookup.shape[0]
        self._idx = 0

    def __iter__(self):
        """
        Return the iterator object.

        :return:    self
        """
        return self

    def __next__(self):
        """
        Creates batch of image-mask pairs from CULane dataset

        Iterator version to use as standalone batch control

        Masks are build from splines annotations so that each
        pixel is encoded for semantic segmentation.

        :return:    batch_x: np.ndarray NxHxWxC
                    CULane dataset images
                    batch_y: np.ndarray NxHxW
                    binary masks for semantic segmentation
        """
        if self._idx + self._batch_size > self._max_idx:
            # reset index and shuffle on epoch end
            self._idx = 0
            self._lookup = self._lookup.sample(frac=1).reset_index(drop=True)

        # select subset of data for batch
        batch = self._lookup.loc[self._idx:self._idx + self._batch_size - 1]
        self._idx += self._batch_size
        batch_x, batch_y = self._get_batch(metadata=batch)      

        return np.array(batch_x), np.array(batch_y)


class CULaneImageGenerator(CULaneImage):

    def __init__(self, path, batch_size, lookup_name, augment=False):
        CULaneImage.__init__(self, path, batch_size, lookup_name, augment=augment)
        self._max_idx = self._lookup.shape[0]
        self._idx = 0

    def __call__(self):
        """
        Creates batch of image-mask pairs from CULane dataset
        
        Generator version to use with tf.data.Dataset.from_generator()

        Masks are build from splines annotations so that each
        pixel is encoded for semantic segmentation.

        :return:    img: np.ndarray HxWxC
                    CULane dataset image
                    mask: np.ndarray HxW
                    binary mask for semantic segmentation
        """
        # reset idx and shuffle batch on epoch start
        self._lookup = self._lookup.sample(frac=1).reset_index(drop=True)
        self._idx = 0

        while self._idx < self._max_idx:
            # get one img-mask pair
            # batch is controlled by tf.data.Dataset
            obs = self._lookup.loc[self._idx]
            img, mask = self._get_batch(metadata=obs)
            self._idx += 1

            yield img, mask
    
    def _get_batch(self, metadata):
        # load image and switch color channels
        img_path = metadata['img_path'].replace('/', os.path.sep)
        img = Image.open(os.path.join(self._path, img_path[1:]))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        
        # create empty mask and rebuild lines from splines
        mask_path = os.path.splitext(metadata['img_path'])[0]
        mask_path += '.lines.txt'
        mask_path = os.path.join(self._path, mask_path[1:]).replace('/', os.path.sep)
        mask = self._create_mask(size=(height, width), splines_path=mask_path)
        
        if self._augment:
            img, mask = self._augment_image_mask(img, mask)
        
        return img, mask
