import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from utils.decoding import mask_from_splines


class CULaneImage:

    def __init__(self, path, lookup_name, batch_size, size, **kwargs):
        self._path = path
        self._batch_size = batch_size
        self._lookup_name = lookup_name
        self._augment = kwargs.get('augment', False)
        self._augment_proba = kwargs.get('augment_proba', 0.4)
        self._augmentations = kwargs.get('augmentations', ('flip', 'rotate', 'crop', 'brightness'))
        self._scale = kwargs.get('scale', True)
        self._size = (size[1], size[0])
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

            if img_path.startswith('/'):
                img_path = img_path[1:]
            
            img = Image.open(os.path.join(self._path, img_path))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, _ = img.shape

            # create empty mask and rebuild lines from splines
            mask_path = os.path.splitext(row['img_path'])[0]
            mask_path += '.lines.txt'

            if mask_path.startswith('/'):
                mask_path = mask_path[1:]

            mask_path = os.path.join(self._path, mask_path).replace('/', os.path.sep)
            mask = self._create_mask(size=(height, width), splines_path=mask_path)

            if self._augment:
                img, mask = self._augment_image_mask(img, mask)

            img = cv2.resize(img, self._size)
            mask = cv2.resize(mask, self._size)
            mask = np.expand_dims(mask, axis=-1)

            batch_x.append(img)
            batch_y.append(mask)
        
        return batch_x, batch_y

    def _augment_image_mask(self, img, mask):
        pair = [img, mask]
        p = [self._augment_proba, 1 - self._augment_proba]

        if 'flip' in self._augmentations and np.random.choice([True, False], p=p):
            # horizontal flip
            pair = [cv2.flip(x, 1) for x in pair]
        
        if 'rotate' in self._augmentations and np.random.choice([True, False], p=p):
            # randomly rotate img-mask pair
            theta = np.random.randint(-5, 5)
            height, width, _ = pair[0].shape
            rot_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), theta, 1)
            pair = [cv2.warpAffine(x, rot_matrix, (width, height)) for x in pair]
        
        if 'crop' in self._augmentations and np.random.choice([True, False], p=p):
            # crop random part of img
            # ratio is width // height
            crop_ratio = self._size[0] // self._size[1]
            crop_scale = np.random.uniform(0.99, 0.9999)

            # cropped img has to be roughly same ratio as original img
            crop_height = int(self._size[1] * crop_scale)
            crop_width = crop_height * crop_ratio

            # get random start point
            max_x = self._size[1] - crop_width
            max_y = self._size[0] - crop_height
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)

            # crop image and resize to required size
            pair = [img[y:y + crop_height, x:x + crop_width] for img in pair]
            pair = [cv2.resize(x, self._size) for x in pair]
        
        if 'brightness' in self._augmentations and np.random.choice([True, False], p=p):
            # brightness correction
            # https://docs.opencv.org/3.4/Basic_Linear_Transform_Tutorial_gamma.png
            gamma = np.random.uniform(0.67, 2.)
            lookup = np.empty((1,256), np.uint8)

            for i in range(256):
                lookup[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

            pair[0] = cv2.LUT(pair[0], lookup)

        return pair[0], pair[1]


class CULaneImageIterator(CULaneImage):

    def __init__(self, path, batch_size, lookup_name, **kwargs):
        CULaneImage.__init__(self, path, batch_size, lookup_name, **kwargs)
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
        batch_x = np.array(batch_x).astype(float)
        batch_y = np.array(batch_y).astype(float)

        if self._scale:
            batch_x *= (1 / 255)

        return batch_x, batch_y


class CULaneImageGenerator(CULaneImage):

    def __init__(self, path, batch_size, lookup_name, **kwargs):
        CULaneImage.__init__(self, path, batch_size, lookup_name, **kwargs)
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
            # get batch of img-mask pairs
            # generator StopIteration is controlled by tf.data.Dataset
            batch = self._lookup.loc[self._idx:self._idx + self._batch_size - 1]
            self._idx += self._batch_size

            batch_x, batch_y = self._get_batch(metadata=batch)
            batch_x = np.array(batch_x).astype(float)
            batch_y = np.array(batch_y).astype(float)

            if self._scale:
                batch_x *= (1 / 255)
            
            if batch_x.shape[0] == self._batch_size:
              yield batch_x, batch_y
