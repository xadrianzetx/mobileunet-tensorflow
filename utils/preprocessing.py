import cv2
import numpy as np


class LanePriorPreprocessor:

    def __init__(self, gamma=0.5, horizon_trim=0.2, fov_trim=0.2):
        self._gamma = gamma
        self._h_trim = horizon_trim
        self._fov_left = fov_trim
        self._fov_right = (1 - fov_trim)
    
    def _isolate_lanes(self, img):
        # isolate white regions of img
        mask = cv2.inRange(img, 120, 255)
        img = cv2.bitwise_or(img, img, mask=mask)

        return img

    def gray_contrast(self, img):
        """
        Improves contrast between road and lanes on grayscale image

        :param img: np.ndarray HxWxC
                    original image

        :return:    np.ndarray HxWxC
                    grayscale img with enhanced contrast
        """
        # convert img to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # enhance contrast
        lookup = [np.clip(pow(i / 255.0, self._gamma) * 255.0, 0, 255) for i in range(256)]
        lookup = np.expand_dims(lookup, axis=0).astype(np.uint8)

        return cv2.LUT(gray, lookup)
    
    def apply_prior(self, img):
        """

        :param img:
        :return:
        """
        img = self._isolate_lanes(img)
        height, width = img.shape
        mask = np.zeros_like(img, dtype=np.uint8)

        # FOV correction
        lb = [0, height]
        rb = [width, height]
        lt = [width * self._fov_left, self._h_trim * height]
        rt = [width * self._fov_right, self._h_trim * height]

        vertices = np.array([[lt, lb, rb, rt]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        prior = cv2.bitwise_and(img, mask)
        prior = cv2.Canny(prior, 220, 250)

        # TODO with canny edges, get tall and narrow shapes
        # and apply them as lane priors to original img
        # dunno if this is robust enougsh

        return prior
