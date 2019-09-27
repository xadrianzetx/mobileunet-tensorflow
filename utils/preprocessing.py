import cv2
import numpy as np


class LanePriorPreprocessor:

    def __init__(self, gamma=0.5):
        self._gamma = gamma
    
    def _isolate_lanes(self, img):
        pass

    def gray_contrast(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lookup = [np.clip(pow(i / 255.0, self._gamma) * 255.0, 0, 255) for i in range(256)]
        lookup = np.expand_dims(lookup, axis=0).astype(np.uint8)
        
        return cv2.LUT(gray, lookup)
    
    def apply_prior(self, img):
        pass
