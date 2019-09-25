import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


class DiceCoefficient(Layer):

    def __init__(self, smooth, name=None, **kwargs):
        Layer.__init__(name=name, **kwargs)
        self._smooth = smooth
    
    def __call__(self, y_true, y_pred):
        """
        Sørensen–Dice coefficient
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        
        """
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred)
        dice = (2. * intersection + self._smooth) / (union + self._smooth)

        return dice
