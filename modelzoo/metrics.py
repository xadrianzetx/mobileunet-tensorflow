import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer


def dice_coefficient(smooth=1):
    def dice(y_true, y_pred):
        """
        Sørensen–Dice coefficient
        https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        
        """
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred)
        coef = (2. * intersection + smooth) / (union + smooth)

        return coef
    
    return dice
