import tensorflow.keras.backend as K


class FocalLoss:

    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        loss = K.mean(self.alpha * K.pow((1 - bce_exp), self.gamma) * bce)

        return loss


class FocalTverskyLoss:

    def __init__(self, alpha, beta, gamma, smooth=1e-6):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        # true pos, false pos, false neg
        tp = K.sum(y_pred * y_true)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = K.pow((1 - tversky), self.gamma)
        
        return focal_tversky
