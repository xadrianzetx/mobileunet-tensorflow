import tensorflow.keras.backend as K


def focal_loss(alpha, gamma):
    def focal(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        bce = K.binary_crossentropy(y_true, y_pred)
        bce_exp = K.exp(-bce)
        loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)

        return loss
    
    return focal


def focal_tversky_loss(alpha, beta, gamma, smooth=1e-6):
    def focal_tversky(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        # true pos, false pos, false neg
        tp = K.sum(y_pred * y_true)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        
        tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
        loss = K.pow((1 - tversky), gamma)

        return loss

    return focal_tversky
