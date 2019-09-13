import tensorflow.keras.backend as K


def focal_loss(y_true, y_pred, alpha=0.8, gamma=2):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    bce = K.binary_crossentropy(y_true, y_pred)
    bce_exp = K.exp(-bce)
    loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)

    return loss


def focal_tversky_loss(y_true, y_pred, alpha, beta, gamma, smooth):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    # true pos, false pos, false neg
    tp = K.sum((y_pred * y_true))
    fp = K.sum(((1 - y_true) * y_pred))
    fn = K.sum((y_true * (1 - y_pred)))

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    focal_tversky = K.pow((1 - tversky), gamma)

    return focal_tversky
