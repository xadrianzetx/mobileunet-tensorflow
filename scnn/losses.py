import tensorflow.keras.backend as K


def focal_loss(y_true, y_pred, alpha=0.8, gamma=2):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    bce = K.binary_crossentropy(y_true, y_pred)
    bce_exp = K.exp(-bce)
    loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)

    return loss
