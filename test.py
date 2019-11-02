import cv2
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from utils.decoding import image_mask_overlay
from modelzoo.losses import focal_tversky_loss
from modelzoo.metrics import dice_coefficient


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', type=str, default='image')
    parser.add_argument('-f', type=str, default='img.jpg')
    parser.add_argument('-s', type=str, default='model.h5')
    parser.add_argument('--flatbuff', action='store_true')

    return parser.parse_args()


def load_net(src, flatbuff):
    if flatbuff:
        # load .tflite flatbuffer model
        model = tf.lite.Interpreter(model_path=src)
        model.allocate_tensors()

    else:
        # load keras .h5 object
        obj = {
            'focal_tversky': focal_tversky_loss(0.7, 0.3, 0.75),
            'dice': dice_coefficient()
        }
        model = tf.keras.models.load_model(src, custom_objects=obj)

    return model


def predict(model, img, flatbuff, normalize=True):
    original = img.copy()

    if normalize:
        img *= (1 / 255)

    # add batch channel and run inference
    # then round probas to classes with threshold 0.5
    img = np.expand_dims(img, axis=0)

    if flatbuff:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], img)
        model.invoke()
        pred = model.get_tensor(output_details[0]['index'])

    else:
        pred = model.predict(img)

    mask = np.round(pred[0, :, :, 0])
    out = image_mask_overlay(original, mask)

    return out


def from_img(src, model_src, flatbuff, shrink=2):
    # lane segmentation on single frame
    # load image and resize to size expected by net
    img = Image.open(src)
    img = np.array(img)
    height, width, _ = img.shape
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32)

    # run inference
    model = load_net(model_src, flatbuff)
    pred = predict(model, img, flatbuff, normalize=True)

    # resize back to original shape and show
    pred = cv2.resize(pred, (int(width) // shrink, int(height) // 2))
    cv2.imshow('image', pred.astype(np.uint8))
    cv2.waitKey(0)


def from_video(src, model_src, flatbuff, shrink=2):
    # lane segmentation on video feed
    model = load_net(model_src, flatbuff)
    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        _, frame = cap.read()
        frame = np.array(frame)
        height, width, _ = frame.shape
        frame = cv2.resize(frame, (512, 512))
        frame = frame.astype(np.float32)

        pred = predict(model, frame, flatbuff, normalize=True)
        pred = cv2.resize(pred, (int(width) // shrink, int(height) // 2))
        cv2.imshow('image', pred.astype(np.uint8))

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = arguments()

    if args.m == 'image':
        # single image pred
        from_img(args.f, args.s, args.flatbuff)

    else:
        # video feed
        from_video(args.f, args.s, args.flatbuff)
