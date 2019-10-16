import cv2
import argparse
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from utils.decoding import image_mask_overlay
from modelzoo.losses import focal_tversky_loss
from modelzoo.metrics import dice_coefficient


def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', type=str, default='image')
    parser.add_argument('-f', type=str, default='img.jpg')
    parser.add_argument('-s', type=str, default='model.h5')

    return parser.parse_args()


def load_net(src):
    obj = {'focal_tversky': focal_tversky_loss(alpha=0.7, beta=0.3, gamma=0.75), 'dice': dice_coefficient()}
    model = load_model(src, custom_objects=obj)

    return model


def predict(model, img, normalize=True):
  original = img.copy()
  
  if normalize:
    img *= (1 / 255)
  
  # add batch channel and run inference
  # then round probas to classes with threshold 0.5
  pred = model.predict(np.expand_dims(img, axis=0))
  mask = np.round(pred[0, :, : ,0])
  out = image_mask_overlay(original, mask)
  
  return out


def from_img(src, model_src, shrink=2):
    # lane segmentation on single frame
    # load image and resize to size expected by net
    img = Image.open(src)
    img = np.array(img)
    height, width, _ = img.shape
    img = cv2.resize(img, (512, 512))
    img = img.astype(np.float32)
    
    # run inference
    model = load_net(model_src)
    pred = predict(model, img, normalize=True)

    # resize back to original shape and show
    pred = cv2.resize(pred, (int(width) // shrink, int(height) // 2))
    cv2.imshow('image', pred.astype(np.uint8))
    cv2.waitKey(0)


def from_video(src, model_src, shrink=2):
    # lane segmentation on video feed
    model = load_net(model_src)
    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        _, frame = cap.read()
        frame = np.array(frame)
        height, width, _ = frame.shape
        frame = cv2.resize(frame, (512, 512))
        frame = frame.astype(np.float32)

        pred = predict(model, frame, normalize=True)
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
        from_img(args.f, args.s)
    
    else:
        # video feed
        from_video(args.f, args.s)
