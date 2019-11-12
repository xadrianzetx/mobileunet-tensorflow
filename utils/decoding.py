import os
import cv2
import numpy as np


def mask_from_splines(line, mask):
    """
    Builds segmentation masks from spline coordinates

    :param line:    str
                    spline coordinates for single line
    :param mask:    np.ndarray HxW
                    base mask filled with 0 and shaped as original image

    :return:        np.ndarray HxW
                    mask updated with segmented lines
    """
    line = line.split()

    # ground truth file is similar as in run length encoding
    # odd indices are x and even are y spline coordinates
    x_coords = [round(float(x)) for x in line[0:][::2]]
    y_coords = [int(y) for y in line[1:][::2]]

    # remove negative and sort by x coordinates
    x_coords, y_coords = zip(*((x, y) for x, y in zip(x_coords, y_coords) if x > 0))
    xs_coords, ys_coords = (np.array(list(x)) for x in zip(*sorted(zip(x_coords, y_coords))))

    # interpolate between spline points to form line
    x_interp = np.linspace(xs_coords.min(), xs_coords.max(), num=500).astype(int)
    y_interp = np.interp(x_interp, xs_coords, ys_coords).astype(int)

    for x, y in zip(x_interp, y_interp):
        cv2.circle(mask, (x, y), 5, (1), -1)

    return mask


def image_mask_overlay(img, mask, threshold=0.5):
    """
    Overlays masks over original img

    :param img:     np.ndarray HxWxC
                    original image
    :param mask:    np.ndarray HxW
                    mask

    :return:        np.ndarray HxWxC
                    original img with mask overlay
    """
    img[mask > threshold] = (255, 0, 255)

    return img


def extract_frames(src, dst, n=3):
    """
    Extracts every nth frame from video as .jpg

    :param src: str
                path to video source
    :param dst: str
                path to save directory
    :param n:   int
                every nth frame will be saved

    :return:    void
    """
    frame_count = 0
    name = os.path.splitext(os.path.basename(src))[0]

    # init capture
    cap = cv2.VideoCapture(src)

    while cap.isOpened():
        fetched, frame = cap.read()

        if not fetched:
            # no frames has been grabbed
            break

        if frame_count % n == 0:
            # save every nth frame
            frame_name = '{}_{}.jpg'.format(name, frame_count)
            output_path = os.path.join(dst.replace('/', os.path.sep), frame_name)
            cv2.imwrite(output_path, frame)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
