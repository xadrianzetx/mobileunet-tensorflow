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
