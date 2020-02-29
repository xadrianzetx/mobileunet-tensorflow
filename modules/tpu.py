import cv2
import numpy as np
from edgetpu.basic.basic_engine import BasicEngine


class TPUBenchTest:

    def __init__(self, model, env='Windows'):
        self._env = env
        self._model = model
        self._interpreter = self._get_interpreter()

    @property
    def input_size(self):
        _, *size, _ = self._interpreter.get_input_tensor_shape()

        return size

    @property
    def inference_time(self):
        return self._interpreter.get_inference_time()

    def _get_interpreter(self):
        try:
            interpreter = BasicEngine(self._model)

        except RuntimeError:
            raise RuntimeError('TPU not detected')

        return interpreter

    def preprocess(self, img, mean, std):
        """
        Perform preprocessing steps on image

        Input is quantized with passed params and
        cast to uint8 to run on edgetpu

        :param img:     np.array HxWxC
                        input image
        :param mean:    int
                        post-quantization mean
                        img input should be 0 centered
        :param std:     float
                        post-quantization standard deviation

        :return:        np.array BxHxWxC
                        preprocessed image input
        """
        _, *size, _ = self._interpreter.get_input_tensor_shape()
        img = cv2.resize(img, tuple(size))
        img = img.astype(np.float32)

        img *= (1 / 255)
        expd = np.expand_dims(img, axis=0)
        quantized = (expd / std + mean)

        return quantized.astype(np.uint8)

    def postprocess(self, pred_obj, frame, mean, std, upsample=True):
        """
        Perform postprocessing steps on image

        egetpu.BasicEngine inference output (img mask) is dequantized and
        postprocessed with morphology filters. Mask is then overlaid with
        original image to form final prediction

        :param pred_obj:    list
                            edgetpu.BasicEngine.run_inference() output
        :param frame:       np.array HxWxC
                            raw input image
        :param mean:        int
                            post-quantization mean
                            img input should be 0 centered
        :param std:         float
                            post-quantization standard deviation
        :param upsample:    bool
                            if true, morph filters will be used to
                            smooth out upsampled mask

        :return:            np.array BxHxWxC
                            preprocessed image input
        """
        _, *size, _ = self._interpreter.get_input_tensor_shape()
        pred = pred_obj[1].reshape((size))

        # dequantize and cast back to float
        dequantized = (std * (pred - mean))
        dequantized = dequantized.astype(np.float32)
        mask = cv2.resize(dequantized, (frame.shape[1], frame.shape[0]))

        if upsample:
            # perform closing operation on mask to smooth out lane edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel, iterations=4)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

        frame[mask != 0] = (255, 0, 255)

        return frame

    def invoke(self, img):
        """
        Runs inference on edgetpu

        :param img: np.array BxHxWxC
                    preprocessed image input

        :return:    list
                    pred object
        """
        img = img.flatten()
        return self._interpreter.run_inference(img)
