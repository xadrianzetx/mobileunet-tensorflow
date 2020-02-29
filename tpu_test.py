import cv2
import argparse
import numpy as np
from modules import tpu


def parser():
    p = argparse.ArgumentParser()

    p.add_argument('-m', type=str)
    p.add_argument('-v', type=str)
    p.add_argument('--quantstd', type=float, default=0.003921568859368563)
    p.add_argument('--dequantstd', type=float, default=0.00390625)

    return p.parse_args()


def main():
    args = parser()
    cap = cv2.VideoCapture(args.v)
    runtime = tpu.TPUBenchTest(model=args.m)

    while True:
        ret, frame = cap.read()

        if not ret:
            # no frame retrieved
            break

        frame = np.array(frame)
        frmcpy = frame.copy()

        # quantize model input, run inference and dequantize outputs
        frame = runtime.preprocess(frame, mean=0, std=args.quantstd)
        pred_obj = runtime.invoke(frame)
        print('[Inference time] {}ms'.format(runtime.inference_time))
        pred = runtime.postprocess(pred_obj, frmcpy, mean=0, std=args.dequantstd)

        # resize output frame to fit foo
        out = cv2.resize(pred, (pred.shape[1] // 2, pred.shape[0] // 2))
        cv2.imshow('TPU Benchmark', out)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            # exit on key press
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
