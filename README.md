# MobileUnet

Lane segmentation model trained with tensorflow implementation of MobileNetV2 based U-Net

<p align="center">
<image src="https://github.com/xadrianzetx/fast-scnn-tensorflow/blob/master/gifs/nightride.gif"></image>
</p>

## Setup

Training environment is contenerized, however you still need some stuff installed on host machine. This includes:

* Docker
* NVIDIA driver package
* nvidia-docker

This will allow to take full advantage of host GPUs from docker container.

If you are running Ubuntu 16.04 then use ```envsetup.sh``` which will take care of setting up your host with all necessities.

```
sudo dos2unix envsetup.sh && \ 
sudo chmod -x envsetup.sh && \
sudo bash envsetup.sh
```

## Training

First, build docker image with

```
docker build . --tag mobileunet:v1
```

then you can start training with

```
docker run -d --gpus all -v $(pwd)/data:/app/data \
-e --mode="train" \
-e --data-train=${DATA_DIR} \
-e --epochs=${EPOCHS} \
-e --model-name=${MODEL_NAME} mobileunet:v1
```
Container has volume mapped to ```$(pwd)/data``` so make sure such path exists on your host. This directory should also contain training data. Baseline for this model was trained on [CU Lane Dataset](https://xingangpan.github.io/projects/CULane.html). Tensorboard files and json logs will be available at ```$(pwd)/data/logs```, model checkpoints and .h5 are saved to ```$(pwd)/data/models```.

## Testing

Run ```python test.py -m image -f ${IMG_PATH} -s ${SAVEPOINT}``` to test the model. You can pass ```-m image``` to run model on .jpg or ```-m video``` to run it on video feed. FlatBuffer model (.tflite) supported by passing ```--flatbuff```

# References
* [Ronneberger et al., 2015, U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)

* [Sandler et al., 2019, MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/pdf/1801.04381.pdf)

* [Pan et al., 2018, Spatial As Deep: Spatial CNN for Traffic Scene Understanding](https://xingangpan.github.io/projects/CULane.html)