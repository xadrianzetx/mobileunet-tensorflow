FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && \
    apt-get -y install python3-pip && \
    pip3 install -U pip && \
    apt-get -y install libglib2.0-0 libsm6 libxext6 libxrender-dev

# fix for https://github.com/tensorflow/tensorflow/issues/26182
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib/:/usr/local/cuda-10.0/lib64/

WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt
RUN mkdir /app/data

CMD ["python3", "train.py"]
