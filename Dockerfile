FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04

RUN apt-get update && \
        apt-get -y install python3-pip && \
        pip3 install -U pip && \
        apt-get -y install libglib2.0-0 && \
        apt-get install -y libsm6 libxext6 libxrender-dev

WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt
RUN mkdir /app/data

CMD ["python3", "train.py"]
