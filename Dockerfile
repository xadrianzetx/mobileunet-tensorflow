FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04
RUN apt-get update && apt-get -y install python3-pip && pip3 install -U pip
WORKDIR /app
COPY . /app
RUN pip3 install -r requirements.txt
RUN mkdir /app/data
EXPOSE 6006
CMD ["python3", "train.py"]
