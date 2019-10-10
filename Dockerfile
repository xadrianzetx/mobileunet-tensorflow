FROM nvidia/cuda:10.1-base-ubuntu16.04
RUN apt-get update && apt-get -y install python3-pip
RUN pip3 install -r requirements.txt
WORKDIR /app
COPY . /app
RUN mkdir /app/data
CMD ["python3", "main.py"]
