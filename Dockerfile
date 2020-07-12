FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN ["pip", "install", "nltk"]
WORKDIR /usr/src/
COPY requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"]