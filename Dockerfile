FROM tensorflow/tensorflow:1.15.0-gpu-py3
ARG HOME=/app
WORKDIR ${HOME}
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
