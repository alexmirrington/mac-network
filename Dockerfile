FROM tensorflow/tensorflow:1.15.0-gpu-py3
ARG HOME=/usr/src/mac-network
WORKDIR ${HOME}
RUN curl http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip -o glove.zip && \
    mkdir -p data/glove && \
    unzip glove.zip -d data/glove && \
    rm glove.zip
COPY . .
RUN pip install -r requirements.txt
