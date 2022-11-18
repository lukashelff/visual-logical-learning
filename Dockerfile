# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.12-py3
#FROM python:3.9.7-slim
#FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
# installations for python
RUN apt-get update && apt-get install -y git build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev liblzma-dev python3-opencv
# install python and required packages
RUN mkdir /home/python
WORKDIR /home/python
RUN wget https://www.python.org/ftp/python/3.10.7/Python-3.10.7.tgz
RUN tar xzf Python-3.10.7.tgz
WORKDIR /home/python/Python-3.10.7
RUN ./configure --enable-optimizations
RUN make install

COPY requirements.txt /home/python/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN pip3 install -r /home/python/requirements.txt
RUN ln -s /usr/local/bin/python3.10 /usr/bin/python & ln -s /usr/local/bin/pip3.10 /usr/bin/pip



EXPOSE 8282
# create workdir
RUN mkdir /home/workdir
WORKDIR /home/workdir


ENV PYTHONPATH "${PYTHONPATH}:./"
ENV NVIDIA_VISIBLE_DEVICES all