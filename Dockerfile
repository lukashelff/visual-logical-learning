# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.12-py3
#FROM python:3.9.7-slim
#FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
# installations for python
RUN apt-get update && apt-get install -y git build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev liblzma-dev python3-opencv

# install python
RUN mkdir /home/python
WORKDIR /home/python
RUN wget https://www.python.org/ftp/python/3.10.7/Python-3.10.7.tgz
RUN tar xzf Python-3.10.7.tgz
WORKDIR /home/python/Python-3.10.7
RUN ./configure --enable-optimizations
RUN make install

# installation SWI-Prolog
RUN apt-get update && apt-get install -y libprotobuf-dev protobuf-compiler cmake software-properties-common
RUN add-apt-repository ppa:swi-prolog/devel
RUN apt-get update && apt-get install -y swi-prolog

#WORKDIR /home
#RUN apt-get update && apt-get install -y build-essential cmake ninja-build pkg-config ncurses-dev libreadline-dev libedit-dev libgoogle-perftools-dev libgmp-dev libssl-dev unixodbc-dev zlib1g-dev libarchive-dev libossp-uuid-dev libxext-dev libice-dev libjpeg-dev libxinerama-dev libxft-dev libxpm-dev libxt-dev libdb-dev libpcre2-dev libyaml-dev default-jdk junit4
#RUN git clone https://github.com/SWI-Prolog/swipl-devel.git
##RUN git clone --depth 1 --branch V8.5.4 https://github.com/SWI-Prolog/swipl-devel.git
#WORKDIR /home/swipl-devel
#RUN git pull
#RUN git submodule update --init
#RUN mkdir build
#WORKDIR /home/swipl-devel/build
#RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
#RUN make
##RUN ctest -j 4
#RUN make install


# install yap
WORKDIR /home
RUN git clone --depth 1 https://github.com/vscosta/yap-6.3 yap
RUN mkdir /home/yap/BUILD
WORKDIR /home/yap/BUILD
RUN cmake ../
RUN make install

# install required python packages
WORKDIR /home/python
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