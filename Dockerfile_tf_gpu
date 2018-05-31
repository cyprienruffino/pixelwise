FROM nvidia/cuda:8.0-cudnn6-devel
FROM tensorflow/tensorflow:1.4.1-devel-gpu-py3

MAINTAINER Cyprien Ruffino <ruffino.cyprien@gmail.com>

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    gfortran \
    kmod \
    git \
    wget \
    python3\
    liblapack-dev \
    libopenblas-dev \
    python3-dev \
    python3-pip \
    python3-nose \
    python3-numpy \
    python3-scipy \
    libhdf5-dev \
    python3-h5py \
    python3-yaml

RUN pip3 install pandas \
    numpy \
    h5py \
    Pillow \
    scipy \
    sklearn \
    progressbar2 \
    keras==2.0.6
