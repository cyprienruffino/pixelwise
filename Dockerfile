FROM nvidia/cuda:8.0-cudnn6-devel

MAINTAINER Cyprien Ruffino <ruffino.cyprien@gmail.com>

RUN apt-get update && apt-get install -y \
    cmake \
    cpio \
    gcc \
    g++ \
    gfortran \
    wget \
    build-essential \
    kmod \
    git \
    wget \
    python\
    python3\
    liblapack-dev \
    libopenblas-dev \
    python-dev \
    python-pip \
    python-nose \
    python-numpy \
    python-scipy \
    python-h5py \
    python-yaml \
    python3-dev \
    python3-pip \
    python3-nose \
    python3-numpy \
    python3-scipy \
    libhdf5-dev \
    python3-h5py \
    python3-yaml

RUN pip3 install cython \
    pandas \
    numpy \
    h5py \
    Pillow \
    scipy \
    sklearn \
    progressbar2 \
    Theano \
    keras==2.0.6 

RUN pip3 install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

# Installing libgpuarray
RUN git clone https://github.com/Theano/libgpuarray.git && \
    cd libgpuarray && \
    mkdir Build && cd Build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make && make install && cd .. && \
    python3 setup.py build && \
    python3 setup.py install


RUN mkdir /root/.keras && \
    echo '{ \n\
        "image_data_format": "channels_last",\n\
        "floatx": "float32",\n\
        "backend": "theano",\n\
        "epsilon": 1e-07\n\
    }' > /root/.keras/keras.json

RUN echo '[cuda]\n\
root=/usr/local/cuda-8.0\n\
[global]\n\
device=cuda0\n\
floatX=float32\n\
' > /root/.theanorc

RUN rm -rf /usr/lib/python3/dist-packages/numpy

RUN pip3 install numpy
