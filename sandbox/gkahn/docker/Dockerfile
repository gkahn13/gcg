FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04
# FROM ubuntu:16.04
# FROM tensorflow/tensorflow:0.12.1-gpu

# ========== Anaconda ==========
# https://github.com/ContinuumIO/docker-images/blob/master/anaconda/Dockerfile
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget  --no-check-certificate --quiet https://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh && \
    /bin/bash /Anaconda2-2.5.0-Linux-x86_64.sh -b -p /opt/conda && \
    rm /Anaconda2-2.5.0-Linux-x86_64.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH
# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8
ENTRYPOINT [ "/usr/bin/tini", "--" ]

# ========== Special Deps ==========
RUN apt-get -y install git make cmake unzip
RUN pip install awscli
# ALE requires zlib
RUN apt-get -y install zlib1g-dev
# MUJOCO requires graphics stuff (Why?)
RUN apt-get -y build-dep glfw
RUN apt-get -y install libxrandr2 libxinerama-dev libxi6 libxcursor-dev
# copied from requirements.txt
#RUN pip install imageio tabulate nose
RUN apt-get install -y vim ack-grep
RUN pip install --upgrade pip
# usual pip install pygame will fail
RUN apt-get build-dep -y python-pygame
RUN pip install Pillow

# ========== OpenAI Gym ==========
RUN apt-get -y install libgtk2.0-0
RUN pip install gym
#RUN apt-get -y install ffmpeg
RUN apt-get -y install libav-tools
CMD alias ffmpeg="avconv"

# ========== Add codebase stub ==========
CMD mkdir /root/code
ADD environment.yml /root/code/environment.yml
RUN conda env create -f /root/code/environment.yml

ENV PYTHONPATH /root/code/rllab:$PYTHONPATH
ENV PATH /opt/conda/envs/rllab3/bin:$PATH
RUN echo "source activate rllab3" >> /root/.bashrc
ENV BASH_ENV /root/.bashrc
WORKDIR /root/code

# gpu theanno
ENV THEANO_FLAGS mode=FAST_RUN,device=cpu,floatX=float32
RUN pip install gtimer

# Upgrade tensorflow
RUN pip uninstall tensorflow -y
# RUN pip install tensorflow
# RUN pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl || pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl
# RUN pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
# RUN pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp35-cp35m-linux_x86_64.whl

# ========== Install Tensorflow ==========
#RUN apt-get update && apt-get install -y --no-install-recommends \
#        build-essential \
#        curl \
#        libfreetype6-dev \
#        libpng12-dev \
#        libzmq3-dev \
#        pkg-config \
#        python \
#        python-dev \
#        rsync \
#        software-properties-common \
#        unzip \
#        && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

#RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py

#RUN pip --no-cache-dir install \
#        ipykernel \
#        jupyter \
#        matplotlib \
#        numpy \
#        scipy \
#        sklearn \
#        && \
#    python -m ipykernel.kernelspec

RUN pip --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl

# Create correct path for libuda so Tensorflow can open it
# ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64
# ENV CUDA_HOME /usr/local/cuda
# ENV PATH $PATH:$CUDA_HOME/bin
# RUN ln -s /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so





RUN pip install lxml
RUN pip install cloudpickle
RUN pip install --upgrade gym

# ========== Install PLE and gym-ple ==========
RUN git clone https://github.com/ntasfi/PyGame-Learning-Environment.git ple
RUN pip install -e ple
RUN git clone https://github.com/gkahn13/gym-ple.git gym-ple
RUN pip install -e gym-ple

# RUN apt-get -y install alsa-utils
# RUN echo "export TERM=xterm" >> /root/.bashrc

# ========== RNN critic dependencies ==========
RUN pip install pandas
