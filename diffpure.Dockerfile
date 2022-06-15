FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        zlib1g-dev \
        swig \
        cmake \
        vim \
        locales \
        locales-all \
        screen \
        zip \
        unzip \
        python3-tk \
        xvfb \
        ghostscript
RUN apt-get clean

ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip setuptools

RUN pip install numpy==1.16.4 \
                pyyaml==5.3.1 \
                wheel==0.34.2 \
                scipy==1.5.2 \
                torch==1.7.1 \
                torchvision==0.8.2 \
                pillow==7.2.0 \
                matplotlib==3.3.0 \
                tqdm==4.46.0 \
                tensorboardX==2.0 \
                seaborn==0.10.1 \
                pandas==1.1.0 \
                requests==2.24.0 \
                xvfbwrapper==0.2.9 \
                torchdiffeq==0.2.1 \
                timm==0.5.4 \
                foolbox \
                torchsde \
                git+https://github.com/RobustBench/robustbench.git \