FROM python:3.7.8-buster

LABEL maintainer="Vitali Shchutski and Cedric Soares"


#Linux libraries install
RUN \
    apt-get update \
    && apt-get install -y \
    apt-utils \
    software-properties-common \
    && add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" \
    && apt-get install -y \
    autoconf \
    automake \
    cmake \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtool \
    build-essential \
    libopencv-dev \
    libv4l-dev \ 
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    openexr \
    libgtk-3-dev \
    gfortran \
    libatlas-base-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    git \
    nano \
    wget \
    unzip \
    poppler-utils \
    && python -m pip install --upgrade pip \
    && apt-get autoremove \
    && apt-get clean -y 

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt 

#OpenCV install
WORKDIR /opt

RUN \
    git clone https://github.com/opencv/opencv.git \
    && git clone https://github.com/opencv/opencv_contrib.git \
    && cd opencv \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D WITH_CUDA=OFF \
    -D BUILD_opencv_python3=yes \
    -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D BUILD_EXAMPLES=ON .. \
    && make -j $(nproc) \
    && make install 

WORKDIR /app

EXPOSE 8501

VOLUME .

ENTRYPOINT ["/bin/bash"]

CMD ["streamlit", "run", "app.py"]
