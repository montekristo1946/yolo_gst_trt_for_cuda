# yolo11, gstreemer and nvjpeg for gpu

install ubuntu 24.10

-------- install repository nvidia --------
sudo apt install software-properties-common
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update

-------- install driver ------------------------------
sudo ubuntu-drivers list
sudo apt install nvidia-driver-565

-------- install cuda ---------------------------------
sudo apt-get update
sudo apt-get install cuda-toolkit-12-6

-------- install cudnn 9.5.1 ---------------------------------
instruction from https://developer.nvidia.com/cudnn-downloads
sudo apt-get -y install cudnn9-cuda-12

-------- install opencv ---------------------------------
sudo apt  install cmake
export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'
sudo apt install gcc-13 g++-13

 apt install -y build-essential cmake pkg-config unzip yasm git checkinstall
 apt install -y libavcodec-dev libavformat-dev libswscale-dev
 apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
 apt install -y libxvidcore-dev libx264-dev libmp3lame-dev libopus-dev
 apt install -y libmp3lame-dev libvorbis-dev
 apt install -y ffmpeg
 apt install -y libva-dev
 apt install -y libdc1394-25 libdc1394-dev libxine2-dev libv4l-dev v4l-utils
 ln -s /usr/include/libv4l1-videodev.h /usr/include/linux/videodev.h
 apt install -y libgtk-3-dev
 apt install -y libtbb-dev
 apt install -y libatlas-base-dev gfortran
 apt install -y libprotobuf-dev protobuf-compiler
 apt install -y libgoogle-glog-dev libgflags-dev
 apt install -y libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
 apt install -y wget

cd /usr/local && mkdir opencv && cd opencv
wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.10.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.10.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
cd opencv-4.10.0 && mkdir build && cd build

cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=8.6 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=/usr/local/opencv/opencv_contrib-4.10.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D HAVE_opencv_python3=OFF \
-D BUILD_opencv_python=OFF \
-D BUILD_opencv_python3=OFF \
-D BUILD_opencv_python2=OFF \
-D BUILD_TESTS=OFF \
-D BUILD_EXAMPLES=OFF ..


make -j12
make install

-------- install tensorrt ---------------------------------
version="10.6.0.26-1+cuda12.6"

sudo apt-get install -y libnvinfer-dev=${version} libnvinfer-dispatch-dev=${version} libnvinfer-dispatch10=${version} libnvinfer-headers-dev=${version} libnvinfer-lean-dev=${version} libnvinfer-lean10=${version} libnvinfer-plugin-dev=${version} libnvinfer-plugin10=${version} libnvinfer-vc-plugin10=${version} libnvinfer10=${version} libnvonnxparsers-dev=${version} libnvonnxparsers10=${version} tensorrt-dev=${version}

sudo apt-get install -y libnvinfer-headers-plugin-dev=${version}
sudo apt-get install -y libnvinfer-vc-plugin-dev=${version}

-------- Install Dependencies -------------------------------
sudo apt install \
libssl3 \
libssl-dev \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev

download https://catalog.ngc.nvidia.com/orgs/nvidia/resources/deepstream/files
wget --content-disposition 'https://api.ngc.nvidia.com/v2/resources/org/nvidia/deepstream/7.1/files?redirect=true&path=deepstream-7.1_7.1.0-1_amd64.deb' -O deepstream-7.1_7.1.0-1_amd64.deb
sudo apt-get install -y ./deepstream-7.1_7.1.0-1_amd64.deb


cd /usr/lib/x86_64-linux-gnu && ln -s libyaml-cpp.so.0.7 /usr/lib/x86_64-linux-gnu/libyaml-cpp.0.8.0


-------- Install Dependencies on Docker images -------------------------------

