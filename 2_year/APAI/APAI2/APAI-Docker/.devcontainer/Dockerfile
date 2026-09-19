# Use Ubuntu 20.04 (Focal) where GCC 9 is native
FROM ubuntu:20.04

# Avoid tzdata prompts and speed up apt
ENV DEBIAN_FRONTEND=noninteractive

# Install base, build tools, gcc-9/g++-9 (optional), and additional system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        make \
        cmake \
        git \
        pkg-config \
        python3 \
        python3-pip \
        # gcc-9 g++-9 \
        libftdi-dev \
        libftdi1 \
        doxygen \
        libsdl2-dev \
        curl \
        libusb-1.0-0-dev \
        scons \
        gtkwave \
        libsndfile1-dev \
        rsync \
        autoconf \
        automake \
        texinfo \
        libtool \
        libsdl2-ttf-dev \
        wget \
    && gcc --version && g++ --version && cmake --version \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional: create a non-root user (uncomment if you want it)
# ARG USER=dev
# ARG UID=1000
# ARG GID=1000
# RUN groupadd -g ${GID} ${USER} && \
#     useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USER}
# USER ${USER}

# Python packages
RUN python3 -m pip install --no-cache-dir \
        argcomplete \
        pyelftools \
        torch torchvision torchaudio \
        # torch torchvision --index-url https://download.pytorch.org/whl/cpu \
        numpy \
        jupyterlab

# Work directory
WORKDIR /workspace
