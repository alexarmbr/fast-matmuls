FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

RUN apt update && \
    apt install -y --no-install-recommends gnupg && \
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64 /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt update && \
    apt install -y nsight-systems-cli

