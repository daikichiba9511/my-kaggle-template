# References:
# 1. https://hub.docker.com/r/nvidia/cuda/tags
# 2. https://github.com/astral-sh/uv-docker-example/blob/main/Dockerfile
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Tokyo

RUN apt update -y \
    && apt upgrade -yq \
    && apt install -yq --no-install-recommends \
    tzdata \
    sudo \
    git \
    vim \
    # for opencv \
    libgl1-mesa-dev

ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME
RUN useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
RUN mkdir -p /etc/sudoers.d
RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
RUN chmod 0440 /etc/sudoers.d/$USERNAME

ENV USER $USERNAME
USER $USERNAME

WORKDIR /workspace/working
ENV UV_LINK_MODE=copy
ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"
ENV PATH="/workspace/working/.venv/bin:${PATH}"