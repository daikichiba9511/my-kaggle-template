FROM nvcr.io/nvidia/pytorch:24.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ Asia/Tokyo

RUN apt update -y \
    && apt upgrade -yq \
    && apt install -yq --no-install-recommends \
    tzdata \
    sudo \
    vim \
    # for opencv \
    libgl1-mesa-dev

ARG USERNAME=docker
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ENV USER $USERNAME
USER $USERNAME