# syntax=docker/dockerfile:1-labs

FROM ubuntu:22.04

WORKDIR /root

RUN apt update \
    && apt install flex bison libssl-dev libelf-dev bc vim python3 python3-pip lsb-release wget software-properties-common gnupg -y \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install CodeChecker==6.21.0 tensorflow==2.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN wget https://apt.llvm.org/llvm.sh \
    && chmod +x llvm.sh \
    && ./llvm.sh 15 \
    && ln -s /usr/bin/clang-15 /usr/bin/clang \
    && ln -s /usr/bin/clang++-15 /usr/bin/clang++

ADD ./ /root/Goshawk/