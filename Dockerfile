# syntax=docker/dockerfile:1-labs

FROM python:3.9

ADD https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.0/clang+llvm-15.0.0-x86_64-linux-gnu-rhel-8.4.tar.xz /root/
ADD https://github.com/Yunlongs/Goshawk.git /root/goshawk/

RUN pip3 install CodeChecker==6.21.0 tensorflow==2.11.0 \
    && tar xf /root/clang+llvm-15.0.0-x86_64-linux-gnu-rhel-8.4.tar.xz --directory=/root \
    && rm /root/clang+llvm-15.0.0-x86_64-linux-gnu-rhel-8.4.tar.xz \
    && echo "export PATH=/root/clang+llvm-15.0.0-x86_64-linux-gnu-rhel-8.4/bin:$PATH" >> ~/.bashrc \
    && apt update \
    && apt install flex bison libssl-dev libelf-dev bc vim -y \
    && apt clean
