# syntax=docker/dockerfile:1-labs

FROM ubuntu:22.04

WORKDIR /root


RUN apt update \
    && apt install flex bison libssl-dev libelf-dev bc vim python3 python3-pip lsb-release wget software-properties-common gnupg unzip -y \
    && apt clean \
    && apt autoremove \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install "Cython<3" -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install PyYAML==5.4.1 --no-cache-dir --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip3 install CodeChecker==6.21.0 tensorflow==2.11.0 -i https://pypi.tuna.tsinghua.edu.cn/simple \
    && wget https://yunlongs-1253041399.cos.ap-chengdu.myqcloud.com/clang+llvm-15.0.0-x86_64-linux-gnu-rhel-8.4.tar.xz \
    && tar -xvf clang+llvm-15.0.0-x86_64-linux-gnu-rhel-8.4.tar.xz \
    && rm clang+llvm-15.0.0-x86_64-linux-gnu-rhel-8.4.tar.xz \
    && echo "export PATH=/root/clang+llvm-15.0.0-x86_64-linux-gnu-rhel-8.4/bin:\$PATH" >> ~/.bashrc

# The version of clang installed via llvm.sh is 15.0.7, not 15.0.0!
#RUN wget https://apt.llvm.org/llvm.sh \
#    && chmod +x llvm.sh \
#    && ./llvm.sh 15 \
#    && ln -s /usr/bin/clang-15 /usr/bin/clang \
#    && ln -s /usr/bin/clang++-15 /usr/bin/clang++

ADD ./ /root/Goshawk/