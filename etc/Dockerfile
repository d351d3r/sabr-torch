# Build and run:
# docker build -t clion/remote-cpp-env:1.0 -f Dockerfile .
# docker run -d --volume $HOME/project:/project  --cap-add sys_ptrace -p127.0.0.1:2222:22 --name clion_remote_env clion/remote-cpp-env:latest
FROM ubuntu:latest

RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get -y install tzdata
RUN mkdir /var/run/sshd


RUN apt-get update \
  && apt-get install -y build-essential \
      ssh \
      gcc \
      g++ \
      gdb \
      clang \
      make \
      ninja-build \
      cmake \
      autoconf \
      automake \
      libtool \
      valgrind \
      locales-all \
      dos2unix \
      rsync \
      tar \
      python3 \
      python3-dev \
      git \
      sudo \
  && apt-get clean

#ARG UID=1000
#RUN useradd -m -u ${UID} -s /bin/bash builder
#USER builder


RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_development # \
#  && mkdir /run/sshd

ENV DEBIAN_FRONTEND=keyboard-interactive
RUN useradd -m remote && yes password | passwd remote
RUN usermod -aG sudo remote
RUN mkdir /project
CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_development"]
