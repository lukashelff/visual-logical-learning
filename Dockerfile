# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.12-py3
#FROM python:3.9.7-slim
#FROM nvidia/cuda:11.4.0-runtime-ubuntu20.04
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
#FROM swipl:stable

RUN apt-get update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
# installations for python
RUN apt-get update && apt-get install -y git build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget libbz2-dev liblzma-dev python3-opencv

# install python
RUN mkdir /home/python
WORKDIR /home/python
RUN wget https://www.python.org/ftp/python/3.10.7/Python-3.10.7.tgz
RUN tar xzf Python-3.10.7.tgz
WORKDIR /home/python/Python-3.10.7
RUN ./configure --enable-optimizations
RUN make install

# install required python packages
WORKDIR /home/python
COPY requirements.txt /home/python/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN pip3 install -r /home/python/requirements.txt
RUN ln -s /usr/local/bin/python3.10 /usr/bin/python & ln -s /usr/local/bin/pip3.10 /usr/bin/pip


# installation SWI-Prolog
#RUN apt-get update && apt-get install -y software-properties-common
#RUN add-apt-repository "deb http://archive.ubuntu.com/ubuntu focal main universe restricted multiverse"
#RUN apt-get install -y libarchive13 libossp-uuid16 libgmp-dev libedit-dev libreadline-dev libncursesw5-dev libjs-jquery
#RUN dpkg -i swi-prolog-nox_8.4.3-1-g10c53c6e3-focalppa2.deb
RUN apt-get update && apt-get install -y libprotobuf-dev protobuf-compiler cmake software-properties-common
RUN add-apt-repository ppa:swi-prolog/stable
RUN apt-get update && apt-get install -y swi-prolog




# install yap
WORKDIR /home
RUN git clone --depth 1 https://github.com/vscosta/yap-6.3 yap
RUN mkdir /home/yap/BUILD
WORKDIR /home/yap/BUILD
RUN cmake ../
RUN make install


#RUN git clone https://github.com/SWI-Prolog/swipl-devel.git
#RUN git clone --depth 1 https://github.com/SWI-Prolog/swipl-devel.git
#RUN git pull
#RUN git submodule update --init
#RUN wget -q https://www.swi-prolog.org/download/stable/src/swipl-8.4.2.tar.gz
#RUN tar -xzf swipl-8.4.2.tar.gz
#WORKDIR /home/swipl-8.4.2
#RUN mkdir build
#WORKDIR /home/swipl-8.4.2/build
#RUN cmake -DCMAKE_BUILD_TYPE=PGO \
#    -DSWIPL_PACKAGES_X=OFF \
#	-DSWIPL_PACKAGES_JAVA=OFF \
#	-DCMAKE_INSTALL_PREFIX=$Home \
#    -G Ninja ..
#RUN ninja
#RUN ctest -j 4
#RUN ninja install


# installation SWI-Prolog
#WORKDIR /home
#RUN apt-get update && apt-get install -y build-essential cmake pkg-config software-properties-common
##    ncurses-dev libreadline-dev libedit-dev libgoogle-perftools-dev libgmp-dev libssl-dev unixodbc-dev zlib1g-dev  \
##    libarchive-dev libossp-uuid-dev libxext-dev libice-dev libjpeg-dev libxinerama-dev libxft-dev libxpm-dev  \
##    libxt-dev libdb-dev libpcre3 libyaml-dev default-jdk junit4 \
##    libtcmalloc-minimal4 \
##    ca-certificates \
##RUN add-apt-repository ppa:ubuntugis/ppa
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    libtcmalloc-minimal4 \
#    libarchive13 \
#    libyaml-dev \
#    libgmp10 \
#    libossp-uuid16 \
#    libssl1.1 \
#    ca-certificates \
#    libdb5.3 \
#    libpcre3 \
#    libedit2 \
#    libgeos++-dev libgeos-c1v5 libgeos-dev libgeos-doc \
#    libspatialindex6 \
#    unixodbc \
#    odbc-postgresql \
#    tdsodbc \
#    libmariadbclient-dev-compat \
#    libsqlite3-0 \
#    libserd-0-0 \
#    libraptor2-0 \
#    ninja-build gcc g++ wget git autoconf libarchive-dev libgmp-dev libossp-uuid-dev libpcre2-dev libreadline-dev libedit-dev libssl-dev zlib1g-dev libdb-dev unixodbc-dev libsqlite3-dev libserd-dev libraptor2-dev libgeos++-dev libspatialindex-dev libgoogle-perftools-dev libgeos-dev libspatialindex-dev



#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    libtcmalloc-minimal4 \
#    libarchive13 \
#    libyaml-dev \
#    libgmp10 \
#    libossp-uuid16 \
#    libssl1.1 \
#    ca-certificates \
#    libdb5.3 \
#    libpcre2-8-0 \
#    libedit2 \
#    libgeos++-dev \
#    libspatialindex6 \
#    unixodbc \
#    odbc-postgresql \
#    tdsodbc \
#    libmariadbclient-dev-compat \
#    libsqlite3-0 \
#    libserd-0-0 \
#    libraptor2-0 && \
#    dpkgArch="$(dpkg --print-architecture)" && \
#    rm -rf /var/lib/apt/lists/*
#ENV LANG C.UTF-8
#RUN set -eux; \
#    SWIPL_VER=9.0.2; \
#    SWIPL_CHECKSUM=33b5de34712d58f14c1e019bd1613df9a474f5e5fd024155a0f6e67ebb01c307; \
#    BUILD_DEPS='make cmake ninja-build gcc g++ wget git autoconf libarchive-dev libgmp-dev libossp-uuid-dev libpcre2-dev libreadline-dev libedit-dev libssl-dev zlib1g-dev libdb-dev unixodbc-dev libsqlite3-dev libserd-dev libraptor2-dev libgeos++-dev libspatialindex-dev libgoogle-perftools-dev libgeos-dev libspatialindex-dev'; \
#    dpkgArch="$(dpkg --print-architecture)"; \
#    apt-get update; apt-get install -y --no-install-recommends $BUILD_DEPS; rm -rf /var/lib/apt/lists/*; \
#    mkdir /tmp/src; \
#    cd /tmp/src; \
#    wget -q https://www.swi-prolog.org/download/stable/src/swipl-$SWIPL_VER.tar.gz; \
#    echo "$SWIPL_CHECKSUM  swipl-$SWIPL_VER.tar.gz" >> swipl-$SWIPL_VER.tar.gz-CHECKSUM; \
#    sha256sum -c swipl-$SWIPL_VER.tar.gz-CHECKSUM; \
#    tar -xzf swipl-$SWIPL_VER.tar.gz; \
#    mkdir swipl-$SWIPL_VER/build; \
#    cd swipl-$SWIPL_VER/build; \
#    cmake -DCMAKE_BUILD_TYPE=PGO \
#          -DSWIPL_PACKAGES_X=OFF \
#	  -DSWIPL_PACKAGES_JAVA=OFF \
#	  -DCMAKE_INSTALL_PREFIX=/usr \
#	  -G Ninja \
#          ..; \
#    ninja; \
#    ninja install; \
#    rm -rf /tmp/src; \
#    mkdir -p /usr/share/swi-prolog/pack; \
#    cd /usr/share/swi-prolog/pack; \
#    # usage: install_addin addin-name git-url git-commit
#    install_addin () { \
#        git clone "$2" "$1"; \
#        git -C "$1" checkout -q "$3"; \
#        # the prosqlite plugin lib directory must be removed?
#        if [ "$1" = 'prosqlite' ]; then rm -rf "$1/lib"; fi; \
#        swipl -g "pack_rebuild($1)" -t halt; \
#        find "$1" -mindepth 1 -maxdepth 1 ! -name lib ! -name prolog ! -name pack.pl -exec rm -rf {} +; \
#        find "$1" -name .git -exec rm -rf {} +; \
#        find "$1" -name '*.so' -exec strip {} +; \
#    }; \
#    dpkgArch="$(dpkg --print-architecture)"; \
#    install_addin space https://github.com/JanWielemaker/space.git 8ab230a67e2babb3e81fac043512a7de7f4593bf; \
#    install_addin prosqlite https://github.com/nicos-angelopoulos/prosqlite.git cfd2f68709f5fb61833c0e2f8e9c6546e542009c; \
#    [ "$dpkgArch" = 'armhf' ] || [ "$dpkgArch" = 'armel' ] || install_addin rocksdb https://github.com/JanWielemaker/rocksdb.git 634c31e928e2a5100fbcfd26c21cd32eeb6bf369; \
#    [ "$dpkgArch" = 'armhf' ] || [ "$dpkgArch" = 'armel' ] ||  install_addin hdt https://github.com/JanWielemaker/hdt.git e0a0eff87fc3318434cb493690c570e1255ed30e; \
#    install_addin rserve_client https://github.com/JanWielemaker/rserve_client.git 48a46160bc2768182be757ab179c26935db41de7; \
#    apt-get purge -y --auto-remove $BUILD_DEPS




EXPOSE 8282
# create workdir
RUN mkdir /home/workdir
WORKDIR /home/workdir


ENV PYTHONPATH "${PYTHONPATH}:./"
ENV NVIDIA_VISIBLE_DEVICES all