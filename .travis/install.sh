#!/usr/bin/env bash
set -x
set -e

function installswig() {
    # Need SWIG >= 3.0.8
    cd /tmp/ &&
        wget https://github.com/swig/swig/archive/rel-3.0.12.tar.gz &&
        tar zxf rel-3.0.12.tar.gz && cd swig-rel-3.0.12 &&
        ./autogen.sh && ./configure --prefix "${HOME}" 1>/dev/null &&
        make >/dev/null &&
        make install >/dev/null
}

function installcmake() {
    cd /tmp/ &&
        wget https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.tar.gz &&
        tar zxf cmake-3.17.0-Linux-x86_64.tar.gz &&
        mkdir -p $TRAVIS_BUILD_DIR/cmake &&
        cp -r cmake-3.17.0-Linux-x86_64/* $TRAVIS_BUILD_DIR/cmake
}

if [ "${TRAVIS_OS_NAME}" == linux ]; then
    installswig
    installcmake
fi
