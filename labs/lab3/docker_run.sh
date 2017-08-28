#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
viskit_port=$("$DIR/findport.sh" 5000 1)
xhost=xhost
if hash nvidia-docker 2>/dev/null; then
    docker=nvidia-docker
else
    docker=docker
fi

if [[ $(uname) == 'Darwin' ]]; then
    # if xhost not defined, check 
    if ! hash $xhost 2>/dev/null; then
        xhost=/opt/X11/bin/xhost
        if [ ! -f $xhost ]; then
            echo "xhost not found!"
            exit
        fi
    fi
    ip=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
    $xhost + $ip >/dev/null
    $docker run --rm -p $viskit_port:$viskit_port -e VISKIT_PORT=$viskit_port \
        -e DISPLAY=$ip:0 \
        -v "$DIR":/root/code/bootcamp_pg \
        -ti dementrock/deeprlbootcamp \
          ${1-/bin/bash} "${@:2}"
    $xhost - $ip >/dev/null
elif [[ $(uname) == 'Linux' ]]; then
    $xhost +local:root >/dev/null
    $docker run --rm -p $viskit_port:$viskit_port -e VISKIT_PORT=$viskit_port \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "$DIR":/root/code/bootcamp_pg \
        -ti dementrock/deeprlbootcamp \
          ${1-/bin/bash} "${@:2}"
    $xhost -local:root >/dev/null
else
    echo "This script only supports macOS or Linux"
fi
