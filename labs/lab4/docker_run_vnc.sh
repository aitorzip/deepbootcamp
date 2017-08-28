#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
vnc_port=$("$DIR/findport.py" 3000 1)
viskit_port=$("$DIR/findport.py" 5000 1)
if hash nvidia-docker 2>/dev/null; then
    docker=nvidia-docker
else
    docker=docker
fi

echo "Connect to this VNC address to view the display: localhost:$vnc_port Password: 3284"
$docker run --rm -p $vnc_port:5900 -p $viskit_port:$viskit_port -e VISKIT_PORT=$viskit_port \
    -v "$DIR":/root/code/bootcamp_pg \
    -ti dementrock/deeprlbootcamp \
      ./launch_bg_screen_buffer.sh ${1-/bin/bash} "${@:2}"
