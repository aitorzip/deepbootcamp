#!/bin/bash

killall() {
kill -INT "$xvfb_pid" 
kill -INT "$x11vnc_pid" 
exit
}

trap killall SIGINT
trap killall SIGTERM
trap killall SIGKILL

Xvfb :99 -screen 0 1024x768x24 -ac  +extension GLX +render +extension RANDR -noreset & export xvfb_pid=$!

mkdir ~/.x11vnc
x11vnc -storepasswd 3284 ~/.x11vnc/passwd

command="${1-/bin/bash} ${@:2}"

env DISPLAY=:99.0 x11vnc -q -nopw -ncache 10 -forever -rfbauth ~/.x11vnc/passwd -display :99 2>/dev/null >/dev/null & export x11vnc_pid="$!"

DISPLAY=:99 $command

killall
