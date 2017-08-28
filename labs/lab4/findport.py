#!/usr/bin/env python
#
# Usage: findport.py 3000 100
#
from __future__ import print_function
import socket
from contextlib import closing
import sys

if len(sys.argv) != 3:
    print("Usage: {} <base_port> <increment>".format(sys.argv[0]))
    sys.exit(1)

base = int(sys.argv[1])
increment = int(sys.argv[2])


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        for port in range(base, 65536, increment):
            try:
                s.bind(('', port))
                return s.getsockname()[1]
            except socket.error:
                continue


print(find_free_port())
