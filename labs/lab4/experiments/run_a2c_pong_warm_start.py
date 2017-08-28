#!/usr/bin/env python
from utils import SnapshotSaver
import numpy as np
import os
import logger
import pickle

log_dir = "data/local/a2c-pong-warm-start"

np.random.seed(42)

# Clean up existing logs
os.system("rm -rf {}".format(log_dir))

with logger.session(log_dir):
    with open("pong_warm_start.pkl", "rb") as f:
        state = pickle.load(f)
    saver = SnapshotSaver(log_dir, interval=10)
    alg_state = state['alg_state']
    env = alg_state['env_maker'].make()
    alg = state['alg']
    alg(env=env, snapshot_saver=saver, **alg_state)
