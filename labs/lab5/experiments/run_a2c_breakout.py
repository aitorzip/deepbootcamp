#!/usr/bin/env python
from algs import a2c
from env_makers import EnvMaker
from models import CategoricalCNNPolicy
from utils import SnapshotSaver
import numpy as np
import os
import logger

log_dir = "data/local/a2c-breakout"

np.random.seed(42)

# Clean up existing logs
os.system("rm -rf {}".format(log_dir))

with logger.session(log_dir):
    env_maker = EnvMaker('BreakoutNoFrameskip-v4')
    env = env_maker.make()
    policy = CategoricalCNNPolicy(
        env.observation_space, env.action_space, env.spec)
    vf = policy.create_vf()
    a2c(
        env=env,
        env_maker=env_maker,
        n_envs=16,
        policy=policy,
        vf=vf,
        snapshot_saver=SnapshotSaver(log_dir, interval=10),
    )
