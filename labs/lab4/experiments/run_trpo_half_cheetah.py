#!/usr/bin/env python
import chainer

from algs import trpo
from env_makers import EnvMaker
from models import GaussianMLPPolicy, MLPBaseline
from utils import SnapshotSaver
import numpy as np
import os
import logger

log_dir = "data/local/trpo-half-cheetah"

np.random.seed(42)

# Clean up existing logs
os.system("rm -rf {}".format(log_dir))

with logger.session(log_dir):
    env_maker = EnvMaker('RoboschoolHalfCheetah-v1')
    env = env_maker.make()
    policy = GaussianMLPPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_spec=env.spec,
        hidden_sizes=(256, 64),
        hidden_nonlinearity=chainer.functions.tanh,
    )
    baseline = MLPBaseline(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_spec=env.spec,
        hidden_sizes=(256, 64),
        hidden_nonlinearity=chainer.functions.tanh,
    )
    trpo(
        env=env,
        env_maker=env_maker,
        n_envs=16,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        n_iters=5000,
        snapshot_saver=SnapshotSaver(log_dir, interval=10),
    )
