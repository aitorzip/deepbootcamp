#!/usr/bin/env python
from algs import trpo
from env_makers import EnvMaker
from models import CategoricalMLPPolicy, MLPBaseline
from utils import SnapshotSaver
import numpy as np
import os
import logger

log_dir = "data/local/trpo-cartpole"

np.random.seed(42)

# Clean up existing logs
os.system("rm -rf {}".format(log_dir))

with logger.session(log_dir):
    env_maker = EnvMaker('CartPole-v0')
    env = env_maker.make()
    policy = CategoricalMLPPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_spec=env.spec
    )
    baseline = MLPBaseline(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_spec=env.spec
    )
    trpo(
        env=env,
        env_maker=env_maker,
        n_envs=16,
        policy=policy,
        baseline=baseline,
        batch_size=2000,
        n_iters=100,
        snapshot_saver=SnapshotSaver(log_dir)
    )
