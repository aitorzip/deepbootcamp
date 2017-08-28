#!/usr/bin/env python

import cloudexec
import numpy as np
from env_makers import EnvMaker
from models import MLPBaseline, CategoricalMLPPolicy
from algs import trpo
from utils import SnapshotSaver
import logger


def run(v):
    np.random.seed(v['seed'])
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
        snapshot_saver=SnapshotSaver(logger.get_dir())
    )


cloudexec.remote_call(
    task=cloudexec.Task(
        run,
        variant=dict(seed=0),
    ),
    config=cloudexec.Config(
        exp_group="trpo-cartpole",
    ),
    mode=cloudexec.local_mode,
)
