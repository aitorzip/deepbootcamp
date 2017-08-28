#!/usr/bin/env python

import cloudexec
from cloudexec import VariantGenerator
import numpy as np
from env_makers import EnvMaker
from models import MLPBaseline, TimeDependentBaseline, LinearFeatureBaseline
from models import GaussianMLPPolicy
from algs import trpo
from utils import SnapshotSaver
import chainer
import logger


def run(v):
    np.random.seed(v['seed'])
    env_maker = EnvMaker('Pendulum-v0')
    env = env_maker.make()
    policy = GaussianMLPPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=chainer.functions.tanh,
    )
    if v['baseline'] == 'mlp':
        baseline = MLPBaseline(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env_spec=env.spec,
            hidden_sizes=(64, 64),
            hidden_nonlinearity=chainer.functions.tanh,
        )
    elif v['baseline'] == 'time_dependent':
        baseline = TimeDependentBaseline(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env_spec=env.spec,
        )
    elif v['baseline'] == 'linear_feature':
        baseline = LinearFeatureBaseline(
            observation_space=env.observation_space,
            action_space=env.action_space,
            env_spec=env.spec,
        )
    else:
        raise ValueError
    trpo(
        env=env,
        env_maker=env_maker,
        n_envs=16,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        n_iters=100,
        snapshot_saver=SnapshotSaver(logger.get_dir()),
    )


vg = VariantGenerator()
vg.add("seed", [0, 100, 200])
vg.add("baseline", ['mlp', 'linear_feature', 'time_dependent'])

for variant in vg.variants():
    cloudexec.remote_call(
        task=cloudexec.Task(
            run,
            variant=variant,
        ),
        config=cloudexec.Config(
            exp_group="trpo-pendulum-baseline",
        ),
        mode=cloudexec.local_mode,
    )
