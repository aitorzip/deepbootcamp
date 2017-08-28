#!/usr/bin/env python
from utils import SnapshotSaver
import click
import time
import os


@click.command()
@click.argument("dir")
def main(dir):
    env = None
    while True:
        saver = SnapshotSaver(dir)
        state = saver.get_state()
        if state is None:
            time.sleep(1)
            continue
        alg_state = state['alg_state']
        if env is None:
            env = alg_state['env_maker'].make()
        policy = alg_state['policy']
        ob = env.reset()
        done = False
        while not done:
            action, _ = policy.get_action(ob)
            ob, _, done, _ = env.step(action)
            env.render()


if __name__ == "__main__":
    main()
