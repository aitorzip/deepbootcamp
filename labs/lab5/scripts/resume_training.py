#!/usr/bin/env python
from utils import SnapshotSaver
import click
import logger


@click.command()
@click.argument("dir")  # , "Directory which contains snapshot files")
@click.option("--interval", help="Interval between saving snapshots", type=int, default=10)
def main(dir, interval):
    with logger.session(dir):
        saver = SnapshotSaver(dir, interval=interval)
        state = saver.get_state()
        alg_state = state['alg_state']
        env = alg_state['env_maker'].make()
        alg = state['alg']
        alg(env=env, snapshot_saver=saver, **alg_state)


if __name__ == "__main__":
    main()
