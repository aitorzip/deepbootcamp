#!/usr/bin/env python
import click
import numpy as np
import gym

from simplepg.simple_utils import include_bias, weighted_sample


def point_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)


def cartpole_get_action(theta, ob, rng=np.random):
    ob_1 = include_bias(ob)
    logits = ob_1.dot(theta.T)
    return weighted_sample(logits, rng=rng)


@click.command()
@click.argument("env_id", type=str, default="Point-v0")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'CartPole-v0':
        env = gym.make('CartPole-v0')
        get_action = cartpole_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    elif env_id == 'Point-v0':
        from simplepg import point_env
        env = gym.make('Point-v0')
        get_action = point_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be one of 'CartPole-v0', 'Point-v0'")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))

    while True:
        ob = env.reset()
        done = False
        # Only render the first trajectory
        # Collect a new trajectory
        rewards = []
        while not done:
            action = get_action(theta, ob, rng=rng)
            next_ob, rew, done, _ = env.step(action)
            ob = next_ob
            env.render()
            rewards.append(rew)

        print("Episode reward: %.2f" % np.sum(rewards))


if __name__ == "__main__":
    main()
