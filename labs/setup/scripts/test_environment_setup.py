#!/usr/bin/env python


def main():
    import roboschool
    import gym
    import chainer
    env = gym.make('CartPole-v0')
    env.reset()
    env.step(env.action_space.sample())
    env = gym.make('RoboschoolHalfCheetah-v1')
    env.reset()
    env.step(env.action_space.sample())
    print("Your environment has been successfully set up!")


if __name__ == "__main__":
    main()
