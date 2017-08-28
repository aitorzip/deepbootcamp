#!/usr/bin/env python
from collections import deque

import time
import chainer as C
import chainer.functions as F
import numpy as np
import pickle
import click
import gym

from simpledqn.replay_buffer import ReplayBuffer
import logger
from simpledqn.wrappers import NoopResetEnv, EpisodicLifeEnv

nprs = np.random.RandomState


def assert_allclose(a, b):
    if isinstance(a, (np.ndarray, float, int)):
        np.testing.assert_allclose(a, b)
    elif isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list))
        assert len(a) == len(b)
        for a_i, b_i in zip(a, b):
            assert_allclose(a_i, b_i)
    elif isinstance(a, C.Variable):
        assert isinstance(b, C.Variable)
        assert_allclose(a.data, b.data)
    else:
        raise NotImplementedError


rng = nprs(42)


# ---------------------

class Adam(object):
    def __init__(self, shape, stepsize, beta1=0.9, beta2=0.999, epsilon=1e-08):
        self.stepsize, self.beta1, self.beta2, self.epsilon = stepsize, beta1, beta2, epsilon
        self.t = 0
        self.v = np.zeros(shape, dtype=np.float32)
        self.m = np.zeros(shape, dtype=np.float32)

    def step(self, g):
        self.t += 1
        a = self.stepsize * \
            np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        self.v = self.beta2 * self.v + (1 - self.beta2) * (g * g)
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        step = - a * self.m / (np.sqrt(self.v) + self.epsilon)
        return step


# ---------------------

class NN(object):
    """Simple transparent neural network (multilayer perceptron) model.
    """

    def __init__(self, dims=None, out_fn=None):
        assert dims is not None
        assert out_fn is not None
        assert len(dims) >= 2

        self._out_fn = out_fn
        self.lst_w, self.lst_b = [], []
        for i in range(len(dims) - 1):
            shp = dims[i + 1], dims[i]
            # Correctly init weights.
            std = 0.01 if i == len(dims) - 2 else 1.0
            out = rng.randn(*shp).astype(np.float32)
            out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
            self.lst_w.append(C.Variable(out))
            self.lst_b.append(C.Variable(np.zeros(shp[0], dtype=np.float32)))
        self.train_vars = self.lst_w + self.lst_b

    def set_params(self, params):
        lst_wt, lst_bt = params
        for w, wt in zip(self.lst_w, lst_wt):
            w.data[...] = wt.data
        for b, bt in zip(self.lst_b, lst_bt):
            b.data[...] = bt.data

    def get_params(self):
        return self.lst_w, self.lst_b

    def dump(self, file_path=None):
        file = open(file_path, 'wb')
        pickle.dump(dict(w=self.lst_w, b=self.lst_b), file, -1)
        file.close()

    def load(self, file_path=None):
        file = open(file_path, 'rb')
        params = pickle.load(file)
        file.close()
        return params['w'], params['b']

    def forward(self, x):
        for i, (w, b) in enumerate(zip(self.lst_w, self.lst_b)):
            x = F.linear(x, w, b)
            if i != len(self.lst_w) - 1:
                x = F.tanh(x)
            else:
                return self._out_fn(x)


# ---------------------

def preprocess_obs_gridworld(obs):
    return obs.astype(np.float32)


def preprocess_obs_ram(obs):
    return obs.astype(np.float32) / 255.


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(1.0, float(t) / self.schedule_timesteps)
        return self.initial_p + (self.final_p - self.initial_p) * fraction


class DQN(object):
    def __init__(self, env, get_obs_dim, get_act_dim, obs_preprocessor, replay_buffer, q_dim_hid,
                 opt_batch_size, discount, initial_step, max_steps, learning_start_itr, target_q_update_freq,
                 train_q_freq,
                 log_freq, double_q, final_eps, initial_eps, fraction_eps, render):
        self._env = env
        self._get_obs_dim = get_obs_dim
        self._get_act_dim = get_act_dim
        self._obs_preprocessor = obs_preprocessor
        self._replay_buffer = replay_buffer
        self._initial_step = initial_step
        self._max_steps = max_steps
        self._target_q_update_freq = target_q_update_freq
        self._learning_start_itr = learning_start_itr
        self._train_q_freq = train_q_freq
        self._log_freq = log_freq
        self._double_q = double_q
        self._act_dim = env.action_space.n
        self._opt_batch_size = opt_batch_size
        self._discount = discount
        self._render = render
        nn_args = dict(
            dims=[self._get_obs_dim(env)] + q_dim_hid +
            [self._get_act_dim(env)],
            out_fn=lambda x: x)
        # Q-function, Q(s,a,\theta)
        self._q = NN(**nn_args)
        # Target Q-function, Q(s,a,\theta')
        self._qt = NN(**nn_args)
        self.lst_adam = [Adam(var.shape, stepsize=1e-4)
                         for var in self._q.train_vars]
        self.exploration = LinearSchedule(
            schedule_timesteps=int(fraction_eps * max_steps),
            initial_p=initial_eps,
            final_p=final_eps)

    def eps_greedy(self, obs, epsilon):
        # Check Q function, do argmax.
        rnd = rng.rand()
        if rnd > epsilon:
            obs = self._obs_preprocessor(obs)
            q_values = self._q.forward(obs)
            return F.argmax(q_values, axis=1).data[0]
        else:
            return rng.randint(0, self._act_dim)

    def compute_q_learning_loss(self, l_obs, l_act, l_rew, l_next_obs, l_done):
        """
        :param l_obs: A chainer variable holding a list of observations. Should be of shape N * |S|.
        :param l_act: A chainer variable holding a list of actions. Should be of shape N.
        :param l_rew: A chainer variable holding a list of rewards. Should be of shape N.
        :param l_next_obs: A chainer variable holding a list of observations at the next time step. Should be of
        shape N * |S|.
        :param l_done: A chainer variable holding a list of binary values (indicating whether episode ended after this
        time step). Should be of shape N.
        :return: A chainer variable holding a scalar loss.
        """
        # Hint: You may want to make use of the following fields: self._discount, self._q, self._qt
        # Hint2: Q-function can be called by self._q.forward(argument)
        # Hint3: You might also find https://docs.chainer.org/en/stable/reference/generated/chainer.functions.select_item.html useful
        loss = C.Variable(np.array([0.]))  # TODO: replace this line
        "*** YOUR CODE HERE ***"
        return loss

    def compute_double_q_learning_loss(self, l_obs, l_act, l_rew, l_next_obs, l_done):
        """
        :param l_obs: A chainer variable holding a list of observations. Should be of shape N * |S|.
        :param l_act: A chainer variable holding a list of actions. Should be of shape N.
        :param l_rew: A chainer variable holding a list of rewards. Should be of shape N.
        :param l_next_obs: A chainer variable holding a list of observations at the next time step. Should be of
        shape N * |S|.
        :param l_done: A chainer variable holding a list of binary values (indicating whether episode ended after this
        time step). Should be of shape N.
        :return: A chainer variable holding a scalar loss.
        """
        # Hint: You may want to make use of the following fields: self._discount, self._q, self._qt
        # Hint2: Q-function can be called by self._q.forward(argument)
        # Hint3: You might also find https://docs.chainer.org/en/stable/reference/generated/chainer.functions.select_item.html useful
        loss = C.Variable(np.array([0.]))  # TODO: replace this line
        "*** YOUR CODE HERE ***"
        return loss

    def train_q(self, l_obs, l_act, l_rew, l_next_obs, l_done):
        """Update Q-value function by sampling from the replay buffer."""

        l_obs = self._obs_preprocessor(l_obs)
        l_next_obs = self._obs_preprocessor(l_next_obs)
        if self._double_q:
            loss = self.compute_double_q_learning_loss(
                l_obs, l_act, l_rew, l_next_obs, l_done)
        else:
            loss = self.compute_q_learning_loss(
                l_obs, l_act, l_rew, l_next_obs, l_done)
        for var in self._q.train_vars:
            var.cleargrad()
        loss.backward()
        for var, adam in zip(self._q.train_vars, self.lst_adam):
            var.data += adam.step(var.grad)
        return loss.data

    def _update_target_q(self):
        """Update the target Q-value function by copying the current Q-value function weights."""
        q_params = self._q.get_params()
        self._qt.set_params(q_params)

    def train(self):
        obs = self._env.reset()

        episode_rewards = []
        n_episodes = 0
        l_episode_return = deque([], maxlen=10)
        l_discounted_episode_return = deque([], maxlen=10)
        l_tq_squared_error = deque(maxlen=50)
        log_itr = -1
        for itr in range(self._initial_step, self._max_steps):
            act = self.eps_greedy(obs[np.newaxis, :],
                                  self.exploration.value(itr))
            next_obs, rew, done, _ = self._env.step(act)
            if self._render:
                self._env.render()
            self._replay_buffer.add(obs, act, rew, next_obs, float(done))

            episode_rewards.append(rew)

            if done:
                obs = self._env.reset()
                episode_return = np.sum(episode_rewards)
                discounted_episode_return = np.sum(
                    episode_rewards * self._discount ** np.arange(len(episode_rewards)))
                l_episode_return.append(episode_return)
                l_discounted_episode_return.append(discounted_episode_return)
                episode_rewards = []
                n_episodes += 1
            else:
                obs = next_obs

            if itr % self._target_q_update_freq == 0 and itr > self._learning_start_itr:
                self._update_target_q()

            if itr % self._train_q_freq == 0 and itr > self._learning_start_itr:
                # Sample from replay buffer.
                l_obs, l_act, l_rew, l_obs_prime, l_done = self._replay_buffer.sample(
                    self._opt_batch_size)
                # Train Q value function with sampled data.
                td_squared_error = self.train_q(
                    l_obs, l_act, l_rew, l_obs_prime, l_done)
                l_tq_squared_error.append(td_squared_error)

            if (itr + 1) % self._log_freq == 0 and len(l_episode_return) > 5:
                log_itr += 1
                logger.logkv('Iteration', log_itr)
                logger.logkv('Steps', itr)
                logger.logkv('Epsilon', self.exploration.value(itr))
                logger.logkv('Episodes', n_episodes)
                logger.logkv('AverageReturn', np.mean(l_episode_return))
                logger.logkv('AverageDiscountedReturn',
                             np.mean(l_discounted_episode_return))
                logger.logkv('TDError^2', np.mean(l_tq_squared_error))
                logger.dumpkvs()
                self._q.dump(logger.get_dir() + '/weights.pkl')

    def test(self, epsilon):
        try:
            self._q.set_params(self._q.load(logger.get_dir() + '/weights.pkl'))
        except Exception as e:
            print(e)
        obs = self._env.reset()
        while True:
            act = self.eps_greedy(obs[np.newaxis, :], epsilon)
            obs_prime, rew, done, _ = self._env.step(act)
            self._env.render()
            if done:
                obs = self._env.reset()
                print('Done!')
                time.sleep(1)
            else:
                obs = obs_prime


@click.command()
@click.argument("env_id", type=str, default="GridWorld-v0")
@click.option("--double", type=bool, default=False)
@click.option("--render", type=bool, default=False)
def main(env_id, double, render):
    if env_id == 'GridWorld-v0':
        from simpledqn import gridworld_env
        env = gym.make('GridWorld-v0')

        def get_obs_dim(x): return x.observation_space.n

        def get_act_dim(x): return x.action_space.n
        obs_preprocessor = preprocess_obs_gridworld
        max_steps = 100000
        log_freq = 1000
        target_q_update_freq = 100
        initial_step = 0
        log_dir = "data/local/dqn_gridworld"
    elif env_id == 'Pong-ram-v0':
        env = EpisodicLifeEnv(NoopResetEnv(gym.make('Pong-ram-v0')))

        def get_obs_dim(x): return x.observation_space.shape[0]

        def get_act_dim(x): return x.action_space.n
        obs_preprocessor = preprocess_obs_ram
        max_steps = 10000000
        log_freq = 10000
        target_q_update_freq = 1000
        initial_step = 1000000
        log_dir = "data/local/dqn_pong"
    else:
        raise ValueError(
            "Unsupported environment: must be one of 'GridWorld-v0' 'Pong-ram-v0'")

    logger.session(log_dir).__enter__()
    env.seed(42)

    # Initialize the replay buffer that we will use.
    replay_buffer = ReplayBuffer(max_size=10000)

    # Initialize DQN training procedure.
    dqn = DQN(
        env=env,
        get_obs_dim=get_obs_dim,
        get_act_dim=get_act_dim,
        obs_preprocessor=obs_preprocessor,
        replay_buffer=replay_buffer,

        # Q-value parameters
        q_dim_hid=[256, 256] if env_id == 'Pong-ram-v0' else [],
        opt_batch_size=64,

        # DQN gamma parameter
        discount=0.99,

        # Training procedure length
        initial_step=initial_step,
        max_steps=max_steps,
        learning_start_itr=max_steps // 100,
        # Frequency of copying the actual Q to the target Q
        target_q_update_freq=target_q_update_freq,
        # Frequency of updating the Q-value function
        train_q_freq=4,
        # Double Q
        double_q=double,

        # Exploration parameters
        initial_eps=1.0,
        final_eps=0.05,
        fraction_eps=0.1,

        # Logging
        log_freq=log_freq,
        render=render,
    )

    if env_id == 'Pong-ram-v0':
        # Warm start Q-function
        dqn._q.set_params(dqn._q.load('simpledqn/weights_warm_start.pkl'))
        dqn._qt.set_params(dqn._qt.load('simpledqn/weights_warm_start.pkl'))
        # Warm start replay buffer
        dqn._replay_buffer.load('simpledqn/replay_buffer_warm_start.pkl')
        print("Warm-starting Pong training!")

    if env_id == 'GridWorld-v0':
        # Run tests on GridWorld-v0
        test_args = dict(
            l_obs=nprs(0).rand(64, 16).astype(np.float32),
            l_act=nprs(1).randint(0, 3, size=(64,)),
            l_rew=nprs(2).randint(0, 3, size=(64,)).astype(np.float32),
            l_next_obs=nprs(3).rand(64, 16).astype(np.float32),
            l_done=nprs(4).randint(0, 2, size=(64,)).astype(np.float32),
        )
        if not double:
            tgt = np.array([1.909377098083496], dtype=np.float32)
            actual_var = dqn.compute_q_learning_loss(**test_args)
            test_name = "compute_q_learning_loss"
            assert isinstance(
                actual_var, C.Variable), "%s should return a Chainer variable" % test_name
            actual = actual_var.data
            try:
                assert_allclose(tgt, actual)
                print("Test for %s passed!" % test_name)
            except AssertionError as e:
                print("Warning: test for %s didn't pass!" % test_name)
                print(e)
                input(
                    "** Test failed. Press Ctrl+C to exit or press enter to continue training anyways")
        else:
            tgt = np.array([1.9066928625106812], dtype=np.float32)
            actual_var = dqn.compute_double_q_learning_loss(**test_args)
            test_name = "compute_double_q_learning_loss"
            assert isinstance(
                actual_var, C.Variable), "%s should return a Chainer variable" % test_name
            actual = actual_var.data
            try:
                assert_allclose(tgt, actual)
                print("Test for %s passed!" % test_name)
            except AssertionError as e:
                print("Warning: test for %s didn't pass!" % test_name)
                print(e)
                input(
                    "** Test failed. Press Ctrl+C to exit or press enter to continue training anyways")

    if render:
        dqn.test(epsilon=0.0)
    else:
        # Train the agent!
        dqn.train()

    # Close gym environment.
    env.close()


if __name__ == "__main__":
    main()
