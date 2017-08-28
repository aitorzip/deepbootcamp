import numpy as np
import gym
import gym.spaces
import gym.monitoring
import multiprocessing as mp
import chainer.functions as F
import chainer
import os
import sys
import subprocess
import tblib.pickling_support
import cloudpickle
from collections import defaultdict
import time

import logger
from tqdm import tqdm
from chainer import Variable

tblib.pickling_support.install()


def flatten_dim(space):
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    else:
        assert False


def use_gpu():
    return os.environ.get("USE_GPU", str(False)).lower() == "true"


# Distributions
class Distribution(object):
    def unchain_backward(self):
        """
        Apply unchain_backward() to all its parameters
        """
        raise NotImplementedError

    def as_dict(self):
        """
        Return a dictionary mapping from each distribution parameter name to their value
        """
        raise NotImplementedError

    def sample(self):
        """
        Sample from the current distribution.
        """
        raise NotImplementedError

    def logli(self, a):
        """
        Compute log(p(a))
        """
        raise NotImplementedError

    def kl_div(self, other):
        """
        Compute KL(p_self||p_other)
        """
        raise NotImplementedError

    def likelihood_ratio(self, other, a):
        """
        Compute p_self(a) / p_other(a)
        """
        logli = self.logli(a)
        other_logli = other.logli(a)
        return F.exp(logli - other_logli)


class Categorical(Distribution):
    def __init__(self, logits):
        self.logits = logits

    def as_dict(self):
        return dict(logits=self.logits)

    @classmethod
    def from_dict(cls, d):
        return cls(logits=d["logits"])

    def sample(self):
        return SampleDiscreteLogits()(self.logits)

    def unchain_backward(self):
        self.logits.unchain_backward()

    def logli(self, a):
        all_logli = F.log_softmax(self.logits)
        N = len(a)
        return all_logli[
            np.arange(N),
            a.data.astype(np.int32, copy=False)
        ]

    def entropy(self):
        logli = F.log_softmax(self.logits)
        return F.sum(-logli * F.exp(logli), axis=-1)

    def kl_div(self, other):
        logli = F.log_softmax(self.logits)
        other_logli = F.log_softmax(other.logits)

        # new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * A
        return F.sum(
            F.exp(logli) * (logli - other_logli),
            axis=-1
        )


class SampleDiscreteLogits(chainer.Function):
    """
    Given logits (unnormalized log probabilities for a categorical distribution), sample discrete values proportional to
    their probabilities using the Gumbel-Max trick : http://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    :param logits:
    :return: sample values
    """

    def forward(self, inputs):
        logits = inputs[0]
        xp = chainer.cuda.get_array_module(logits)
        xs = xp.argmax(logits + xp.random.gumbel(size=logits.shape),
                       axis=-1).astype(xp.int32, copy=False)
        return xs,

        # def forward_gpu(self, inputs):
        #     logits = inputs[0]
        #     xs = cupy.argmax(logits + np.random.gumbel(size=logits.shape), axis=-1).astype(np.int32, copy=False)
        #     return xs,


class Gaussian(Distribution):
    def __init__(self, means, log_stds):
        self.means = means
        self.log_stds = log_stds

    def unchain_backward(self):
        self.means.unchain_backward()
        self.log_stds.unchain_backward()

    def as_dict(self):
        return dict(means=self.means, log_stds=self.log_stds)

    @classmethod
    def from_dict(cls, d):
        return cls(means=d["means"], log_stds=d["log_stds"])

    def sample(self):
        xp = chainer.cuda.get_array_module(self.means.data)
        zs = xp.random.normal(size=self.means.data.shape).astype(xp.float32)
        return self.means + chainer.Variable(zs) * F.exp(self.log_stds)
        # return F.gaussian(self.means, self.log_stds * 2)

    def logli(self, a):
        a = F.cast(a, np.float32)
        # transform back to standard normal
        zs = (a - self.means) * F.exp(-self.log_stds)

        # density of standard normal: f(z) = (2*pi*det|Σ|)^(-n/2) * exp(-|x|^2/2)
        # the return value should be log f(z)
        return - F.sum(self.log_stds, axis=-1) - \
            0.5 * F.sum(F.square(zs), axis=-1) - \
            0.5 * self.means.shape[-1] * np.log(2 * np.pi)

    def kl_div(self, other):
        """
        Given the distribution parameters of two diagonal multivariate Gaussians, compute their KL divergence (vectorized)

        Reference: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Kullback.E2.80.93Leibler_divergence_for_multivariate_normal_distributions

        In general, for two n-dimensional distributions, we have

        D_KL(N1||N2) = 1/2 ( tr(Σ_2^{-1}Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) - n + ln(det(Σ_2) / det(Σ_1)) )

        Here, Σ_1 and Σ_2 are diagonal. Hence this equation can be simplified. In terms of the parameters of this method,

            - ln(det(Σ_2) / det(Σ_1)) = sum(2 * (log_stds_2 - log_stds_1), axis=-1)

            - (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) = sum((means_1 - means_2)^2 / vars_2, axis=-1)

            - tr(Σ_2^{-1}Σ_1) = sum(vars_1 / vars_2, axis=-1)

        Where

            - vars_1 = exp(2 * log_stds_1)

            - vars_2 = exp(2 * log_stds_2)

        Combined together, we have

        D_KL(N1||N2) = 1/2 ( tr(Σ_2^{-1}Σ_1) + (μ_2 - μ_1)^T Σ_2^{-1} (μ_2 - μ_1) - n + ln(det(Σ_2) / det(Σ_1)) )
                     = sum(1/2 * ((vars_1 - vars_2) / vars_2 + (means_1 - means_2)^2 / vars_2 + 2 * (log_stds_2 - log_stds_1)), axis=-1)
                     = sum( ((means_1 - means_2)^2 + vars_1 - vars_2) / (2 * vars_2) + (log_stds_2 - log_stds_1)), axis=-1)

        :param means_1: List of mean parameters of the first distribution
        :param log_stds_1: List of log standard deviation parameters of the first distribution
        :param means_2: List of mean parameters of the second distribution
        :param log_stds_2: List of log standard deviation parameters of the second distribution
        :return: An array of KL divergences.
        """

        vars = F.exp(2 * self.log_stds)
        other_vars = F.exp(2 * other.log_stds)

        return F.sum((F.square(self.means - other.means) + vars - other_vars) /
                     (2 * other_vars + 1e-8) + other.log_stds - self.log_stds, axis=-1)

    def entropy(self):
        return F.sum(self.log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)


def env_worker(env_maker, conn, n_worker_envs):
    envs = []
    for _ in range(n_worker_envs):
        envs.append(env_maker.make())
    while True:
        command, data = conn.recv()
        try:
            if command == 'reset':
                obs = []
                for env in envs:
                    obs.append(env.reset())
                conn.send(('success', obs))
            elif command == 'seed':
                seeds = data
                for env, seed in zip(envs, seeds):
                    env.seed(seed)
                conn.send(('success', None))
            elif command == 'step':
                actions = data
                results = []
                for env, action in zip(envs, actions):
                    next_ob, rew, done, info = env.step(action)
                    if done:
                        info["last_observation"] = next_ob
                        next_ob = env.reset()
                    results.append((next_ob, rew, done, info))
                conn.send(('success', results))
            elif command == 'close':
                for env in envs:
                    env.close()
                conn.send(('success', None))
                return
            else:
                raise ValueError("Unrecognized command: {}".format(command))
        except Exception as e:
            conn.send(('error', sys.exc_info()))


class EnvPool(object):
    """
    Using a pool of workers to run multiple environments in parallel. This implementation supports multiple environments
    per worker to be as flexible as possible.
    """

    def __init__(self, env_maker, n_envs=mp.cpu_count(), n_parallel=mp.cpu_count()):
        self.env_maker = env_maker
        self.n_envs = n_envs
        # No point in having more parallel workers than environments
        if n_parallel > n_envs:
            n_parallel = n_envs
        self.n_parallel = n_parallel
        self.workers = []
        self.conns = []
        # try to split evenly, but this isn't always possible
        self.n_worker_envs = [len(d) for d in np.array_split(
            np.arange(self.n_envs), self.n_parallel)]
        self.worker_env_offsets = np.concatenate(
            [[0], np.cumsum(self.n_worker_envs)[:-1]])
        self.last_obs = None

    def start(self):
        workers = []
        conns = []
        for idx in range(self.n_parallel):
            worker_conn, master_conn = mp.Pipe()
            worker = mp.Process(target=env_worker, args=(
                self.env_maker, worker_conn, self.n_worker_envs[idx]))
            worker.start()
            # pin each worker to a single core
            if sys.platform == 'linux':
                subprocess.check_call(
                    ["taskset", "-p", "-c",
                        str(idx % mp.cpu_count()), str(worker.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            workers.append(worker)
            conns.append(master_conn)

        self.workers = workers
        self.conns = conns

        # set initial seeds
        seeds = np.random.randint(
            low=0, high=np.iinfo(np.int32).max, size=self.n_envs)
        self.seed([int(x) for x in seeds])

    def __enter__(self):
        self.start()
        return self

    def reset(self):
        for conn in self.conns:
            conn.send(('reset', None))
        obs = []
        for conn in self.conns:
            status, data = conn.recv()
            if status == 'success':
                obs.extend(data)
            else:
                raise data[1].with_traceback(data[2])
        assert len(obs) == self.n_envs
        self.last_obs = obs
        return obs

    def step(self, actions):
        assert len(actions) == self.n_envs
        for idx, conn in enumerate(self.conns):
            offset = self.worker_env_offsets[idx]
            conn.send(
                ('step', actions[offset:offset + self.n_worker_envs[idx]]))

        results = []

        for conn in self.conns:
            status, data = conn.recv()
            if status == 'success':
                results.extend(data)
            else:
                raise data[1].with_traceback(data[2])
        next_obs, rews, dones, infos = list(map(list, zip(*results)))
        self.last_obs = next_obs
        return next_obs, rews, dones, infos

    def seed(self, seeds):
        assert len(seeds) == self.n_envs
        for idx, conn in enumerate(self.conns):
            offset = self.worker_env_offsets[idx]
            conn.send(('seed', seeds[offset:offset + self.n_worker_envs[idx]]))
        for conn in self.conns:
            status, data = conn.recv()
            if status != 'success':
                raise data[1].with_traceback(data[2])

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        for conn in self.conns:
            conn.send(('close', None))
        for conn in self.conns:
            status, data = conn.recv()
            if status != 'success':
                raise data[1].with_traceback(data[2])
        for worker in self.workers:
            worker.join()
        self.workers = []
        self.conns = []


def parallel_collect_samples(env_pool, policy, num_samples):
    """
    Collect trajectories in parallel using a pool of workers. Actions are computed using the provided policy.
    Collection will continue until at least num_samples trajectories are collected. It will exceed this amount by
    at most env_pool.n_envs. This means that some of the trajectories will not be executed until termination. These
    partial trajectories will have their "finished" entry set to False.

    When starting, it will first check if env_pool.last_obs is set, and if so, it will start from there rather than
    resetting all environments. This is useful for reusing the same episode.

    :param env_pool: An instance of EnvPool.
    :param policy: The policy used to select actions.
    :param num_samples: The minimum number of samples to collect.
    :return:
    """
    trajs = []
    partial_trajs = [None] * env_pool.n_envs
    num_collected = 0

    if env_pool.last_obs is not None:
        obs = env_pool.last_obs
    else:
        obs = env_pool.reset()

    if logger.get_level() <= logger.INFO:
        progbar = tqdm(total=num_samples)
    else:
        progbar = None

    while num_collected < num_samples:
        actions, dists = policy.get_actions(obs)
        next_obs, rews, dones, infos = env_pool.step(actions)
        for idx in range(env_pool.n_envs):
            if partial_trajs[idx] is None:
                partial_trajs[idx] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    distributions=[],
                )
            traj = partial_trajs[idx]
            traj["observations"].append(obs[idx])
            traj["actions"].append(actions[idx])
            traj["rewards"].append(rews[idx])
            traj_dists = traj["distributions"]
            traj_dists.append({k: v[idx] for k, v in dists.items()})
            if dones[idx]:
                trajs.append(
                    dict(
                        observations=np.asarray(traj["observations"]),
                        actions=np.asarray(traj["actions"]),
                        rewards=np.asarray(traj["rewards"]),
                        distributions={
                            k: np.asarray([d[k] for d in traj_dists])
                            for k in traj_dists[0].keys()
                        },
                        last_observation=infos[idx]["last_observation"],
                        finished=True,
                    )
                )
                partial_trajs[idx] = None
        obs = next_obs
        num_collected += env_pool.n_envs
        if progbar is not None:
            progbar.update(env_pool.n_envs)

    if progbar is not None:
        progbar.close()

    for idx in range(env_pool.n_envs):
        if partial_trajs[idx] is not None:
            traj = partial_trajs[idx]
            traj_dists = traj["distributions"]
            trajs.append(
                dict(
                    observations=np.asarray(traj["observations"]),
                    actions=np.asarray(traj["actions"]),
                    rewards=np.asarray(traj["rewards"]),
                    distributions={
                        k: np.asarray([d[k] for d in traj_dists])
                        for k in traj_dists[0].keys()
                    },
                    last_observation=obs[idx],
                    finished=False,
                )
            )

    return trajs


def ordered_params(chain):
    namedparams = sorted(chain.namedparams(), key=lambda x: x[0])
    return [x[1] for x in namedparams]


def get_flat_params(chain):
    xp = chain.xp
    params = ordered_params(chain)
    if len(params) > 0:
        return xp.concatenate([xp.ravel(param.data) for param in params])
    else:
        return xp.zeros((0,), dtype=xp.float32)


def get_flat_grad(chain):
    xp = chain.xp
    params = ordered_params(chain)
    if len(params) > 0:
        return xp.concatenate([xp.ravel(param.grad) for param in params])
    else:
        return xp.zeros((0,), dtype=xp.float32)


def set_flat_params(chain, flat_params):
    offset = 0
    for param in ordered_params(chain):
        param.data[:] = flat_params[offset:offset +
                                    param.data.size].reshape(param.data.shape)
        offset += param.data.size


def set_flat_grad(chain, flat_grad):
    offset = 0
    for param in ordered_params(chain):
        param.grad[:] = flat_grad[offset:offset +
                                  param.grad.size].reshape(param.grad.shape)
        offset += param.grad.size


def cg(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    """
    Demmel p 312. Approximately solve x = A^{-1}b, or Ax = b, where we only have access to f: x -> Ax
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    return x


def explained_variance_1d(ypred, y):
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    if np.isclose(vary, 0):
        if np.var(ypred) > 1e-8:
            return 0
        else:
            return 1
    return 1 - np.var(y - ypred) / (vary + 1e-8)


class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape, dtype=np.float32)
        self._S = np.zeros(shape, dtype=np.float32)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class UniqueChainList(chainer.ChainList):
    def params(self, include_uninit=True):
        return set(super().params(include_uninit=include_uninit))


class SnapshotSaver(object):
    def __init__(self, dir, interval=1, latest_only=None):
        self.dir = dir
        self.interval = interval
        if latest_only is None:
            latest_only = True
            snapshots_folder = os.path.join(dir, "snapshots")
            if os.path.exists(snapshots_folder):
                if os.path.exists(os.path.join(snapshots_folder, "latest.pkl")):
                    latest_only = True
                elif len(os.listdir(snapshots_folder)) > 0:
                    latest_only = False
        self.latest_only = latest_only

    @property
    def snapshots_folder(self):
        return os.path.join(self.dir, "snapshots")

    def get_snapshot_path(self, index):
        if self.latest_only:
            return os.path.join(self.snapshots_folder, "latest.pkl")
        else:
            return os.path.join(self.snapshots_folder, "%d.pkl" % index)

    def save_state(self, index, state):
        if index % self.interval == 0:
            file_path = self.get_snapshot_path(index)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                cloudpickle.dump(state, f, protocol=-1)

    def get_state(self):
        if self.latest_only:
            try:
                with open(self.get_snapshot_path(0), "rb") as f:
                    return cloudpickle.load(f)
            except EOFError:
                pass
        else:
            snapshot_files = os.listdir(self.snapshots_folder)
            snapshot_files = sorted(
                snapshot_files, key=lambda x: int(x.split(".")[0]))[::-1]
            for file in snapshot_files:
                file_path = os.path.join(self.snapshots_folder, file)
                try:
                    with open(file_path, "rb") as f:
                        return cloudpickle.load(f)
                except EOFError:
                    pass
