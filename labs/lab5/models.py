from collections import OrderedDict

from utils import *
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import scipy.optimize
import gym.spaces


class Model(chainer.Chain):
    def __init__(self, observation_space, action_space, env_spec, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.env_spec = env_spec

        self.obs_dim = flatten_dim(observation_space)
        self.action_dim = flatten_dim(action_space)


class NNFeatureModel(Model):
    feature_dim = None

    def compute_features(self, obs):
        raise NotImplementedError

    def feature_links(self):
        raise NotImplementedError


class MLPFeatureModel(NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, *, hidden_sizes=(128, 64), hidden_nonlinearity=F.relu,
                 **kwargs):
        super().__init__(observation_space, action_space, env_spec, **kwargs)
        self.hidden_sizes = hidden_sizes
        if isinstance(hidden_nonlinearity, str):
            if hidden_nonlinearity == 'relu':
                hidden_nonlinearity = F.relu
            elif hidden_nonlinearity == 'tanh':
                hidden_nonlinearity = F.tanh
            elif hidden_nonlinearity == 'elu':
                hidden_nonlinearity = F.elu
            else:
                raise NotImplementedError
        self.hidden_nonlinearity = hidden_nonlinearity
        self.n_layers = len(hidden_sizes)
        self._feature_links = OrderedDict()
        with self.init_scope():
            input_size = self.obs_dim
            for idx, hidden_size in enumerate(hidden_sizes):
                link = L.Linear(input_size, hidden_size)
                name = "fc{}".format(idx + 1)
                setattr(self, name, link)
                self._feature_links[name] = link
                input_size = hidden_size
            self.feature_dim = input_size

    def feature_links(self):
        return self._feature_links

    def compute_features(self, obs):
        obs = F.cast(obs, np.float32)
        h = obs
        for link in self.feature_links().values():
            h = self.hidden_nonlinearity(link(h))
        return h


class CNNFeatureModel(NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, **kwargs):
        super().__init__(observation_space, action_space, env_spec, **kwargs)

        with self.init_scope():
            in_channels = observation_space.shape[-1]
            self.conv1 = L.Convolution2D(in_channels, 16, 8, stride=4)
            self.conv2 = L.Convolution2D(16, 32, 4, stride=2)
            self.fc = L.Linear(2592, 256)
            self.feature_dim = 256

    def feature_links(self):
        return dict(conv1=self.conv1, conv2=self.conv2, fc=self.fc)

    def compute_features(self, obs):
        obs = F.cast(obs, np.float32)
        obs = F.transpose(obs, (0, 3, 1, 2))
        h1 = F.relu(self.conv1(obs))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.fc(h2))
        return h3


class WeightSharingFeatureModel(NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, feature_model, **kwargs):
        super().__init__(observation_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            for name, link in feature_model.feature_links().items():
                setattr(self, name, link)
        self._compute_features = feature_model.compute_features
        self.feature_dim = feature_model.feature_dim
        self.feature_model = feature_model
        self._feature_links = feature_model.feature_links

    def compute_features(self, obs):
        return self._compute_features(obs)

    def feature_links(self):
        return self._feature_links()


# ==============================
# Value Functions
# ==============================

class ValueFunction(Model):
    def compute_state_values(self, obs):
        raise NotImplementedError


class NNFeatureValueFunction(ValueFunction, NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, **kwargs):
        super().__init__(observation_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            self.l_vf = L.Linear(self.feature_dim, 1)

    def compute_state_values(self, obs, feats=None):
        if feats is None:
            feats = super().compute_features(obs)
        return self.l_vf(feats)[..., 0]


class ZeroValueFunction(ValueFunction):
    def compute_state_values(self, obs):
        obs_data = obs.data
        xp = chainer.cuda.get_array_module(obs_data)
        return chainer.Variable(xp.zeros(len(obs_data), dtype=xp.float32))

    pass


class WeightSharingValueFunction(NNFeatureValueFunction, WeightSharingFeatureModel):
    pass


class MLPValueFunction(NNFeatureValueFunction, MLPFeatureModel):
    pass


class CNNValueFunction(NNFeatureValueFunction, CNNFeatureModel):
    pass


# ==============================
# Policies
# ==============================

class Policy(Model):
    feature_dim = None

    def compute_dists(self, obs):
        """
        Given some observations, compute the parameters for the action distributions.
        :param obs: A chainer variable containing a list of observations.
        :return: An instance of the Distribution class, represeting the action distribution.
        """
        raise NotImplementedError

    def get_actions(self, obs):
        with chainer.no_backprop_mode():
            dists = self.compute_dists(Variable(np.asarray(obs)))
            actions = dists.sample()
            return actions.data, {k: v.data for k, v in dists.as_dict().items()}

    def get_action(self, ob):
        actions, dists = self.get_actions(np.expand_dims(ob, 0))
        return actions[0], {k: v[0] for k, v in dists.items()}


class NormalizingPolicy(Policy):
    """
    This policy wraps an existing policy. It mains a running average of past observations,
    to roughly ensure that the input dimensions are
    whitened.
    Most useful for relatively low-dimensional observations (i.e. not images)
    """

    def __init__(self, policy, clip=None):
        super().__init__(policy.observation_space, policy.action_space, policy.env_spec)
        with self.init_scope():
            self.policy = policy
        self.running_stat = RunningStat(policy.observation_space.shape)
        self.clip = float(clip)

    def get_actions(self, obs):
        obs = np.asarray(obs)
        for ob in obs:
            self.running_stat.push(ob)
        obs = (obs - self.running_stat.mean[None, ...]) / \
            (self.running_stat.std[None, ...] + 1e-8)
        if self.clip is not None:
            obs = np.clip(obs, -self.clip, self.clip)
        return self.policy.get_actions(obs)

    def compute_dists(self, obs):
        mean_var = Variable(self.running_stat.mean.astype(np.float32))
        std_var = Variable(self.running_stat.std.astype(np.float32))

        obs = obs - F.broadcast_to(mean_var, obs.shape)
        obs = obs / (F.broadcast_to(std_var, obs.shape) + 1e-8)

        if self.clip is not None:
            obs = F.clip(obs, -self.clip, self.clip)

        return self.policy.compute_dists(obs)

    @property
    def distribution(self):
        return self.policy.distribution


class NNFeaturePolicy(Policy, NNFeatureModel):
    def create_vf(self):
        return WeightSharingValueFunction(
            observation_space=self.observation_space,
            action_space=self.action_space,
            env_spec=self.env_spec,
            feature_model=self,
        )


class CategoricalPolicy(NNFeaturePolicy):
    def __init__(self, observation_space, action_space, env_spec, **kwargs):
        super().__init__(observation_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            assert self.feature_dim is not None
            self.l_act = L.Linear(self.feature_dim, self.action_dim)

    def compute_dists(self, obs, feats=None):
        if feats is None:
            feats = super().compute_features(obs)
        logits = self.l_act(feats)
        return Categorical(logits=logits)

    @property
    def distribution(self):
        return Categorical


class GaussianPolicy(NNFeaturePolicy):
    def __init__(self, observation_space, action_space, env_spec, **kwargs):
        super().__init__(observation_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            assert self.feature_dim is not None
            self.l_act = L.Linear(self.feature_dim, self.action_dim)
            self.log_std = chainer.Parameter(
                shape=(self.action_dim,), initializer=chainer.initializers.Zero())

    def compute_dists(self, obs, feats=None):
        if feats is None:
            feats = super().compute_features(obs)
        means = self.l_act(feats)
        # for this policy, the variance is independent of the state
        log_stds = F.tile(self.log_std.reshape((1, -1)), (len(feats), 1))
        return Gaussian(means=means, log_stds=log_stds)

    @property
    def distribution(self):
        return Gaussian


class CategoricalMLPPolicy(CategoricalPolicy, MLPFeatureModel):
    pass


class GaussianMLPPolicy(GaussianPolicy, MLPFeatureModel):
    pass


class CategoricalCNNPolicy(CategoricalPolicy, CNNFeatureModel):
    pass


# ==============================
# Baselines
# ==============================


class Baseline(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, obs):
        raise NotImplementedError

    def update(self, trajs):
        raise NotImplementedError


class ZeroBaseline(Baseline):
    def predict(self, obs):
        return np.zeros(len(obs))

    def update(self, trajs):
        pass


class TimeDependentBaseline(Baseline):
    def __init__(self, observation_space, action_space, env_spec):
        self.observation_space = observation_space
        self.action_space = action_space
        self.env_spec = env_spec
        # 1D array: b[t] = moving average of past returns R_t
        self.b = np.zeros(env_spec.timestep_limit + 1)
        # # 1D array: ns[t] = number of past trajectories with at least t time steps
        # self.ns = np.zeros(env_spec.timestep_limit)

    def predict(self, obs):
        T = obs.shape[0]
        return self.b[:T]

    def update(self, trajs):
        all_returns = [traj['returns'] for traj in trajs]
        max_T = self.env_spec.timestep_limit + 1
        padded_returns = [
            np.pad(returns, (0, max_T - len(returns)),
                   mode='constant', constant_values=(np.nan,))
            for returns in all_returns
        ]
        self.b[:] = np.nanmean(padded_returns, axis=0)
        self.b[np.isnan(self.b)] = 0


class LinearFeatureBaseline(Baseline):
    def __init__(self, observation_space, action_space, env_spec):
        self.observation_space = observation_space
        self.action_space = action_space
        self.env_spec = env_spec
        self.coeffs = None

    def get_features(self, obs):
        o = np.clip(obs, -10, 10)
        l = len(obs)
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    def update(self, trajs):
        featmat = np.concatenate(
            [self.get_features(traj["observations"]) for traj in trajs])
        returns = np.concatenate([traj["returns"] for traj in trajs])
        self.coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat), featmat.T.dot(returns))[0]

    def predict(self, obs):
        if self.coeffs is None:
            return np.zeros(len(obs))
        return self.get_features(obs).dot(self.coeffs)


class NNFeatureBaseline(Baseline, NNFeatureModel):
    def __init__(self, observation_space, action_space, env_spec, mixture_fraction=0.1, **kwargs):
        # For input, we will concatenate the observation with the time, so we need to increment the observation
        # dimension
        if isinstance(observation_space, gym.spaces.Box) and len(observation_space.shape) == 1:
            self.concat_time = True
            obs_space = gym.spaces.Box(
                low=np.append(observation_space.low, 0),
                high=np.append(observation_space.high, 2 ** 32),
            )
        else:
            obs_space = observation_space
        self.mixture_fraction = mixture_fraction
        super().__init__(obs_space, action_space, env_spec, **kwargs)
        with self.init_scope():
            assert self.feature_dim is not None
            self.l_b = L.Linear(self.feature_dim, self.action_dim)

    def compute_baselines(self, obs):
        feats = self.compute_features(obs)
        return self.l_b(feats)[..., 0]

    # Value functions can themselves be used as baselines
    def predict(self, obs):
        with chainer.no_backprop_mode():
            if self.concat_time:
                ts = np.arange(len(obs)) / self.env_spec.timestep_limit
                obs = np.concatenate([obs, ts[:, None]], axis=-1)
            values = self.compute_baselines(Variable(obs))
            return values.data

    # By default, when used as baselines, value functions are updated via L-BFGS
    def update(self, trajs):
        obs = np.concatenate([traj['observations'] for traj in trajs], axis=0)
        if self.concat_time:
            ts = np.concatenate([np.arange(len(traj['observations'])) / self.env_spec.timestep_limit for traj in trajs],
                                axis=0)
            obs = np.concatenate([obs, ts[:, None]], axis=-1)
        returns = np.concatenate([traj['returns'] for traj in trajs], axis=0)
        baselines = np.concatenate([traj['baselines']
                                    for traj in trajs], axis=0)

        # regress to a mixture of current and past predictions
        targets = returns * (1. - self.mixture_fraction) + \
            baselines * self.mixture_fraction

        # use lbfgs to perform the update
        cur_params = get_flat_params(self)

        obs = Variable(obs)
        targets = Variable(targets.astype(np.float32))

        def f_loss_grad(x):
            set_flat_params(self, x)
            self.cleargrads()
            values = self.compute_baselines(obs)
            loss = F.mean(F.square(values - targets))
            loss.backward()
            flat_grad = get_flat_grad(self)
            return loss.data.astype(np.float64), flat_grad.astype(np.float64)

        new_params = scipy.optimize.fmin_l_bfgs_b(
            f_loss_grad, cur_params, maxiter=10)[0]

        set_flat_params(self, new_params)


class MLPBaseline(NNFeatureBaseline, MLPFeatureModel):
    pass
