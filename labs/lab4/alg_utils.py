from utils import *


# ==============================
# Shared utilities
# ==============================

def compute_cumulative_returns(rewards, baselines, discount):
    # This method builds up the cumulative sum of discounted rewards for each time step:
    # R[t] = sum_{t'>=t} γ^(t'-t)*r_t'
    # Note that we use γ^(t'-t) instead of γ^t'. This gives us a biased gradient but lower variance
    returns = []
    # Use the last baseline prediction to back up
    cum_return = baselines[-1]
    for reward in rewards[::-1]:
        cum_return = cum_return * discount + reward
        returns.append(cum_return)
    return returns[::-1]


def compute_advantages(rewards, baselines, discount, gae_lambda):
    # Given returns R_t and baselines b(s_t), compute (generalized) advantage estimate A_t
    deltas = rewards + discount * baselines[1:] - baselines[:-1]
    advs = []
    cum_adv = 0
    multiplier = discount * gae_lambda
    for delta in deltas[::-1]:
        cum_adv = cum_adv * multiplier + delta
        advs.append(cum_adv)
    return advs[::-1]


def compute_pg_vars(trajs, policy, baseline, discount, gae_lambda):
    """
    Compute chainer variables needed for various policy gradient algorithms
    """
    for traj in trajs:
        # Include the last observation here, in case the trajectory is not finished
        baselines = baseline.predict(np.concatenate(
            [traj["observations"], [traj["last_observation"]]]))
        if traj['finished']:
            # If already finished, the future cumulative rewards starting from the final state is 0
            baselines[-1] = 0.
        # This is useful when fitting baselines. It uses the baseline prediction of the last state value to perform
        # Bellman backup if the trajectory is not finished.
        traj['returns'] = compute_cumulative_returns(
            traj['rewards'], baselines, discount)
        traj['advantages'] = compute_advantages(
            traj['rewards'], baselines, discount, gae_lambda)
        traj['baselines'] = baselines[:-1]

    # First, we compute a flattened list of observations, actions, and advantages
    all_obs = np.concatenate([traj['observations'] for traj in trajs], axis=0)
    all_acts = np.concatenate([traj['actions'] for traj in trajs], axis=0)
    all_advs = np.concatenate([traj['advantages'] for traj in trajs], axis=0)
    all_dists = {
        k: np.concatenate([traj['distributions'][k] for traj in trajs], axis=0)
        for k in trajs[0]['distributions'].keys()
    }

    # Normalizing the advantage values can make the algorithm more robust to reward scaling
    all_advs = (all_advs - np.mean(all_advs)) / (np.std(all_advs) + 1e-8)

    # Form chainer variables
    all_obs = Variable(all_obs)
    all_acts = Variable(all_acts)
    all_advs = Variable(all_advs.astype(np.float32, copy=False))
    all_dists = policy.distribution.from_dict(
        {k: Variable(v) for k, v in all_dists.items()})

    return all_obs, all_acts, all_advs, all_dists


# ==============================
# Helper methods for logging
# ==============================

def log_reward_statistics(env):
    # keep unwrapping until we get the monitor
    while not isinstance(env, gym.wrappers.Monitor):  # and not isinstance()
        if not isinstance(env, gym.Wrapper):
            assert False
        env = env.env
    # env.unwrapped
    assert isinstance(env, gym.wrappers.Monitor)
    all_stats = None
    for _ in range(10):
        try:
            all_stats = gym.wrappers.monitoring.load_results(env.directory)
        except FileNotFoundError:
            time.sleep(1)
            continue
    if all_stats is not None:
        episode_rewards = all_stats['episode_rewards']
        episode_lengths = all_stats['episode_lengths']

        recent_episode_rewards = episode_rewards[-100:]
        recent_episode_lengths = episode_lengths[-100:]

        if len(recent_episode_rewards) > 0:
            logger.logkv('AverageReturn', np.mean(recent_episode_rewards))
            logger.logkv('MinReturn', np.min(recent_episode_rewards))
            logger.logkv('MaxReturn', np.max(recent_episode_rewards))
            logger.logkv('StdReturn', np.std(recent_episode_rewards))
            logger.logkv('AverageEpisodeLength',
                         np.mean(recent_episode_lengths))
            logger.logkv('MinEpisodeLength', np.min(recent_episode_lengths))
            logger.logkv('MaxEpisodeLength', np.max(recent_episode_lengths))
            logger.logkv('StdEpisodeLength', np.std(recent_episode_lengths))

        logger.logkv('TotalNEpisodes', len(episode_rewards))
        logger.logkv('TotalNSamples', np.sum(episode_lengths))


def log_baseline_statistics(trajs):
    # Specifically, compute the explained variance, defined as
    baselines = np.concatenate([traj['baselines'] for traj in trajs])
    returns = np.concatenate([traj['returns'] for traj in trajs])
    logger.logkv('ExplainedVariance',
                 explained_variance_1d(baselines, returns))


def log_action_distribution_statistics(dists):
    with chainer.no_backprop_mode():
        entropy = F.mean(dists.entropy()).data
        logger.logkv('Entropy', entropy)
        logger.logkv('Perplexity', np.exp(entropy))
        if isinstance(dists, Gaussian):
            logger.logkv('AveragePolicyStd', F.mean(
                F.exp(dists.log_stds)).data)
            for idx in range(dists.log_stds.shape[-1]):
                logger.logkv('AveragePolicyStd[{}]'.format(
                    idx), F.mean(F.exp(dists.log_stds[..., idx])).data)
        elif isinstance(dists, Categorical):
            probs = F.mean(F.softmax(dists.logits), axis=0).data
            for idx in range(len(probs)):
                logger.logkv('AveragePolicyProb[{}]'.format(idx), probs[idx])
