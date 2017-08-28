from alg_utils import *
from models import WeightSharingValueFunction
from simplepg.simple_utils import test_once
import tests.a2c_tests


def samples_generator(env_pool, policy, vf, k):
    def compute_dists_values(obs):
        # Special handling when value function and policy share weights. We try to avoid an additional forward pass
        if isinstance(vf, WeightSharingValueFunction) and vf.feature_model is policy:
            feats = policy.compute_features(obs)
            dists = policy.compute_dists(obs=None, feats=feats)
            values = vf.compute_state_values(obs=None, feats=feats)
        else:
            dists = policy.compute_dists(obs)
            values = vf.compute_state_values(obs)
        return dists, values

    obs = Variable(np.asarray(env_pool.reset()))

    dists, values = compute_dists_values(obs)

    while True:
        all_actions = []
        all_rewards = []
        all_dones = []
        all_dists = []
        all_values = []

        for _ in range(k):
            # To reuse computation, we retain the computation graph for actions and distributions, so that we can
            # backprop later without an additional forward pass

            # Make sure we don't accidentally differentiate through the actions
            with chainer.no_backprop_mode():
                actions = dists.sample()

            actions_val = actions.data

            next_obs, rews, dones, _ = env_pool.step(actions_val)

            all_actions.append(actions)
            all_rewards.append(rews)
            all_dones.append(dones)
            all_dists.append(dists.as_dict())
            all_values.append(values)

            obs = Variable(np.asarray(next_obs))

            dists, values = compute_dists_values(obs)

        yield all_actions, all_rewards, all_dones, all_dists, all_values, chainer.Variable(values.data)


def compute_returns_advantages(rewards, dones, values, next_values, discount):
    """
    Compute returns and advantages given rewards, terminal indicators, values at each state, values at the future
    states, and the discount factor.
    :param rewards: A matrix of shape T * N, where T is the number of time steps and N is the number of environments.
           Each entry is the reward value.
    :param dones: A matrix of shape T * N, where each entry is a binary flag indicating whether
           an episode has finished after this time step.
    :param values: A matrix of shape T * N, where each entry is the estimated value V(s_t), t = 0, ..., T-1.
    :param next_values: A vector of length N, where each entry is the estimated value at the future state V(s_T).
    :param discount: The discount factor.
    :return: A tuple (returns, advantages), each of which should be a matrix of shape T * N
    """
    Rs = np.zeros_like(rewards)
    As = np.zeros_like(rewards)
    "*** YOUR CODE HERE ***"


def a2c(env, env_maker, policy, vf, joint_model=None, k=20, n_envs=16, discount=0.99,
        optimizer=chainer.optimizers.RMSprop(lr=1e-3), max_grad_norm=1.0, vf_loss_coeff=0.5,
        ent_coeff=0.01, last_epoch=-1, epoch_length=10000, n_epochs=8000, snapshot_saver=None):
    """
    This method implements (Synchronous) Advantage Actor-Critic algorithm. Rather than having asynchronous workers,
    which can be more efficient due to less coordination but also less stable and harder to extend / debug, we use a

    pool of environment workers performing simulation, while computing actions and performing gradient updates
    centrally. This also makes it easier to utilize GPUs for neural network computation.
    :param env: An environment instance, which should have the same class as what env_maker.make() returns.
    :param env_maker: An object such that calling env_maker.make() will generate a new environment.
    :param policy: A stochastic policy which we will be optimizing.
    :param vf: A value function which estimates future returns given a state. It can potentially share weights with the
           policy by calling policy.create_vf().
    :param joint_model: The joint model of policy and value function. This is usually automatically computed.
    :param k: Number of simulation steps per environment for each gradient update.
    :param n_envs: Number of environments running simultaneously.
    :param discount: Discount factor.
    :param optimizer: A chainer optimizer instance. By default we use the RMSProp algorithm.
    :param max_grad_norm: If provided, apply gradient clipping with the specified maximum L2 norm.
    :param vf_loss_coeff: Coefficient for the value function loss.
    :param ent_coeff: Coefficient for the entropy loss (the negative bonus).
    :param last_epoch: The index of the last epoch. This is normally -1 when starting afresh, but may be different when
           loaded from a snapshot.
    :param epoch: The starting epoch. This is normally 0, but may be different when loaded from a snapshot. Since A2C
           is an online algorithm, an epoch is just an artificial boundary so that we record logs after each epoch.
    :param epoch_length: Number of total environment steps per epoch.
    :param n_epochs: Total number of epochs to run the algorithm.
    :param snapshot_saver: An object for saving snapshots.
    """

    # ensures that shared parameters are only counted once
    if joint_model is None:
        joint_model = UniqueChainList(policy, vf)

    if getattr(optimizer, 'target', None) is not joint_model:
        optimizer.setup(joint_model)

    try:
        # remove existing hook if necessary (this should only be needed when restarting experiments)
        optimizer.remove_hook('gradient_clipping')
    except KeyError:
        pass
    if max_grad_norm is not None:
        # Clip L2 norm of gradient, to improve stability
        optimizer.add_hook(chainer.optimizer.GradientClipping(
            threshold=max_grad_norm), 'gradient_clipping')

    epoch = last_epoch + 1
    global_t = epoch * epoch_length

    loggings = defaultdict(list)

    logger.info("Starting env pool")
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:

        gen = samples_generator(env_pool, policy, vf, k)

        logger.info("Starting epoch {}".format(epoch))

        if logger.get_level() <= logger.INFO:
            progbar = tqdm(total=epoch_length)
        else:
            progbar = None

        while global_t < epoch_length * n_epochs:

            # Run k steps in the environment
            # Note:
            # - all_actions, all_values, all_dists, and next_values are chainer variables
            # - all_rewards, all_dones are lists numpy arrays
            # The first dimension of each variable is time, and the second dimension is the index of the environment
            all_actions, all_rewards, all_dones, all_dists, all_values, next_values = next(
                gen)

            global_t += n_envs * k

            # Compute returns and advantages

            # Size: (k, n_envs)
            all_values = F.stack(all_values)
            all_rewards = np.asarray(all_rewards, dtype=np.float32)
            all_dones = np.asarray(all_dones, dtype=np.float32)

            all_values_data = all_values.data
            next_values_data = next_values.data

            test_once(compute_returns_advantages)

            all_returns, all_advs = compute_returns_advantages(
                all_rewards,
                all_dones,
                all_values_data,
                next_values_data,
                discount
            )

            all_returns = chainer.Variable(all_returns.astype(np.float32))
            all_advs = chainer.Variable(all_advs.astype(np.float32))

            # Concatenate data
            # Size: (k*n_envs,) + action_shape
            all_flat_actions = F.concat(all_actions, axis=0)
            # Size: key -> (k*n_envs,) + dist_shape
            all_flat_dists = {k: F.concat(
                [d[k] for d in all_dists], axis=0) for k in all_dists[0].keys()}
            all_flat_dists = policy.distribution.from_dict(all_flat_dists)

            # Prepare variables needed for gradient computation
            logli = all_flat_dists.logli(all_flat_actions)
            ent = all_flat_dists.entropy()
            # Flatten advantages
            all_advs = F.concat(all_advs, axis=0)

            # Form the loss - you should only need to use the variables provided as input arguments below
            def compute_total_loss(logli, all_advs, ent_coeff, ent, vf_loss_coeff, all_returns, all_values):
                """
                :param logli: A chainer variable, which should be a vector of size N
                :param all_advs: A chainer variable, which should be a vector of size N
                :param ent_coeff: A scalar
                :param ent: A chainer variable, which should be a vector of size N
                :param vf_loss_coeff: A scalar
                :param all_returns: A chainer variable, which should be a vector of size N
                :param all_values: A chainer variable, which should be a vector of size N
                :return: A tuple of (policy_loss, vf_loss, total_loss)
                policy_loss should be the weighted sum of the surrogate loss and the average entropy loss
                vf_loss should be the (unweighted) squared loss of value function prediction.
                total_loss should be the weighted sum of policy_loss and vf_loss
                """
                policy_loss = Variable(np.array(0.))
                vf_loss = Variable(np.array(0.))
                total_loss = Variable(np.array(0.))
                "*** YOUR CODE HERE ***"
                return policy_loss, vf_loss, total_loss

            test_once(compute_total_loss)

            policy_loss, vf_loss, total_loss = compute_total_loss(
                logli=logli, all_advs=all_advs, ent_coeff=ent_coeff,
                ent=ent, vf_loss_coeff=vf_loss_coeff,
                all_returns=all_returns, all_values=all_values
            )

            joint_model.cleargrads()
            total_loss.backward()
            optimizer.update()

            vf_loss_data = vf_loss.data
            all_returns_data = all_returns.data
            all_flat_dists_data = {
                k: v.data
                for k, v in all_flat_dists.as_dict().items()
            }

            loggings["vf_loss"].append(vf_loss_data)
            loggings["vf_preds"].append(all_values_data)
            loggings["vf_targets"].append(all_returns_data)
            loggings["dists"].append(all_flat_dists_data)

            if progbar is not None:
                progbar.update(k * n_envs)

            # An entire epoch has passed
            if global_t // epoch_length > epoch:
                logger.logkv('Epoch', epoch)
                log_reward_statistics(env)
                all_dists = {
                    k: Variable(np.concatenate([d[k] for d in loggings["dists"]], axis=0))
                    for k in loggings["dists"][0].keys()
                }
                log_action_distribution_statistics(
                    policy.distribution.from_dict(all_dists))
                logger.logkv('|VfPred|', np.mean(np.abs(loggings["vf_preds"])))
                logger.logkv('|VfTarget|', np.mean(
                    np.abs(loggings["vf_targets"])))
                logger.logkv('VfLoss', np.mean(loggings["vf_loss"]))
                logger.dumpkvs()

                if snapshot_saver is not None:
                    logger.info("Saving snapshot")

                    snapshot_saver.save_state(
                        epoch,
                        dict(
                            alg=a2c,
                            alg_state=dict(
                                env_maker=env_maker,
                                policy=policy,
                                vf=vf,
                                joint_model=joint_model,
                                k=k,
                                n_envs=n_envs,
                                discount=discount,
                                last_epoch=epoch,
                                n_epochs=n_epochs,
                                epoch_length=epoch_length,
                                optimizer=optimizer,
                                vf_loss_coeff=vf_loss_coeff,
                                ent_coeff=ent_coeff,
                                max_grad_norm=max_grad_norm,
                            )
                        )
                    )

                # Reset stored logging information
                loggings = defaultdict(list)

                if progbar is not None:
                    progbar.close()

                epoch = global_t // epoch_length

                logger.info("Starting epoch {}".format(epoch))

                if progbar is not None:
                    progbar = tqdm(total=epoch_length)

        if progbar is not None:
            progbar.close()
