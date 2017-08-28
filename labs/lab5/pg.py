from alg_utils import *
from simplepg.simple_utils import test_once, nprs
import tests.pg_tests


def pg(env, env_maker, policy, baseline, n_envs=mp.cpu_count(), last_iter=-1, n_iters=100, batch_size=1000,
       optimizer=chainer.optimizers.Adam(), discount=0.99, gae_lambda=0.97, snapshot_saver=None):
    """
    This method implements policy gradient algorithm.
    :param env: An environment instance, which should have the same class as what env_maker.make() returns.
    :param env_maker: An object such that calling env_maker.make() will generate a new environment.
    :param policy: A stochastic policy which we will be optimizing.
    :param baseline: A baseline used for variance reduction and estimating future returns for unfinished trajectories.
    :param n_envs: Number of environments running simultaneously.
    :param last_iter: The index of the last iteration. This is normally -1 when starting afresh, but may be different when
           loaded from a snapshot.
    :param n_iters: The total number of iterations to run.
    :param batch_size: The number of samples used per iteration.
    :param optimizer: A Chainer optimizer instance. By default we use the Adam algorithm with learning rate 1e-3.
    :param discount: Discount factor.
    :param gae_lambda: Lambda parameter used for generalized advantage estimation.
    :param snapshot_saver: An object for saving snapshots.
    """

    if getattr(optimizer, 'target', None) is not policy:
        optimizer.setup(policy)

    logger.info("Starting env pool")
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        for iter in range(last_iter + 1, n_iters):
            logger.info("Starting iteration {}".format(iter))
            logger.logkv('Iteration', iter)

            logger.info("Start collecting samples")
            trajs = parallel_collect_samples(env_pool, policy, batch_size)

            logger.info("Computing input variables for policy optimization")
            all_obs, all_acts, all_advs, _ = compute_pg_vars(
                trajs, policy, baseline, discount, gae_lambda
            )

            # Begin policy update

            # Now, you need to implement the computation of the policy gradient
            # The policy gradient is given by -1/T \sum_t \nabla_\theta(log(p_\theta(a_t|s_t))) * A_t
            # Note the negative sign in the front, since optimizers are most often minimizing a loss rather
            # This is the same as \nabla_\theta(-1/T \sum_t log(p_\theta(a_t|s_t)) * A_t) = \nabla_\theta(L), where L is the surrogate loss term

            logger.info("Computing policy gradient")

            # Methods that may be useful:
            # - `dists.logli(actions)' returns the log probability of the actions under the distribution `dists'.
            #   This method returns a chainer variable.

            dists = policy.compute_dists(all_obs)

            def compute_surr_loss(dists, all_acts, all_advs):
                """
                :param dists: An instance of subclass of Distribution
                :param all_acts: A chainer variable, which should be a matrix of size N * |A|
                :param all_advs: A chainer variable, which should be a vector of size N
                :return: A chainer variable, which should be a scalar
                """
                surr_loss = Variable(np.array(0.))
                logli = dists.logli(all_acts)
                surr_loss = -F.mean(logli * all_advs)
                "*** YOUR CODE HERE ***"
                return surr_loss

            test_once(compute_surr_loss)

            surr_loss = compute_surr_loss(dists, all_acts, all_advs)

            # reset gradients stored in the policy parameters
            policy.cleargrads()
            surr_loss.backward()

            # apply the computed gradient
            optimizer.update()

            # Update baseline
            logger.info("Updating baseline")
            baseline.update(trajs)

            # log statistics
            logger.info("Computing logging information")
            logger.logkv('SurrLoss', surr_loss.data)
            log_action_distribution_statistics(dists)
            log_reward_statistics(env)
            log_baseline_statistics(trajs)
            logger.dumpkvs()

            if snapshot_saver is not None:
                logger.info("Saving snapshot")
                snapshot_saver.save_state(
                    iter,
                    dict(
                        alg=pg,
                        alg_state=dict(
                            env_maker=env_maker,
                            policy=policy,
                            baseline=baseline,
                            n_envs=n_envs,
                            last_iter=iter,
                            n_iters=n_iters,
                            batch_size=batch_size,
                            optimizer=optimizer,
                            discount=discount,
                            gae_lambda=gae_lambda
                        )
                    )
                )
