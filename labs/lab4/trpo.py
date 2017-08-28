from alg_utils import *
from simplepg.simple_utils import test_once
import tests.trpo_tests


def fvp(policy, f_kl, grad0, v, eps=1e-5, damping=1e-8):
    """
    Approximately compute the Fisher-vector product of the provided policy, F(x)v, where x is the current policy parameter
    and v is the vector we want to form product with.

    Define g(x) to be the gradient of the KL divergence (f_kl) evaluated at x. Note that for small \\epsilon, Taylor expansion gives
    g(x + \\epsilon v) ≈ g(x) + \\epsilon F(x)v
    So
    F(x)v \\approx (g(x + \epsilon v) - g(x)) / \\epsilon
    Since x is always the current parameters, we cache the computation of g(x) and this is provided as an input, grad0

    :param policy: The policy to compute Fisher-vector product.
    :param f_kl: A function which computes the average KL divergence.
    :param grad0: The gradient of KL divergence evaluated at the current parameter x.
    :param v: The vector we want to compute product with.
    :param eps: A small perturbation for finite difference computation.
    :param damping: A small damping factor to ensure that the Fisher information matrix is positive definite.
    :return:
    """

    flat_params = get_flat_params(policy)

    # compute g(x + \epsilon v)
    set_flat_params(policy, flat_params + eps * v)
    policy.cleargrads()
    f_kl().backward()
    grad_plus = get_flat_grad(policy)

    # don't forget to restore the policy parameters!
    set_flat_params(policy, flat_params)

    # form the finite difference
    return (grad_plus - grad0) / (eps) + damping * flat_params


def linesearch(f, x0, dx, expected_improvement, y0=None, backtrack_ratio=0.8, max_backtracks=15, accept_ratio=0.1,
               atol=1e-7):
    """
    Perform line search on the function f at x, where
    :param f: The function to perform line search on.
    :param x0: The current parameter value.
    :param dx: The full descent direction. We will shrink along this direction.
    :param y0: The initial value of f at x (optional).
    :param backtrack_ratio: Ratio to shrink the descent direction per line search step.
    :param max_backtracks: Maximum number of backtracking steps
    :param expected_improvement: Expected amount of improvement when taking the full descent direction dx, typically
           computed by y0 - y \\approx (f_x|x=x0).dot(dx), where f_x|x=x0 is the gradient of f w.r.t. x, evaluated at x0.
    :param accept_ratio: minimum acceptance ratio of actual_improvement / expected_improvement
    :return: The descent step obtained
    """
    if expected_improvement >= atol:
        if y0 is None:
            y0 = f(x0)
        for ratio in backtrack_ratio ** np.arange(max_backtracks):
            x = x0 - ratio * dx
            y = f(x)
            actual_improvement = y0 - y
            if actual_improvement / (expected_improvement * ratio) >= accept_ratio:
                logger.logkv("ExpectedImprovement",
                             expected_improvement * ratio)
                logger.logkv("ActualImprovement", actual_improvement)
                logger.logkv("ImprovementRatio", actual_improvement /
                             (expected_improvement * ratio))
                return x
    logger.logkv("ExpectedImprovement", expected_improvement)
    logger.logkv("ActualImprovement", 0.)
    logger.logkv("ImprovementRatio", 0.)
    return x0


def trpo(env, env_maker, policy, baseline, n_envs=mp.cpu_count(), last_iter=-1, n_iters=100, batch_size=1000,
         discount=0.99, gae_lambda=0.97, step_size=0.01, use_linesearch=True, kl_subsamp_ratio=1., snapshot_saver=None):
    """
    This method implements Trust Region Policy Optimization. Without the line search step, this algorithm is equivalent
    to an approximate procedure for computing natural gradient using conjugate gradients, where it performs approximate
    Hessian-vector product computation using finite differences.

    :param env: An environment instance, which should have the same class as what env_maker.make() returns.
    :param env_maker: An object such that calling env_maker.make() will generate a new environment.
    :param policy: A stochastic policy which we will be optimizing.
    :param baseline: A baseline used for variance reduction and estimating future returns for unfinished trajectories.
    :param n_envs: Number of environments running simultaneously.
    :param last_iter: The index of the last iteration. This is normally -1 when starting afresh, but may be different when
           loaded from a snapshot.
    :param n_iters: The total number of iterations to run.
    :param batch_size: The number of samples used per iteration.
    :param discount: Discount factor.
    :param gae_lambda: Lambda parameter used for generalized advantage estimation. For details see the following paper:
    :param step_size: The maximum value of average KL divergence allowed per iteration.
    :param use_linesearch: Whether to perform line search using the surrogate loss derived in the TRPO algorithm.
           Without this step, the algorithm is equivalent to an implementation of natural policy gradient where we use
           conjugate gradient algorithm to approximately compute F^{-1}g, where F is the Fisher information matrix, and
           g is the policy gradient.
    :param kl_subsamp_ratio: The ratio we use to subsample data in computing the Hessian-vector products. This can
           potentially save a lot of time.
    :param snapshot_saver: An object for saving snapshots.
    """

    logger.info("Starting env pool")
    with EnvPool(env_maker, n_envs=n_envs) as env_pool:
        for iter in range(last_iter + 1, n_iters):
            logger.info("Starting iteration {}".format(iter))
            logger.logkv('Iteration', iter)

            logger.info("Start collecting samples")
            trajs = parallel_collect_samples(env_pool, policy, batch_size)

            logger.info("Computing input variables for policy optimization")
            all_obs, all_acts, all_advs, all_dists = compute_pg_vars(
                trajs, policy, baseline, discount, gae_lambda)

            logger.info("Performing policy update")

            # subsample for kl divergence computation
            mask = np.zeros(len(all_obs), dtype=np.bool)
            mask_ids = np.random.choice(len(all_obs), size=int(
                np.ceil(len(all_obs) * kl_subsamp_ratio)), replace=False)
            mask[mask_ids] = 1
            if kl_subsamp_ratio < 1:
                subsamp_obs = all_obs[mask]
                subsamp_dists = policy.distribution.from_dict(
                    {k: v[mask] for k, v in all_dists.as_dict().items()})
            else:
                subsamp_obs = all_obs
                subsamp_dists = all_dists

            # Define helper functions to compute surrogate loss and/or KL divergence. They share part of the computation
            # graph, so we use a common function to decide whether we should compute both (which is needed in the line
            # search phase)
            def f_loss_kl_impl(need_loss, need_kl):
                retval = dict()
                if need_loss:
                    new_dists = policy.compute_dists(all_obs)
                    old_dists = all_dists
                elif need_kl:
                    # if only kl is needed, compute distribution from sub-sampled data
                    new_dists = policy.compute_dists(subsamp_obs)
                    old_dists = subsamp_dists

                def compute_surr_loss(old_dists, new_dists, all_acts, all_advs):
                    """
                    :param old_dists: An instance of subclass of Distribution
                    :param new_dists: An instance of subclass of Distribution
                    :param all_acts: A chainer variable, which should be a matrix of size N * |A|
                    :param all_advs: A chainer variable, which should be a vector of size N
                    :return: A chainer variable, which should be a scalar
                    """
                    surr_loss = Variable(np.array(0.))
                    "*** YOUR CODE HERE ***"
                    return surr_loss

                def compute_kl(old_dists, new_dists):
                    """
                    :param old_dists: An instance of subclass of Distribution
                    :param new_dists: An instance of subclass of Distribution
                    :return: A chainer variable, which should be a scalar
                    """
                    kl = Variable(np.array(0.))
                    "*** YOUR CODE HERE ***"
                    return kl

                test_once(compute_surr_loss)
                test_once(compute_kl)

                if need_loss:
                    retval["surr_loss"] = compute_surr_loss(
                        old_dists, new_dists, all_acts, all_advs)
                if need_kl:
                    retval["kl"] = compute_kl(old_dists, new_dists)
                return retval

            def f_surr_loss():
                return f_loss_kl_impl(need_loss=True, need_kl=False)["surr_loss"]

            def f_kl():
                return f_loss_kl_impl(need_loss=False, need_kl=True)["kl"]

            def f_surr_loss_kl():
                retval = f_loss_kl_impl(need_loss=True, need_kl=True)
                return retval["surr_loss"], retval["kl"]

            # Step 1: compute gradient in Euclidean space

            logger.info("Computing gradient in Euclidean space")

            policy.cleargrads()

            surr_loss = f_surr_loss()
            surr_loss.backward()

            # obtain the flattened gradient vector
            flat_grad = get_flat_grad(policy)

            # Optimize memory usage: after getting the gradient, we do not need the rest of the computation graph
            # anymore
            surr_loss.unchain_backward()

            # Step 2: Perform conjugate gradient to compute approximate natural gradient

            logger.info(
                "Computing approximate natural gradient using conjugate gradient algorithm")

            policy.cleargrads()

            f_kl().backward()
            flat_kl_grad = get_flat_grad(policy)

            def Fx(v):
                return fvp(policy, f_kl, flat_kl_grad, v)

            descent_direction = cg(Fx, flat_grad)

            # Step 3: Compute initial step size

            # We'd like D_KL(old||new) <= step_size
            # The 2nd order approximation gives 1/2*d^T*H*d <= step_size, where d is the descent step
            # Hence given the initial direction d_0 we can rescale it so that this constraint is tight
            # Let this scaling factor be \alpha, i.e. d = \alpha*d_0
            # Solving 1/2*\alpha^2*d_0^T*H*d_0 = step_size we get \alpha = \sqrt(2 * step_size / d_0^T*H*d_0)

            scale = np.sqrt(
                2.0 * step_size *
                (1. / (descent_direction.dot(Fx(descent_direction)) + 1e-8))
            )

            descent_step = descent_direction * scale

            cur_params = get_flat_params(policy)

            if use_linesearch:
                # Step 4: Perform line search

                logger.info("Performing line search")

                expected_improvement = flat_grad.dot(descent_step)

                def f_barrier(x):
                    set_flat_params(policy, x)
                    with chainer.no_backprop_mode():
                        surr_loss, kl = f_surr_loss_kl()
                    return surr_loss.data + 1e100 * max(kl.data - step_size, 0.)

                new_params = linesearch(
                    f_barrier,
                    x0=cur_params,
                    dx=descent_step,
                    y0=surr_loss.data,
                    expected_improvement=expected_improvement
                )

            else:
                new_params = cur_params - descent_step

            set_flat_params(policy, new_params)

            logger.info("Updating baseline")
            baseline.update(trajs)

            # log statistics
            logger.info("Computing logging information")
            with chainer.no_backprop_mode():
                mean_kl = f_kl().data
            logger.logkv('MeanKL', mean_kl)
            log_action_distribution_statistics(all_dists)
            log_reward_statistics(env)
            log_baseline_statistics(trajs)
            logger.dumpkvs()

            if snapshot_saver is not None:
                logger.info("Saving snapshot")
                snapshot_saver.save_state(
                    iter,
                    dict(
                        alg=trpo,
                        alg_state=dict(
                            env_maker=env_maker,
                            policy=policy,
                            baseline=baseline,
                            n_envs=n_envs,
                            last_iter=iter,
                            n_iters=n_iters,
                            batch_size=batch_size,
                            discount=discount,
                            gae_lambda=gae_lambda,
                            step_size=step_size,
                            use_linesearch=use_linesearch,
                            kl_subsamp_ratio=kl_subsamp_ratio,
                        )
                    )
                )
