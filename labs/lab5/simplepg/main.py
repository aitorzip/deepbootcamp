#!/usr/bin/env python
from collections import defaultdict
import numpy as np
import gym
import click
from simplepg.simple_utils import gradient_check, log_softmax, softmax, weighted_sample, include_bias, test_once, nprs
import tests.simplepg_tests


##############################
# Methods for Point-v0
##############################

def point_get_logp_action(theta, ob, action):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :param action: A vector of size |A|
    :return: A scalar
    """
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    zs = action - mean
    return -0.5 * np.log(2 * np.pi) * theta.shape[0] - 0.5 * np.sum(np.square(zs))


def point_get_grad_logp_action(theta, ob, action):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :param action: A vector of size |A|
    :return: A matrix of size |A| * (|S|+1)
    """
    grad = np.zeros_like(theta)
    ob_1 = include_bias(ob)
    grad = np.outer(action - theta.dot(ob_1), ob_1)
    "*** YOUR CODE HERE ***"
    return grad


def point_get_action(theta, ob, rng=np.random):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :return: A vector of size |A|
    """
    ob_1 = include_bias(ob)
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)


def point_test_grad_impl():
    # check gradient implementation
    rng = nprs(42)
    test_ob = rng.uniform(size=(4,))
    test_act = rng.uniform(size=(4,))
    test_theta = rng.uniform(size=(4, 5))
    # Check that the shape matches
    assert point_get_grad_logp_action(
        test_theta, test_ob, test_act).shape == test_theta.shape
    gradient_check(
        lambda x: point_get_logp_action(
            x.reshape(test_theta.shape), test_ob, test_act),
        lambda x: point_get_grad_logp_action(
            x.reshape(test_theta.shape), test_ob, test_act).flatten(),
        test_theta.flatten()
    )


##############################
# Methods for Cartpole-v0
##############################

def compute_logits(theta, ob):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :return: A vector of size |A|
    """
    ob_1 = include_bias(ob)
    logits = ob_1.dot(theta.T)
    return logits


def cartpole_get_logp_action(theta, ob, action):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :param action: An integer
    :return: A scalar
    """
    return log_softmax(compute_logits(theta, ob))[action]


def cartpole_get_grad_logp_action(theta, ob, action):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :param action: An integer
    :return: A matrix of size |A| * (|S|+1)
    """
    grad = np.zeros_like(theta)
    ob_1 = include_bias(ob)
    logits = theta.dot(ob_1)
    probs = softmax(logits)
    dlogits = -probs
    dlogits[action] += 1
    grad = np.outer(dlogits, ob_1)
    "*** YOUR CODE HERE ***"
    return grad


def cartpole_get_action(theta, ob, rng=np.random):
    """
    :param theta: A matrix of size |A| * (|S|+1)
    :param ob: A vector of size |S|
    :return: An integer
    """
    return weighted_sample(compute_logits(theta, ob), rng=rng)


def cartpole_test_grad_impl():
    # check gradient implementation
    rng = nprs(42)
    test_ob = rng.uniform(size=(4,))
    test_act = rng.choice(4)
    test_theta = rng.uniform(size=(4, 5))
    # Check that the shape matches
    assert cartpole_get_grad_logp_action(
        test_theta, test_ob, test_act).shape == test_theta.shape
    gradient_check(
        lambda x: cartpole_get_logp_action(
            x.reshape(test_theta.shape), test_ob, test_act),
        lambda x: cartpole_get_grad_logp_action(
            x.reshape(test_theta.shape), test_ob, test_act).flatten(),
        test_theta.flatten()
    )


def compute_entropy(logits):
    """
    :param logits: A matrix of size N * |A|
    :return: A vector of size N
    """
    logp = log_softmax(logits)
    return -np.sum(logp * np.exp(logp), axis=-1)


@click.command()
@click.argument("env_id", type=str, default="Point-v0")
@click.option("--batch_size", type=int, default=2000)
@click.option("--discount", type=float, default=0.99)
@click.option("--learning_rate", type=float, default=0.1)
@click.option("--n_itrs", type=int, default=100)
@click.option("--render", type=bool, default=False)
@click.option("--use-baseline", type=bool, default=True)
@click.option("--natural", type=bool, default=False)
@click.option("--natural_step_size", type=float, default=0.01)
def main(env_id, batch_size, discount, learning_rate, n_itrs, render, use_baseline, natural, natural_step_size):
    # Check gradient implementation

    rng = np.random.RandomState(42)

    if env_id == 'CartPole-v0':
        cartpole_test_grad_impl()
        env = gym.make('CartPole-v0')
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        get_action = cartpole_get_action
        get_grad_logp_action = cartpole_get_grad_logp_action
    elif env_id == 'Point-v0':
        point_test_grad_impl()
        from simplepg import point_env
        env = gym.make('Point-v0')
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        get_action = point_get_action
        get_grad_logp_action = point_get_grad_logp_action
    else:
        raise ValueError(
            "Unsupported environment: must be one of 'CartPole-v0', 'Point-v0'")

    env.seed(42)
    timestep_limit = env.spec.timestep_limit

    # Initialize parameters
    theta = rng.normal(scale=0.1, size=(action_dim, obs_dim + 1))

    # Store baselines for each time step.
    baselines = np.zeros(timestep_limit)

    # Policy training loop
    for itr in range(n_itrs):
        # Collect trajectory loop
        n_samples = 0
        grad = np.zeros_like(theta)
        episode_rewards = []

        # Store cumulative returns for each time step
        all_returns = [[] for _ in range(timestep_limit)]

        all_observations = []
        all_actions = []

        while n_samples < batch_size:
            observations = []
            actions = []
            rewards = []
            ob = env.reset()
            done = False
            # Only render the first trajectory
            render_episode = n_samples == 0
            # Collect a new trajectory
            while not done:
                action = get_action(theta, ob, rng=rng)
                next_ob, rew, done, _ = env.step(action)
                observations.append(ob)
                actions.append(action)
                rewards.append(rew)
                ob = next_ob
                n_samples += 1
                if render and render_episode:
                    env.render()
            # Go back in time to compute returns and accumulate gradient
            # Compute the gradient along this trajectory
            R = 0.
            for t in reversed(range(len(observations))):
                def compute_update(discount, R_tplus1, theta, s_t, a_t, r_t, b_t, get_grad_logp_action):
                    """
                    :param discount: A scalar
                    :param R_tplus1: A scalar
                    :param theta: A matrix of size |A| * (|S|+1)
                    :param s_t: A vector of size |S|
                    :param a_t: Either a vector of size |A| or an integer, depending on the environment
                    :param r_t: A scalar
                    :param b_t: A scalar
                    :param get_grad_logp_action: A function, mapping from (theta, ob, action) to the gradient (a 
                    matrix of size |A| * (|S|+1) )
                    :return: A tuple, consisting of a scalar and a matrix of size |A| * (|S|+1)
                    """
                    R_t = 0.
                    pg_theta = np.zeros_like(theta)
                    R_t = r_t + discount * R_tplus1
                    pg_theta = get_grad_logp_action(
                        theta, s_t, a_t) * (R_t - b_t)
                    "*** YOUR CODE HERE ***"
                    return R_t, pg_theta

                # Test the implementation, but only once
                test_once(compute_update)

                R, grad_t = compute_update(
                    discount=discount,
                    R_tplus1=R,
                    theta=theta,
                    s_t=observations[t],
                    a_t=actions[t],
                    r_t=rewards[t],
                    b_t=baselines[t],
                    get_grad_logp_action=get_grad_logp_action
                )
                all_returns[t].append(R)
                grad += grad_t

            episode_rewards.append(np.sum(rewards))
            all_observations.extend(observations)
            all_actions.extend(actions)

        def compute_baselines(all_returns):
            """
            :param all_returns: A list of size T, where the t-th entry is a list of numbers, denoting the returns 
            collected at time step t across different episodes
            :return: A vector of size T
            """
            baselines = np.zeros(len(all_returns))
            for t in range(len(all_returns)):
                # Update the baselines
                if len(all_returns[t]) > 0:
                    baselines[t] = np.mean(all_returns[t])
                else:
                    baselines[t] = 0.
                "*** YOUR CODE HERE ***"
            return baselines

        if use_baseline:
            test_once(compute_baselines)
            baselines = compute_baselines(all_returns)
        else:
            baselines = np.zeros(timestep_limit)

        # Roughly normalize the gradient
        grad = grad / (np.linalg.norm(grad) + 1e-8)

        if not natural:

            theta += learning_rate * grad
        else:
            def compute_fisher_matrix(theta, get_grad_logp_action, all_observations, all_actions):
                """
                :param theta: A matrix of size |A| * (|S|+1)
                :param get_grad_logp_action: A function, mapping from (theta, ob, action) to the gradient (a matrix 
                of size |A| * (|S|+1) )
                :param all_observations: A list of vectors of size |S|
                :param all_actions: A list of vectors of size |A|
                :return: A matrix of size (|A|*(|S|+1)) * (|A|*(|S|+1)), i.e. #columns and #rows are the number of 
                entries in theta
                """
                d = len(theta.flatten())
                F = np.zeros((d, d))
                for ob, action in zip(all_observations, all_actions):
                    g = get_grad_logp_action(theta, ob, action).flatten()
                    F += np.outer(g, g)
                F = F / len(all_observations)
                "*** YOUR CODE HERE ***"
                return F

            def compute_natural_gradient(F, grad, reg=1e-4):
                """
                :param F: A matrix of size (|A|*(|S|+1)) * (|A|*(|S|+1))
                :param grad: A matrix of size |A| * (|S|+1)
                :param reg: A scalar
                :return: A matrix of size |A| * (|S|+1)
                """
                natural_grad = np.zeros_like(grad)
                natural_grad = np.linalg.inv(
                    F + reg * np.eye(len(F))).dot(grad.flatten())
                natural_grad = natural_grad.reshape(grad.shape)
                "*** YOUR CODE HERE ***"
                return natural_grad

            def compute_step_size(F, natural_grad, natural_step_size):
                """
                :param F: A matrix of size (|A|*(|S|+1)) * (|A|*(|S|+1))
                :param natural_grad: A matrix of size |A| * (|S|+1)
                :param natural_step_size: A scalar
                :return: A scalar
                """
                step_size = 0.
                natural_grad = natural_grad.flatten()
                gFg = natural_grad.dot(F.dot(natural_grad))
                step_size = np.sqrt(2 * natural_step_size / gFg)
                "*** YOUR CODE HERE ***"
                return step_size

            test_once(compute_fisher_matrix)
            test_once(compute_natural_gradient)
            test_once(compute_step_size)

            F = compute_fisher_matrix(theta=theta, get_grad_logp_action=get_grad_logp_action,
                                      all_observations=all_observations, all_actions=all_actions)
            natural_grad = compute_natural_gradient(F, grad)
            step_size = compute_step_size(F, natural_grad, natural_step_size)
            theta += step_size * natural_grad

        if env_id == 'CartPole-v0':
            logits = compute_logits(theta, np.array(all_observations))
            ent = np.mean(compute_entropy(logits))
            perp = np.exp(ent)

            print("Iteration: %d AverageReturn: %.2f Entropy: %.2f Perplexity: %.2f |theta|_2: %.2f" % (
                itr, np.mean(episode_rewards), ent, perp, np.linalg.norm(theta)))
        else:
            print("Iteration: %d AverageReturn: %.2f |theta|_2: %.2f" % (
                itr, np.mean(episode_rewards), np.linalg.norm(theta)))


if __name__ == "__main__":
    main()
