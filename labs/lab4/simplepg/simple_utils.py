import numpy as np
import scipy.special
import chainer


# Compute gradient approximately using finite difference
def numerical_grad(f, x, eps=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        xplus = np.array(x)
        xplus[i] += eps
        fplus = f(xplus)
        xminus = np.array(x)
        xminus[i] -= eps
        fminus = f(xminus)
        grad[i] = (fplus - fminus) / (2 * eps)
    return grad


def gradient_check(f, g, x):
    # Test the implementation of g(x) = df/dx
    # Perform numerical differentiation and test it
    g_num = numerical_grad(f, x)
    g_test = g(x)
    try:
        np.testing.assert_allclose(g_num, g_test, rtol=1e-5)
        print("Gradient check passed!")
    except AssertionError as e:
        print(e)
        print("Error: Gradient check didn't pass!")
        exit()


def log_softmax(logits):
    return logits - scipy.special.logsumexp(logits, axis=-1, keepdims=True)


def softmax(logits):
    x = logits
    x = x - np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)


def weighted_sample(logits, rng=np.random):
    weights = softmax(logits)
    return min(
        int(np.sum(rng.uniform() > np.cumsum(weights))),
        len(weights) - 1
    )


def include_bias(x):
    # Add a constant term (1.0) to each entry in x
    return np.concatenate([x, np.ones_like(x[..., :1])], axis=-1)


_tested = set()
_tests = dict()

nprs = np.random.RandomState


def register_test(fn_name, kwargs, desired_output=None):
    assert fn_name not in _tests
    _tests[fn_name] = (kwargs, desired_output)


def assert_allclose(a, b):
    if isinstance(a, (np.ndarray, float, int)):
        np.testing.assert_allclose(a, b, rtol=1e-5)
    elif isinstance(a, (tuple, list)):
        assert isinstance(b, (tuple, list))
        assert len(a) == len(b)
        for a_i, b_i in zip(a, b):
            assert_allclose(a_i, b_i)
    elif isinstance(a, chainer.Variable):
        assert isinstance(b, chainer.Variable)
        assert_allclose(a.data, b.data)
    else:
        raise NotImplementedError


def test_once(fn):
    module = fn.__module__
    name = fn.__name__
    key = module + "." + name
    if key in _tested:
        return
    assert key in _tests, "Test for %s not found!" % key
    kwargs, desired_output = _tests[key]
    _tested.add(key)

    if callable(kwargs):
        kwargs = kwargs()

    if callable(desired_output):
        desired_output = desired_output()

    if desired_output is None:
        print("Desired output for %s:" % key, repr(fn(**kwargs)))
        exit()
    else:
        try:
            output = fn(**kwargs)
            assert_allclose(desired_output, output)
            print("Test for %s passed!" % key)
        except AssertionError as e:
            print(e)
            print("Error: test for %s didn't pass!" % key)
            exit()
