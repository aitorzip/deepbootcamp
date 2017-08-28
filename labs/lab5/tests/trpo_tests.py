from simplepg.simple_utils import register_test, nprs
import numpy as np
from chainer import Variable

from utils import Gaussian

register_test(
    "trpo.compute_surr_loss",
    kwargs=lambda: dict(
        old_dists=Gaussian(
            means=Variable(nprs(0).uniform(size=(10, 3)).astype(np.float32)),
            log_stds=Variable(nprs(1).uniform(
                size=(10, 3)).astype(np.float32)),
        ),
        new_dists=Gaussian(
            means=Variable(nprs(2).uniform(size=(10, 3)).astype(np.float32)),
            log_stds=Variable(nprs(3).uniform(
                size=(10, 3)).astype(np.float32)),
        ),
        all_acts=Variable(nprs(4).uniform(size=(10, 3)).astype(np.float32)),
        all_advs=Variable(nprs(5).uniform(size=(10,)).astype(np.float32)),
    ),
    desired_output=lambda: Variable(
        np.array(-0.5629823207855225, dtype=np.float32))
)

register_test(
    "trpo.compute_kl",
    kwargs=lambda: dict(
        old_dists=Gaussian(
            means=Variable(nprs(0).uniform(size=(10, 3)).astype(np.float32)),
            log_stds=Variable(nprs(1).uniform(
                size=(10, 3)).astype(np.float32)),
        ),
        new_dists=Gaussian(
            means=Variable(nprs(2).uniform(size=(10, 3)).astype(np.float32)),
            log_stds=Variable(nprs(3).uniform(
                size=(10, 3)).astype(np.float32)),
        ),
    ),
    desired_output=lambda: Variable(
        np.array(0.5306503176689148, dtype=np.float32))
)
