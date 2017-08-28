from chainer import Variable

from simplepg.simple_utils import register_test, nprs
from utils import Gaussian
import numpy as np

register_test(
    "pg.compute_surr_loss",
    kwargs=lambda: dict(
        dists=Gaussian(
            means=Variable(nprs(0).uniform(size=(10, 3)).astype(np.float32)),
            log_stds=Variable(nprs(1).uniform(
                size=(10, 3)).astype(np.float32)),
        ),
        all_acts=Variable(nprs(2).uniform(size=(10, 3)).astype(np.float32)),
        all_advs=Variable(nprs(3).uniform(size=(10,)).astype(np.float32)),
    ),
    desired_output=lambda: Variable(
        np.array(1.9201269149780273, dtype=np.float32))
)
