from simplepg.simple_utils import register_test, nprs
import numpy as np

register_test(
    "__main__.compute_update",
    kwargs=lambda: dict(
        discount=0.99,
        R_tplus1=1.0,
        theta=nprs(0).uniform(size=(2, 2)),
        s_t=nprs(1).uniform(size=(1,)),
        a_t=nprs(2).choice(2),
        r_t=nprs(3).uniform(),
        b_t=nprs(4).uniform(),
        get_grad_logp_action=lambda theta, *_: theta * 2
    ),
    desired_output=lambda: (
        1.5407979025745755,
        np.array([[0.62978332, 0.82070564], [0.69169275, 0.62527314]])
    )
)

register_test(
    "__main__.compute_baselines",
    kwargs=lambda: dict(
        all_returns=[
            nprs(0).uniform(size=(10,)),
            nprs(1).uniform(size=(20,)),
            [],
        ],
    ),
    desired_output=lambda: np.array([0.61576628, 0.36728075, 0.])
)

register_test(
    "__main__.compute_fisher_matrix",
    kwargs=lambda: dict(
        theta=nprs(1).uniform(size=(2, 2)),
        get_grad_logp_action=lambda theta, ob, action: np.exp(
            theta) * np.linalg.norm(action),
        all_observations=list(nprs(2).uniform(size=(5, 1))),
        all_actions=list(nprs(3).choice(2, size=(5,))),
    ),
    desired_output=lambda: np.array([[0.92104469, 1.24739299, 0.60704379, 0.82124306],
                                     [1.24739299, 1.68937435,
                                         0.82213401, 1.11222925],
                                     [0.60704379, 0.82213401,
                                         0.40009151, 0.54126635],
                                     [0.82124306, 1.11222925, 0.54126635, 0.73225564]])
)

register_test(
    "__main__.compute_natural_gradient",
    kwargs=lambda: dict(
        F=nprs(0).uniform(size=(4, 4)),
        grad=nprs(1).uniform(size=(2, 2)),
        reg=1e-3,
    ),
    desired_output=lambda: np.array(
        [[-0.44691565, 0.5477328], [-0.20366472, 0.72267091]])
)

register_test(
    "__main__.compute_step_size",
    kwargs=lambda: dict(
        F=nprs(0).uniform(size=(2, 2)),
        natural_grad=nprs(1).uniform(size=(1, 2)),
        natural_step_size=1e-2,
    ),
    desired_output=lambda: 0.1607407366467048,
)
