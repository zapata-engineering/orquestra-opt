from typing import Optional, Sequence, Union

import numpy as np
from scipy.optimize import OptimizeResult
from ttopt import TTOpt

from orquestra.opt.optimizers.pso.continuous_pso_optimizer import (  # TODO: where should these Bounds live?
    Bounds,
    _get_bounds_like_array,
)

from ..api import CostFunction, Optimizer, construct_history_info, optimization_result
from ..history.recorder import RecorderFactory
from ..history.recorder import recorder as _recorder


class TensorTrainOptimizer(Optimizer):
    def __init__(
        self,
        n_grid_points: Union[int, Sequence[int]],
        n_evaluations: Optional[int],
        bounds: Bounds,
        maximum_tensor_train_rank: int = 4,
        recorder: RecorderFactory = _recorder,
    ):
        super().__init__(recorder=recorder)
        self.n_grid_points = n_grid_points
        self.n_evaluations = n_evaluations
        self.bounds = _get_bounds_like_array(bounds)
        self.maximum_tensor_train_rank = maximum_tensor_train_rank

    def _minimize(
        self,
        cost_function: CostFunction,
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        n_dim = initial_params.size
        ttopt = TTOpt(
            f=cost_function,
            d=n_dim,
            a=self.bounds[0],
            b=self.bounds[1],
            n=self.n_grid_points,
            evals=self.n_evaluations,
            is_vect=False,
            is_func=True,
        )
        ttopt.minimize(self.maximum_tensor_train_rank)
        return optimization_result(
            opt_value=ttopt.y_min,
            opt_params=ttopt.x_min,
            nit=None,
            nfev=ttopt.k_total,
            **construct_history_info(cost_function, keep_history)  # type: ignore
        )
