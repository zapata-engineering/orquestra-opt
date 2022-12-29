from typing import List, Sequence, Tuple, Union, Optional

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


def _get_n_evaluations_per_candidate(
    evaluation_budget: int, n_rounds: int, n_candidates_per_round: int
) -> List[int]:
    """
    Computes how many evaluations correspond to each candidate in the fairest possible way
    given an evaluation budget.

    Args:
        evaluation_budget: Total number of evaluations available for all the
            candidates.
        n_candidates: Number of candidates to distribute the evaluations over.

    Returns:
        A list of length n_candidates, where each element is the number of evaluations
        that should be allocated to the corresponding candidate.
    """
    total_candidates = (n_rounds - 1) * n_candidates_per_round + 1
    result = [
        int(x)
        for x in np.diff(
            np.round(np.linspace(0, evaluation_budget, total_candidates + 1))
        )
    ]
    if 0 in result:
        raise ValueError(
            f"Cannot allocate {evaluation_budget} evaluations to {total_candidates} candidates, "
            f"because at least one candidate would get 0 evaluations. Evaluations are {result}"
        )
    return result


def _get_tighter_bounds(
    candidate: np.ndarray,
    bounds: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    lower_bound, upper_bound = bounds
    dist_to_lower_bound = candidate - lower_bound
    dist_to_upper_bound = upper_bound - candidate
    bounds_scale = (upper_bound - lower_bound) / 2
    if isinstance(bounds_scale, float):
        bounds_scale = np.ones_like(candidate) * bounds_scale
    closest_distances = np.max(
        np.vstack((dist_to_lower_bound, dist_to_upper_bound)), axis=0
    )
    distances = np.min(np.vstack((closest_distances, bounds_scale)), axis=0)
    return candidate - distances, candidate + distances


class CostFunctionWithBestCandidates:
    def __init__(
        self,
        cost_function: CostFunction,
        n_candidates: int,
        bounds: Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]],
    ):
        self.cost_function = cost_function
        self.candidates: List[dict] = [{}] * n_candidates
        self.candidates_hashes = [0] * n_candidates
        self.bounds = bounds
        self.nfev = 0

    def __call__(self, parameters: np.ndarray) -> float:
        cost = self.cost_function(parameters)
        self.nfev += 1
        for i, candidate in enumerate(self.candidates):
            if not candidate or cost <= candidate["cost"]:
                parameters_hash = hash(parameters.tobytes())
                if parameters_hash in self.candidates_hashes:
                    break
                self.candidates.insert(
                    i,
                    {
                        "cost": cost,
                        "parameters": parameters,
                        "bounds": _get_tighter_bounds(parameters, self.bounds),
                    },
                )
                self.candidates.pop()
                self.candidates_hashes.insert(i, parameters_hash)
                self.candidates_hashes.pop()
                break
        return cost

    def gather_best_candidates(self) -> List[dict]:
        return [c for c in self.candidates if "parameters" in c.keys()]

    def set_bounds_for_candidate(
        self, candidate_index: int, bounds: Tuple[np.ndarray, np.ndarray]
    ):
        assert (
            "cost" in self.candidates[candidate_index]
            and "parameters" in self.candidates[candidate_index]
        ), "Candidate not initialized"
        self.candidates[candidate_index]["bounds"] = bounds


class TensorTrainOptimizer(Optimizer):
    def __init__(
        self,
        n_grid_points: Union[int, Sequence[int]],
        n_evaluations: int,
        bounds: Bounds,
        maximum_tensor_train_rank: int = 4,
        n_rounds: int = 1,
        maximum_number_of_candidates: int = 5,
        recorder: RecorderFactory = _recorder,
    ):
        super().__init__(recorder=recorder)
        self.n_grid_points = n_grid_points
        self.n_evaluations = n_evaluations
        self.bounds = _get_bounds_like_array(bounds)
        self.maximum_tensor_train_rank = maximum_tensor_train_rank
        self.n_rounds = n_rounds
        self.maximum_number_of_candidates = maximum_number_of_candidates
        self.candidates = [None] * maximum_number_of_candidates
        self.evaluations_per_candidate = _get_n_evaluations_per_candidate(
            n_evaluations, n_rounds, maximum_number_of_candidates
        )

    def _preprocess_cost_function(
        self, cost_function: CostFunction
    ) -> CostFunctionWithBestCandidates:
        return CostFunctionWithBestCandidates(
            cost_function, self.maximum_number_of_candidates, self.bounds
        )

    def _minimize(
        self,
        cost_function: CostFunctionWithBestCandidates,
        initial_params: np.ndarray,
        keep_history: bool = False,
    ) -> OptimizeResult:
        n_dim = initial_params.size
        evaluations_per_candidate_iterator = iter(self.evaluations_per_candidate)
        for i in range(self.n_rounds):
            best_candidates = cost_function.gather_best_candidates()
            if len(best_candidates) == 0:
                ttopt = TTOpt(
                    f=cost_function,
                    d=n_dim,
                    a=self.bounds[0],
                    b=self.bounds[1],
                    n=self.n_grid_points,
                    evals=next(evaluations_per_candidate_iterator),
                    is_vect=False,
                    is_func=True,
                )
                ttopt.minimize(self.maximum_tensor_train_rank)
            else:
                for candidate in best_candidates:
                    cost_function.bounds = candidate["bounds"]
                    ttopt = TTOpt(
                        f=cost_function,
                        d=n_dim,
                        a=candidate["bounds"][0],
                        b=candidate["bounds"][1],
                        n=self.n_grid_points,
                        evals=next(evaluations_per_candidate_iterator),
                        is_vect=False,
                        is_func=True,
                    )
                    ttopt.minimize(self.maximum_tensor_train_rank)
        best_candidate = cost_function.gather_best_candidates()[0]
        return optimization_result(
            opt_value=best_candidate["cost"],
            opt_params=best_candidate["parameters"],
            nit=None,
            nfev=cost_function.nfev,
            **construct_history_info(cost_function, keep_history),  # type: ignore
        )
