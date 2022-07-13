################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import Optional, Protocol, Tuple, runtime_checkable

import cvxpy as cp
import numpy as np
from scipy.optimize import LinearConstraint

from .api import Optimizer


def solve_qp_problem_for_psd_matrix(
    matrix: np.ndarray, symmetrize: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Solves a quadratic programming (QP) optimization problem. The matrix should
    be positive semi-definite. This implementation assumes that the domain of
    the solution are variables between 0 and 1. If matrix is not positive
    semi-definite, `solve_qp_problem_with_optimizer` method should be used.

    Notes:
        We are aware of the fact that in the specified domain the solution is
        always a vector of zeros, but decided to leave the implementation as it is
        in case the domain changes in future.

    Args:
        matrix: a matrix representing the problem.
        symmetrize: a flag indicating whether the matrix should be symmetrized.

    Returns:
        np.ndarray: vector representing solution to the problem.
        float: optimal value of the solution.
    """
    if symmetrize:
        matrix = (matrix + matrix.T) / 2

    if not is_matrix_positive_semidefinite(matrix):
        raise ValueError("Input matrix should be positive semi-definite.")

    size = matrix.shape[0]
    P = matrix
    G = np.vstack([np.eye(size), -np.eye(size)])
    h = np.hstack([np.ones(size), np.zeros(size)])

    x = cp.Variable(size)
    problem = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P)), [G @ x <= h])

    problem.solve()
    return x.value, problem.value


def solve_qp_problem_with_optimizer(
    matrix: np.ndarray,
    optimizer: Optimizer,
    number_of_trials: int = 1,
    symmetrize: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Solves a quadratic programming (QP) optimization problem.
    This implementation assumes that the domain of the solution are variables
    between 0 and 1.

    Args:
        matrix: a matrix representing the problem.
        optimizer: an optimizer to be used to solve the problem. Optimizer should
            support constraints.
        number_of_trials: specifies the number of times problem will be solved.
            Only the best solution will be returned.
        symmetrize: a flag indicating whether the matrix should be symmetrized.

    Returns:
        np.ndarray: vector representing solution to the problem
        float: optimal value of the solution
    """
    if symmetrize:
        matrix = (matrix + matrix.T) / 2

    # Use an optimizer to solve non-convex QP relaxations
    if not isinstance(optimizer, _WithConstraints):
        raise ValueError("Optimizer needs to support constraints.")
    size = matrix.shape[0]
    A = np.eye(size)
    lower_bound = np.zeros(size)
    upper_bound = np.ones(size)
    linear_constraint = LinearConstraint(A, lower_bound, upper_bound)

    optimizer.constraints = linear_constraint

    def cost_function(x):
        return x.T @ matrix @ x

    final_value = None
    final_params = None

    for _ in range(number_of_trials):
        initial_params = np.random.uniform(0.0, 1.0, size=size)
        optimization_results = optimizer.minimize(cost_function, initial_params)
        if final_value is None or optimization_results.opt_value < final_value:
            final_value = optimization_results.opt_value
            final_params = optimization_results.opt_params
    assert final_value is not None
    assert final_params is not None
    # We round the values to avoid having values like 1.0000000002 or -1e-14
    # in the output
    return np.around(final_params, decimals=8), final_value


def is_matrix_positive_semidefinite(matrix: np.ndarray) -> bool:
    """
    Checks whether matrix is positive semi-definite.

    Args:
        matrix: a matrix that should be checked.

    Returns:
        bool: True if matrix is positive semi-definite, False otherwise.

    """
    eigenvalues, _ = np.linalg.eig(matrix)
    return np.min(eigenvalues) >= 0


# Temporary solution until constraints added to optimizer.py in orquestra-opt
@runtime_checkable
class _WithConstraints(Protocol):
    """Minimum interface for Optimizers with constraints"""

    constraints: Optional[LinearConstraint]
