import pytest
import dimod
import numpy as np
from zquantum.core.interfaces.mock_objects import MockOptimizer
from zquantum.qubo.convex_opt import (
    solve_qp_problem_for_spd_matrix,
    solve_qp_problem_with_optimizer,
    is_matrix_semi_positive_definite,
)


@pytest.fixture
def optimizer():
    optimizer = MockOptimizer()
    optimizer.constraints = None
    return optimizer


def spd_matrix():
    return np.array([[5, 1, 2], [0, 6, 2], [0, 0, 7]])


def non_spd_matrix():
    return np.array([[-10, 1, 2], [0, -12, 2], [0, 0, -14]])


@pytest.mark.parametrize("matrix", [spd_matrix()])
def test_solve_qp_problem_for_spd_matrix(matrix):
    target_solution = np.array([0, 0, 0])
    solution, optimal_value = solve_qp_problem_for_spd_matrix(matrix)

    assert pytest.approx(optimal_value) == 0
    assert np.allclose(solution, target_solution)


@pytest.mark.parametrize("matrix", [non_spd_matrix()])
def test_solve_qp_problem_for_spd_matrix_fails_for_non_spd_matrix(matrix):
    with pytest.raises(ValueError):
        solution, optimal_value = solve_qp_problem_for_spd_matrix(matrix)


@pytest.mark.parametrize("matrix", [spd_matrix(), non_spd_matrix()])
def test_solve_qp_problem_with_optimizer(matrix, optimizer):
    solution, optimal_value = solve_qp_problem_with_optimizer(matrix, optimizer)

    assert isinstance(optimal_value, float)
    assert len(solution == 3)


@pytest.mark.parametrize("matrix", [spd_matrix()])
def test_solve_qp_problem_with_optimizer_throws_error_when_optimizer_does_not_support_constraints(
    matrix,
):
    optimizer = MockOptimizer()
    with pytest.raises(ValueError):
        solution, optimal_value = solve_qp_problem_with_optimizer(matrix, optimizer)


@pytest.mark.parametrize(
    "matrix,expected", [(spd_matrix(), True), (non_spd_matrix(), False)]
)
def test_is_matrix_semi_positive_definite(matrix, expected):
    assert is_matrix_semi_positive_definite(matrix) == expected