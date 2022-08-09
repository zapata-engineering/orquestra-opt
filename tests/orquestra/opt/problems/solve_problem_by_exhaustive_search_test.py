################################################################################
# © Copyright 2021-2022 Zapata Computing Inc.
################################################################################
import pytest
from orquestra.quantum.operators import PauliTerm

from orquestra.opt.problems import solve_problem_by_exhaustive_search

HAMILTONIAN_SOLUTION_COST_LIST = [
    (
        PauliTerm("1*Z0*Z1") + PauliTerm("I0", -1),
        [(0, 1), (1, 0)],
        -2,
    ),
    (
        PauliTerm("5*Z0*Z1")
        + PauliTerm("5*Z0*Z3")
        + PauliTerm("(0.5)*Z1*Z2")
        + PauliTerm("(0.5)*Z2*Z3")
        + PauliTerm("I0", -11),
        [(0, 1, 0, 1), (1, 0, 1, 0)],
        -22,
    ),
]


class TestSolveProblemByExhaustiveSearch:
    @pytest.mark.parametrize(
        "hamiltonian,target_solutions,target_value", [*HAMILTONIAN_SOLUTION_COST_LIST]
    )
    def test_solve_problem_by_exhaustive_search(
        self, hamiltonian, target_solutions, target_value
    ):
        value, solutions = solve_problem_by_exhaustive_search(hamiltonian)
        assert set(solutions) == set(target_solutions)
        assert value == target_value
