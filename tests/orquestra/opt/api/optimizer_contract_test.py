################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
from typing import Callable

import numpy as np
import pytest

from orquestra.opt.api import CostFunction
from orquestra.opt.api.optimizer_test import (
    NESTED_OPTIMIZER_CONTRACTS,
    _validate_gradients_history_is_recorded_if_keep_history_is_true,
    _validate_optimizer_does_not_record_history_by_default,
    _validate_optimizer_does_not_record_history_if_keep_history_is_false,
    _validate_optimizer_records_history_if_keep_history_is_true,
)
from orquestra.opt.mock_objects import (
    MockNestedOptimizer,
    MockOptimizer,
    mock_cost_function,
)

# Because MockOptimizer is not a precise optimizer, it cannot be used to validate the
# contracts that test for accuracy of optimization result values.
CONTRACTS_THAT_DONT_TEST_ACCURACY = [
    _validate_optimizer_records_history_if_keep_history_is_true,
    _validate_gradients_history_is_recorded_if_keep_history_is_true,
    _validate_optimizer_does_not_record_history_if_keep_history_is_false,
    _validate_optimizer_does_not_record_history_by_default,
]


class MaliciousOptimizer(MockOptimizer):
    def _minimize(
        self, cost_function, initial_params: np.ndarray, keep_history: bool = False
    ):
        keep_history = not keep_history
        results = super()._minimize(
            cost_function, initial_params, keep_history=keep_history
        )
        del results["nit"]
        return results


_good_optimizer = MockOptimizer()
_malicious_optimizer = MaliciousOptimizer()


class TestOptimizerContracts:
    @pytest.mark.parametrize("contract", CONTRACTS_THAT_DONT_TEST_ACCURACY)
    def test_validates_contracts(self, contract):
        assert contract(_good_optimizer)
        assert not contract(_malicious_optimizer)


class MaliciousNestedOptimizer(MockNestedOptimizer):
    def _minimize(
        self,
        cost_function_factory: Callable[[int], CostFunction],
        initial_params: np.ndarray,
        keep_history: bool = False,
    ):
        keep_history = not keep_history
        results = super()._minimize(
            cost_function_factory, initial_params, keep_history=keep_history
        )
        del results["nit"]
        return results


_good_nested_optimizer = MockNestedOptimizer(inner_optimizer=MockOptimizer(), n_iters=5)
_malicious_nested_optimizer = MaliciousNestedOptimizer(
    inner_optimizer=MockOptimizer(), n_iters=5
)


def mock_cost_function_factory(iteration_id: int):
    def modified_cost_function(params):
        return mock_cost_function(params) ** iteration_id

    return modified_cost_function


class TestNestedOptimizerContracts:
    @pytest.mark.parametrize("contract", NESTED_OPTIMIZER_CONTRACTS)
    def test_validate_contracts(self, contract):
        assert contract(
            _good_nested_optimizer,
            mock_cost_function_factory,
            np.array([2]),
        )
        assert not contract(
            _malicious_nested_optimizer,
            mock_cost_function_factory,
            np.array([2]),
        )
