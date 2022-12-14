################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
"""Interfaces related to cost functions."""
from typing import Protocol, Union

import numpy as np

from .functions import (
    CallableStoringArtifacts,
    CallableWithGradient,
    CallableWithGradientStoringArtifacts,
)


class _CostFunction(Protocol):
    """Cost function transforming vectors from R^n to numbers or their estimates."""

    def __call__(self, parameters: np.ndarray) -> float:
        """Compute  value of the cost function for given parameters."""
        ...


CostFunction = Union[
    _CostFunction,
    CallableWithGradient,
    CallableStoringArtifacts,
    CallableWithGradientStoringArtifacts,
]


class ParameterPreprocessor(Protocol):
    """Parameter preprocessor.

    Implementer of this protocol should create new array instead of
    modifying passed parameters in place, which can have unpredictable
    side effects.
    """

    def __call__(self, parameters: np.ndarray) -> np.ndarray:
        """Preprocess parameters."""
        ...
