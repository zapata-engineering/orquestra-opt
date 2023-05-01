################################################################################
# © Copyright 2022 Zapata Computing Inc.
################################################################################
"""Main implementation of the recorder."""
import copy
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    overload,
    runtime_checkable,
)

import numpy as np

from ..api.functions import (
    CallableStoringArtifacts,
    CallableWithGradient,
    CallableWithGradientStoringArtifacts,
    StoreArtifact,
    has_store_artifact_param,
)
from ..api.save_conditions import SaveCondition, always

T = TypeVar("T", covariant=True)
S = TypeVar("S", contravariant=True)

NATIVE_RECORDER_ATTRIBUTES = (
    "predicate",
    "history",
    "target",
    "call_number",
    "gradient",
)


class ArtifactCollection(dict):
    """A dict with additional `forced` attribute.

    The `forced` flag is set whenever an artifact is forced into the dictionary
    despite current save_condition being false.
    """

    forced: bool = False


@dataclass
class HistoryEntry:
    """A history entry storing call number, parameters and target function value."""

    call_number: int
    params: Any
    value: Any


@dataclass
class HistoryEntryWithArtifacts(HistoryEntry):
    """A history entry enhanced with artifacts."""

    artifacts: Dict[str, Any]


def copy_recorder(recorder_to_copy):
    attributes_dict = {
        "target": recorder_to_copy.target,
        "save_condition": recorder_to_copy.predicate,
    }

    recorder_copy = type(recorder_to_copy)(**attributes_dict)
    recorder_copy.call_number = recorder_to_copy.call_number
    recorder_copy.history = recorder_to_copy.history
    return recorder_copy


def deepcopy_recorder(recorder_to_copy, memo):
    attributes_dict = {
        "target": copy.deepcopy(recorder_to_copy.target, memo=memo),
        "save_condition": copy.deepcopy(recorder_to_copy.predicate, memo=memo),
    }

    recorder_copy = type(recorder_to_copy)(**attributes_dict)
    recorder_copy.call_number = recorder_to_copy.call_number
    recorder_copy.history = copy.deepcopy(recorder_to_copy.history, memo=memo)
    return recorder_copy


@runtime_checkable
class SimpleRecorder(Protocol[S, T]):
    """Protocol representing recorder with basic functionalities.

    Simple recorders have target and history attributes. They forward calls made
    to them to the target.

    The target property is exposed so that clients have access to an unwrapped
    callable in case they need to make a call that is not recorder.

    Note that this recorder is structurally a base protocol for any other
    types of recorders. In other words  every recorder is at least a SimpleRecorder.
    """

    @property
    def target(self) -> Callable[[S], T]:
        pass

    def __call__(self, parameters: S) -> T:
        pass

    @property
    def history(self) -> Sequence[HistoryEntry]:
        pass


@runtime_checkable
class SimpleRecorderWithGradient(Protocol):
    """A protocol representing recorders wrapping functions with gradients.

    Aside the attributes of SimpleRecorder, this recorder also has a gradient
    attribute which itself is a recorder (and hence, has its own `history`
    attribute.

    Note that this protocol is not generic, because we only define gradients
    for functions from R^N to R (which translates to Callable[[ndarray], float]).
    """

    @property
    def target(self) -> Callable[[np.ndarray], float]:
        pass

    def __call__(self, parameters: np.ndarray) -> float:
        pass

    @property
    def gradient(self) -> SimpleRecorder[np.ndarray, np.ndarray]:
        pass

    @property
    def history(self) -> Sequence[HistoryEntry]:
        pass


class ArtifactRecorder(Protocol[S, T]):
    """A protocol representing recorders that can store artifacts.

    It is like simple recorder, but its target has to be a CallableStoringArtifacts.
    It also stores a list of HistoryEntryWithArtifact objects instead of usual
    HistoryEntry ones.
    """

    @property
    def target(self) -> CallableStoringArtifacts[S, T]:
        pass

    def __call__(self, parameters: S) -> T:
        pass

    @property
    def history(self) -> Sequence[HistoryEntryWithArtifacts]:
        pass


class ArtifactRecorderWithGradient(Protocol):
    """Protocol for recorders wrapping functions having gradient and storing artifacts.

    It is the most narrow type of recorder, combining both the
    SimpleReorderWithGradient and Artifact recorder. Hence, this recorder is
    non-generic and stores history entries with artifacts.
    """

    @property
    def target(self) -> CallableWithGradientStoringArtifacts:
        pass

    def __call__(self, parameters: np.ndarray) -> float:
        pass

    @property
    def gradient(self) -> SimpleRecorder[np.ndarray, np.ndarray]:
        pass

    @property
    def history(self) -> Sequence[HistoryEntryWithArtifacts]:
        pass


class SimpleRecorderImpl(Generic[S, T]):
    """A basic recorder that stores history entries.

    Args:
        target: a target function. Calls to the recorder will be propagated to this
          function.
        save_condition: a function determining whether given call should be saved
          to the history. See respective protocol for explanation of this parameter.
    """

    def __init__(self, target: Callable[[S], T], save_condition: SaveCondition):
        self.predicate = save_condition
        self.target = target
        self.history: List[HistoryEntry] = []
        self.call_number = 0

    def __call__(self, params: S) -> T:
        """Call the underlying target function, possibly saving call to the history.

        Args:
            params: argument to be passed to the target function.

        Returns:
            The value returned by the target function.
        """
        return_value = self.target(params)
        if self.predicate(return_value, params, self.call_number):
            self.history.append(
                HistoryEntry(self.call_number, copy.copy(params), return_value)
            )
        self.call_number += 1
        return return_value

    def __getattr__(self, item):
        return getattr(self.target, item)

    def __setattr__(self, key, value):
        if key in NATIVE_RECORDER_ATTRIBUTES:
            return object.__setattr__(self, key, value)
        return setattr(self.target, key, value)

    __copy__ = copy_recorder

    __deepcopy__ = deepcopy_recorder


class SimpleRecorderWithGradientImpl(SimpleRecorderImpl):
    """A recorder saving history entries that works with callables with gradient.

    Except having `gradient` attribute, this recorder is the same as `SimpleRecorder`.
    """

    def __init__(self, target: CallableWithGradient, save_condition: SaveCondition):
        super().__init__(target, save_condition)
        self.gradient = recorder(target.gradient, save_condition)


class ArtifactRecorderImpl(Generic[S, T]):
    """A recorder saving history entries with artifacts.

    Parameters to initializer are the same as for `SimpleRecorder`,
    except the target function should now be capable of storing artifacts.
    """

    def __init__(
        self, target: CallableStoringArtifacts[S, T], save_condition: SaveCondition
    ):
        self.predicate = save_condition
        self.target = target
        self.history: List[HistoryEntryWithArtifacts] = []
        self.call_number = 0

    def __call__(self, params: S) -> T:
        artifacts = ArtifactCollection()
        return_value = self.target(params, store_artifact=store_artifact(artifacts))

        if self.predicate(return_value, params, self.call_number) or artifacts.forced:
            self.history.append(
                HistoryEntryWithArtifacts(
                    self.call_number, copy.copy(params), return_value, artifacts
                )
            )
        self.call_number += 1
        return return_value

    def __getattr__(self, name):
        return getattr(self.target, name)

    def __setattr__(self, name, value):
        if name in ("predicate", "history", "target", "call_number", "gradient"):
            return object.__setattr__(self, name, value)
        return setattr(self.target, name, value)

    __copy__ = copy_recorder

    __deepcopy__ = deepcopy_recorder


class ArtifactRecorderWithGradientImpl(ArtifactRecorderImpl):
    """A recorder storing history entries with artifacts supporting callables with
    gradient.
    """

    def __init__(
        self,
        target: CallableWithGradientStoringArtifacts,
        save_condition: SaveCondition,
    ):
        super().__init__(target, save_condition)
        self.gradient = recorder(target.gradient, save_condition)


def store_artifact(artifacts) -> StoreArtifact:
    """Create a function storing artifacts in given artifacts collection.

    Args:
        artifacts: artifact collection.

    Returns:
        A function with signature:
        _store(artifact_name: str, artifact: Any, force: bool = False) -> None:
        This function is intended to be passed to functions that are capable of
        storing artifacts.
    """

    def _store(artifact_name: str, artifact: Any, force: bool = False) -> None:
        artifacts[artifact_name] = artifact
        if force:
            artifacts.forced = True

    return _store


AnyRecorder = Union[SimpleRecorder, ArtifactRecorder]
RecorderFactory = Callable[[Callable], AnyRecorder]
AnyHistory = Union[Sequence[HistoryEntry], Sequence[HistoryEntryWithArtifacts]]


@overload
def recorder(
    function: CallableWithGradientStoringArtifacts,
    save_condition: SaveCondition = always,
) -> ArtifactRecorderWithGradient:
    """The recorder function: variant for artifact-storing callables with gradient."""


@overload
def recorder(
    function: CallableStoringArtifacts[S, T], save_condition: SaveCondition = always
) -> ArtifactRecorder[S, T]:
    """The recorder function: variant for callables with no gradient that store
    artifacts."""


@overload
def recorder(
    function: CallableWithGradient, save_condition: SaveCondition = always
) -> SimpleRecorderWithGradient:
    """The recorder function: variant for callables with gradient that don't store
    artifacts."""


@overload
def recorder(
    function: Callable[[S], T], save_condition: SaveCondition = always
) -> SimpleRecorder[S, T]:
    """The recorder function: variant for callables without gradient that don't store
    artifacts."""


def recorder(function, save_condition: SaveCondition = always):
    """Create a recorder that is suitable for recording calls to given callable.

    Args:
        function: a callable to be recorded.
        save_condition: a condition on which the calls will be saved. See
          `SaveCondition` protocol for explanation of this parameter. By default
          all calls are saved.

    Returns:
        A callable object (the recorder) wrapping the `function`.
        The return type depends on the passed callable. See overloads defined
        above to check for available variants. Here is a summary:
        - recorder is always callable

        - if `function` has gradient, so does the recorder. Calls to gradient
          and calls made by gradient are NOT recorded.

        - if `function` has possibility to store artifacts (i.e. accepts
          `store_artifact` argument, then so does the recorder.
    """
    with_artifacts = has_store_artifact_param(function)
    with_gradient = isinstance(function, CallableWithGradient)

    if with_artifacts and with_gradient:
        return ArtifactRecorderWithGradientImpl(function, save_condition)
    elif with_artifacts:
        return ArtifactRecorderImpl(function, save_condition)
    elif with_gradient:
        return SimpleRecorderWithGradientImpl(function, save_condition)
    else:
        return SimpleRecorderImpl(function, save_condition)
