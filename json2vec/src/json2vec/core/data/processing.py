import inspect
from collections.abc import Callable
from functools import partial
from typing import (
    Any,
    # Callable,
    Iterable,
    Iterator,
    TypeAlias,
    TypeVar,
    Union,
)

# import numba
import numpy as np
from beartype import beartype

from json2vec.core.structs.enums import Tokens

T = TypeVar("T")

RecursiveLike: TypeAlias = Union[T | None, list["RecursiveLike"]]


# FIXME
# /opt/venv/lib/python3.12/site-packages/numpy/_core/numeric.py:387: RuntimeWarning: invalid value encountered in cast multiarray.copyto(a, fill_value, casting='unsafe')

# FIXME renable this optimized version
@beartype
def pad(
    data: Any,
    shape: tuple[int, ...],
    dtype: type = np.int64,
    impute: Any = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pads nested lists into numpy arrays with flags, optimized with Numba.
    Optionally applies a tokenizer to non-None values.
    """

    values = np.full(shape, impute, dtype=dtype)
    flags = np.full(shape, Tokens.padded.value, dtype=np.int8)

    # compute strides
    strides = [1]
    for s in reversed(shape[1:]):
        strides.insert(0, strides[0] * s)
    strides = np.array(strides, dtype=np.int64)

    paths = []
    vals = []
    isnull = []

    def iter_path_values(node, path=[]):
        if len(path) >= len(shape):
            paths.append(tuple(path))
            if node is None:
                vals.append(impute)
                isnull.append(1)
            else:
                vals.append(node)
                isnull.append(0)
            return

        if not isinstance(node, list):
            paths.append(tuple(path))
            if node is None:
                vals.append(impute)
                isnull.append(1)
            else:
                vals.append(node)
                isnull.append(0)
            return

        max_len = shape[len(path)]
        for i, item in enumerate(node[:max_len]):
            iter_path_values(item, path + [i])

    iter_path_values(data)

    # convert to numpy
    paths = np.array(paths, dtype=np.int64)
    vals = np.array(vals, dtype=dtype)
    isnull = np.array(isnull, dtype=np.uint8)

    assign(values, flags, strides, paths, vals, isnull)

    return values, flags


# @numba.njit(parallel=False)
def assign(values, flags, strides, paths, vals, isnull):
    n = paths.shape[0]
    ndims = strides.shape[0]

    for idx in range(n):
        flat_idx = 0
        for d in range(ndims):
            flat_idx += paths[idx, d] * strides[d]

        if isnull[idx] == 1:
            flags.flat[flat_idx] = Tokens.null.value
        else:
            values.flat[flat_idx] = vals[idx]
            flags.flat[flat_idx] = Tokens.valued.value


def pad_nested(
    nested: Any, shape: tuple[int, ...], dtype: type | str = object, pad_value: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pads a nested list structure into numpy arrays of a fixed shape.

    Args:
        nested: The nested lists of arbitrary depth.
        shape: The target shape of the arrays.
        dtype: The dtype of the value array.
        pad_value: Value used to pad missing entries.

    Returns:
        values: ndarray with padded values.
        flags:  ndarray with flags (VALUE, PAD, NULL).
    """
    values = np.full(shape, pad_value, dtype=dtype)
    flags = np.full(shape, Tokens.padded.value, dtype=np.int32)

    # compute strides for flattening multidim index
    strides = [1]
    for s in reversed(shape[1:]):
        strides.insert(0, strides[0] * s)

    strides = tuple(strides)

    def iter_path_values(node, path=[]):
        if len(path) >= len(shape):
            yield path, node
            return

        if not isinstance(node, list):
            yield path, node
            return

        max_len = shape[len(path)]
        for i, item in enumerate(node[:max_len]):
            yield from iter_path_values(item, path + [i])

    for path, value in iter_path_values(nested):
        if len(path) != len(shape):
            continue  # skip incomplete paths

        flat_idx: int = sum(i * s for i, s in zip(path, strides))
        if value is None:
            flags.flat[flat_idx] = Tokens.null.value

        else:
            values.flat[flat_idx] = value
            flags.flat[flat_idx] = Tokens.valued.value

    return values, flags


T = TypeVar("T")

RecursiveLike: TypeAlias = Union[T | None, list["RecursiveLike"]]


# @beartype
# def mock(shape: tuple[int, ...], generator: Callable[[], T]) -> RecursiveLike:
#     flat = [generator() for _ in range(np.prod(shape))]
#     return np.array(flat, dtype=object).reshape(shape)


@beartype
class Pipeline:
    def __init__(self, **arguments):
        self.arguments: dict[str, Any] = arguments
        self.steps: list[Callable] = []

    def __or__(self, function: Callable) -> "Pipeline":
        required = [name for name in inspect.signature(function).parameters.keys()]

        available = set(required) & set(self.arguments.keys())

        self.steps.append(partial(function, **{arg: self.arguments[arg] for arg in available}))

        return self

    def __repr__(self):
        return f"Pipeline({repr(self.source)}, {repr(self.arguments)})"

    def __iter__(self):
        stream = self.steps[0]()

        for step in self.steps[1:]:
            stream = step(stream)

        return iter(stream)


def test_pipeline():
    def source() -> Iterator[int]:
        yield from range(5)

    def step1(pipe: Iterable[int]) -> Iterator[int]:
        yield from (x + 1 for x in pipe)

    def step2(pipe: Iterable[int], multiplier: int) -> Iterator[int]:
        yield from (x * multiplier for x in pipe)

    pipe = Pipeline(multiplier=2) | source | step1 | step2

    assert list(pipe) == [2, 4, 6, 8, 10]
