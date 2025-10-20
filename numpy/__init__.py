"""Lightweight numpy compatibility layer for environments without NumPy.

This module implements a tiny subset of NumPy used by the Salience OS seed
project and its unit tests.  The goal is to provide deterministic, pure-Python
behaviour that mimics the required NumPy API surface well enough for testing
without pulling in the heavy native dependency.

The implementation intentionally focuses on 1D and 2D float arrays.  It does
not attempt to be a drop-in replacement for the full NumPy project.  Only the
operations exercised by the tests are covered.
"""

from __future__ import annotations

import builtins
import math
import random as _py_random
import statistics
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence, Tuple, Union

NumberLike = Union[int, float]

float32 = float
float64 = float
float = float
integer = int
floating = float
bool_ = bool
newaxis = None


def _coerce_scalar(value: NumberLike, dtype=float) -> float:
    try:
        return dtype(value) if dtype is not None else float(value)
    except Exception:
        return float(value)


@dataclass
class ndarray:
    """Very small array wrapper supporting the operations we need."""

    _data: List
    dtype: type = float

    def __post_init__(self) -> None:
        self._normalise()

    def _normalise(self) -> None:
        def _convert(node):
            if isinstance(node, ndarray):
                return node.tolist()
            if isinstance(node, (list, tuple)):
                return [_convert(child) for child in node]
            return _coerce_scalar(node, self.dtype)

        self._data = _convert(self._data)
        self._shape = _infer_shape(self._data)

    # ------------------------------------------------------------------
    # Basic container protocol
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator:
        if self.ndim == 1:
            return iter(self._data)
        return (ndarray(row, dtype=self.dtype) if isinstance(row, list) else row for row in self._data)

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 1

    def __getitem__(self, index):
        if isinstance(index, tuple):
            data = self._data
            for idx in index:
                if idx is None:
                    data = [data]
                else:
                    data = _slice_data(data, idx)
            return _wrap_result(data, self.dtype)
        data = _slice_data(self._data, index)
        return _wrap_result(data, self.dtype)

    def __setitem__(self, index, value) -> None:
        def _assign(target, idx, val):
            if isinstance(idx, tuple):
                if not idx:
                    return
                head, *rest = idx
                if head is None:
                    if not isinstance(target, list) or not target:
                        target[:] = [target]
                    _assign(target[0], tuple(rest), val)
                    return
                target[head] = _assign(target[head], tuple(rest), val) if rest else _coerce_scalar(val, self.dtype)
                return target
            if isinstance(idx, slice):
                if isinstance(val, ndarray):
                    val = val.tolist()
                target[idx] = val
                return target
            target[idx] = _coerce_scalar(val, self.dtype) if not isinstance(val, list) else val
            return target

        _assign(self._data, index if isinstance(index, tuple) else (index,), value)
        self._shape = _infer_shape(self._data)

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------
    def _binary_op(self, other, op):
        return ndarray(_broadcast_op(self._data, other, op), dtype=self.dtype)

    def __add__(self, other):
        return self._binary_op(other, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, other):
        return self._binary_op(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return ndarray(_broadcast_op(other, self._data, lambda a, b: a - b), dtype=self.dtype)

    def __mul__(self, other):
        return self._binary_op(other, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._binary_op(other, lambda a, b: a / b)

    def __rtruediv__(self, other):
        return ndarray(_broadcast_op(other, self._data, lambda a, b: a / b), dtype=self.dtype)

    def __itruediv__(self, other):
        result = self.__truediv__(other)
        self._data = result.tolist()
        return self

    def __neg__(self):
        return ndarray(_broadcast_op(self._data, 0.0, lambda a, _: -a), dtype=self.dtype)

    # ------------------------------------------------------------------
    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def size(self) -> int:
        return sum(1 for _ in _iter_flat(self))

    def astype(self, dtype) -> "ndarray":
        return ndarray(self.tolist(), dtype=dtype)

    def tolist(self):
        if self.ndim == 0:
            return self._data
        return [row.tolist() if isinstance(row, ndarray) else row for row in _wrap_rows(self._data, self.dtype)]

    def copy(self) -> "ndarray":
        return ndarray(self.tolist(), dtype=self.dtype)

    def max(self):
        return builtins_max(_iter_flat(self))

    def min(self):
        return builtins_min(_iter_flat(self))

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"ndarray({self.tolist()}, dtype={self.dtype.__name__})"


def _wrap_rows(data, dtype):
    if isinstance(data, list) and data and isinstance(data[0], list):
        return [ndarray(row, dtype=dtype) for row in data]
    return data


def _wrap_result(data, dtype):
    if isinstance(data, list):
        return ndarray(data, dtype=dtype)
    return _coerce_scalar(data, dtype)


def _slice_data(data, index):
    if isinstance(index, slice):
        return data[index]
    return data[index]


def _infer_shape(data) -> Tuple[int, ...]:
    if isinstance(data, list):
        if not data:
            return (0,)
        sub_shape = _infer_shape(data[0])
        return (len(data),) + sub_shape
    return ()


def _iter_flat(data) -> Iterator[float]:
    if isinstance(data, ndarray):
        data = data._data
    if isinstance(data, list):
        for item in data:
            yield from _iter_flat(item)
    elif isinstance(data, (tuple, set)):
        for item in data:
            yield from _iter_flat(item)
    else:
        try:
            iterator = iter(data)  # type: ignore[arg-type]
        except TypeError:
            yield float(data)
        else:
            for item in iterator:
                yield from _iter_flat(item)


def _broadcast_op(left, right, op):
    if isinstance(left, ndarray):
        left = left._data
    if isinstance(right, ndarray):
        right = right._data
    if isinstance(left, list) and isinstance(right, list):
        return [_broadcast_op(l_item, r_item, op) for l_item, r_item in zip(left, right)]
    if isinstance(left, list):
        return [_broadcast_op(item, right, op) for item in left]
    if isinstance(right, list):
        return [_broadcast_op(left, item, op) for item in right]
    return op(float(left), float(right))


# ----------------------------------------------------------------------
# Array construction helpers
# ----------------------------------------------------------------------

def array(data, dtype=float) -> ndarray:
    return ndarray(list(data) if isinstance(data, tuple) else data, dtype=dtype)


def asarray(data, dtype=None) -> ndarray:
    return array(data, dtype=dtype or float)


def fromiter(iterable: Iterable, dtype=float) -> ndarray:
    return ndarray(list(iterable), dtype=dtype)


def zeros(shape, dtype=float) -> ndarray:
    return full(shape, 0.0, dtype=dtype)


def ones_like(arr: ndarray, dtype=None) -> ndarray:
    dtype = dtype or arr.dtype
    return full(arr.shape, 1.0, dtype=dtype)


def full(shape, fill_value, dtype=float) -> ndarray:
    if isinstance(shape, int):
        data = [_coerce_scalar(fill_value, dtype) for _ in range(int(shape))]
        return ndarray(data, dtype=dtype)
    dims = [int(dim) for dim in shape]
    if not dims:
        return ndarray(_coerce_scalar(fill_value, dtype), dtype=dtype)

    def _build(depth: int) -> List:
        if depth == len(dims) - 1:
            return [_coerce_scalar(fill_value, dtype) for _ in range(dims[depth])]
        return [_build(depth + 1) for _ in range(dims[depth])]

    return ndarray(_build(0), dtype=dtype)


def eye(n: int, dtype=float) -> ndarray:
    rows = []
    for i in range(n):
        row = [_coerce_scalar(1.0 if i == j else 0.0, dtype) for j in range(n)]
        rows.append(row)
    return ndarray(rows, dtype=dtype)


def vstack(arrays: Sequence[ndarray]) -> ndarray:
    stacked: List[List[float]] = []
    for arr in arrays:
        coerced = asarray(arr).tolist()
        if not isinstance(coerced, list):
            coerced = [coerced]
        stacked.append(coerced)
    flat: List[List[float]] = []
    for block in stacked:
        if block and isinstance(block[0], list):
            flat.extend(block)
        else:
            flat.append(block)
    return ndarray(flat)


def clip(value, min_value, max_value):
    if isinstance(value, ndarray):
        return ndarray(
            _broadcast_op(
                value,
                0.0,
                lambda a, _: builtins_min(builtins_max(a, min_value), max_value),
            ),
            dtype=value.dtype,
        )
    return builtins_min(builtins_max(float(value), min_value), max_value)


def tanh(value):
    if isinstance(value, ndarray):
        return ndarray(_broadcast_op(value, 0.0, lambda a, _: math.tanh(a)))
    return math.tanh(float(value))


def dot(left, right) -> float:
    left_iter = list(_iter_flat(left))
    right_iter = list(_iter_flat(right))
    return sum(a * b for a, b in zip(left_iter, right_iter))


class _LinalgModule:
    @staticmethod
    def norm(arr) -> float:
        return math.sqrt(builtins_sum(x * x for x in _iter_flat(arr)))

    @staticmethod
    def solve(matrix, values):
        rows = [list(_iter_flat(row)) for row in matrix]
        if len(rows) != 2 or len(rows[0]) != 2 or len(rows[1]) != 2:
            raise NotImplementedError("Only 2x2 systems supported in stub")
        b = list(_iter_flat(values))
        det = rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0]
        if abs(det) < 1e-12:
            raise ValueError("Singular matrix")
        x0 = (b[0] * rows[1][1] - rows[0][1] * b[1]) / det
        x1 = (rows[0][0] * b[1] - b[0] * rows[1][0]) / det
        return ndarray([x0, x1])


linalg = _LinalgModule()


def mean(arr) -> float:
    values = list(_iter_flat(arr))
    return builtins_sum(values) / len(values) if values else 0.0


def std(arr) -> float:
    values = list(_iter_flat(arr))
    if not values:
        return 0.0
    mu = mean(values)
    variance = builtins_sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def median(arr) -> float:
    values = list(_iter_flat(arr))
    return statistics.median(values) if values else 0.0


def abs(arr):  # type: ignore[override]
    if isinstance(arr, ndarray):
        return ndarray(_broadcast_op(arr, 0.0, lambda a, _: math.fabs(a)), dtype=arr.dtype)
    return math.fabs(float(arr))


def exp(arr):
    if isinstance(arr, ndarray):
        return ndarray(_broadcast_op(arr, 0.0, lambda a, _: math.exp(a)))
    return math.exp(float(arr))


def log(arr):
    if isinstance(arr, ndarray):
        return ndarray(_broadcast_op(arr, 0.0, lambda a, _: math.log(a)))
    return math.log(float(arr))


def sum(arr):  # type: ignore[override]
    return builtins_sum(_iter_flat(arr))


def max(arr):  # type: ignore[override]
    return builtins_max(_iter_flat(arr))


def min(arr):  # type: ignore[override]
    return builtins_min(_iter_flat(arr))


def histogram(data, bins=10, range=None):
    values = list(_iter_flat(data))
    if not values:
        return ([0] * bins, [])
    if range is None:
        lo, hi = builtins_min(values), builtins_max(values)
    else:
        lo, hi = range
    if hi <= lo:
        hi = lo + 1e-6
    bin_width = (hi - lo) / bins
    edges = [lo + i * bin_width for i in builtins_range(bins + 1)]
    counts = [0] * bins
    for value in values:
        index = int((value - lo) / bin_width)
        if index >= bins:
            index = bins - 1
        if index < 0:
            index = 0
        counts[index] += 1
    return counts, edges


def isfinite(arr):
    return [math.isfinite(x) for x in _iter_flat(arr)]


def isscalar(value) -> bool:
    return not isinstance(value, (ndarray, list, tuple, dict, set))


class _DefaultRNG:
    def __init__(self, seed=None):
        self._rng = _py_random.Random(seed)

    def normal(self, loc=0.0, scale=1.0, size=None):
        def _sample() -> float:
            # Box-Muller transform with deterministic pairing.
            u1 = builtins_max(self._rng.random(), 1e-6)
            u2 = self._rng.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            return loc + scale * z0

        if size is None:
            return _sample()
        if isinstance(size, tuple):
            total = 1
            for dim in size:
                total *= int(dim)
            samples = [_sample() for _ in range(total)]
            # reshape recursively
            def _reshape_list(values, dims):
                if not dims:
                    return values.pop(0) if values else 0.0
                dim = int(dims[0])
                if dim <= 0:
                    return []
                return [_reshape_list(values, dims[1:]) for _ in range(dim)]

            structured = _reshape_list(samples[:], list(size))
            return ndarray(structured).astype(float32)
        return ndarray([_sample() for _ in range(int(size))], dtype=float32)

    def shuffle(self, seq):
        self._rng.shuffle(seq)


class _RandomModule:
    @staticmethod
    def default_rng(seed=None):
        return _DefaultRNG(seed)

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None):
        rng = _DefaultRNG()
        return rng.normal(loc=loc, scale=scale, size=size)


random = _RandomModule()


# Preserve built-in functions for overrides above
builtins_sum = builtins.sum
builtins_max = builtins.max
builtins_min = builtins.min
builtins_range = builtins.range

__all__ = [
    "array",
    "asarray",
    "fromiter",
    "zeros",
    "full",
    "ones_like",
    "eye",
    "vstack",
    "clip",
    "tanh",
    "dot",
    "linalg",
    "mean",
    "std",
    "median",
    "abs",
    "exp",
    "log",
    "sum",
    "max",
    "min",
    "histogram",
    "isfinite",
    "isscalar",
    "random",
    "ndarray",
    "float32",
    "float64",
    "float",
    "integer",
    "floating",
    "bool_",
    "newaxis",
]
