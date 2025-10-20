"""Minimal torch stub for environments without the real dependency."""

from __future__ import annotations

import builtins
import pickle
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

__SALIENT_STUB__ = True

_rng = random.Random(13)
builtins_any = builtins.any


def manual_seed(seed: int) -> None:
    _rng.seed(int(seed))


class Tensor:
    """Very small tensor object supporting the operations used in tests."""

    def __init__(self, data: Any, *, requires_grad: bool = False) -> None:
        self._data = _to_nested(data)
        self.requires_grad = bool(requires_grad)
        self.grad: Tensor | None = None
        self._backward: Callable[[float], None] | None = None
        self._shape = _infer_shape(self._data)

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    def item(self) -> Any:
        if self._shape:
            raise ValueError("Only scalar tensors can be converted to Python scalars")
        return self._data

    def tolist(self) -> Any:
        return deepcopy(self._data)

    def clone(self) -> "Tensor":
        return Tensor(self.tolist(), requires_grad=self.requires_grad)

    def detach(self) -> "Tensor":
        return Tensor(self.tolist(), requires_grad=False)

    def numpy(self) -> Any:  # pragma: no cover - compatibility hook
        return self.tolist()

    def to(self, *args: Any, **kwargs: Any) -> "Tensor":  # pragma: no cover
        return self

    def cpu(self) -> "Tensor":  # pragma: no cover
        return self

    # ------------------------------------------------------------------
    # Gradients
    # ------------------------------------------------------------------
    def sum(self, dim: int | None = None) -> "Tensor":
        if dim is not None:
            raise NotImplementedError("dim-specific sums are not implemented in the stub")
        total = _sum_nested(self._data)
        out = Tensor(total, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(grad: float | "Tensor" = 1.0) -> None:
                grad_value = grad.item() if isinstance(grad, Tensor) else float(grad)
                grad_data = _fill_like(self._data, grad_value)
                self._accumulate_grad(grad_data)

            out._backward = _backward

        return out

    def backward(self, gradient: float | "Tensor" | None = None) -> None:
        grad_value = gradient
        if grad_value is None:
            grad_value = 1.0
        if isinstance(grad_value, Tensor):
            grad_value = grad_value.item()
        if self._backward:
            self._backward(float(grad_value))
        elif self.requires_grad:
            grad_data = _fill_like(self._data, float(grad_value))
            self._accumulate_grad(grad_data)

    def _accumulate_grad(self, grad: Any) -> None:
        grad_data = grad.tolist() if isinstance(grad, Tensor) else grad
        if self.grad is None:
            self.grad = Tensor(grad_data)
        else:
            combined = _combine_nested(self.grad.tolist(), grad_data, lambda a, b: a + b)
            self.grad = Tensor(combined)

    def resize_rows(self, new_rows: int, fill_value: float = 0.0) -> None:
        if not self._shape:
            return
        current = self._shape[0]
        if new_rows <= current:
            return
        remainder_shape = self._shape[1:]
        filler = _build_full(remainder_shape, fill_value)
        for _ in range(new_rows - current):
            self._data.append(deepcopy(filler))
        self._shape = _infer_shape(self._data)

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterable:
        if not self._shape:
            return iter([self._data])
        return iter(self.tolist())

    def __len__(self) -> int:
        return self._shape[0] if self._shape else 1

    def __getitem__(self, index: Any) -> Any:
        result = _index_nested(self._data, index)
        if isinstance(result, list):
            return Tensor(result, requires_grad=self.requires_grad)
        return result

    def size(self, dim: int | None = None) -> tuple[int, ...] | int:
        if dim is None:
            return self._shape
        return self._shape[dim]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    # ------------------------------------------------------------------
    # Comparison helpers used in tests
    # ------------------------------------------------------------------
    def __eq__(self, other: Any) -> "Tensor":  # pragma: no cover - exercised indirectly
        rhs = other.tolist() if isinstance(other, Tensor) else other
        result = _combine_nested(self._data, rhs, lambda a, b: a == b)
        return Tensor(result)

    def __ne__(self, other: Any) -> "Tensor":  # pragma: no cover - exercised indirectly
        rhs = other.tolist() if isinstance(other, Tensor) else other
        result = _combine_nested(self._data, rhs, lambda a, b: a != b)
        return Tensor(result)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Tensor({self.tolist()}, requires_grad={self.requires_grad})"

    __hash__ = object.__hash__


TensorType = Tensor


def tensor(
    data: Any,
    *,
    dtype: Any = None,
    device: Any = None,
    requires_grad: bool | None = None,
) -> Tensor:
    return Tensor(data, requires_grad=bool(requires_grad) if requires_grad is not None else False)


def zeros(shape: Sequence[int], *, requires_grad: bool = False) -> Tensor:
    return Tensor(_build_full(tuple(int(dim) for dim in shape), 0.0), requires_grad=requires_grad)


def zeros_like(other: Tensor) -> Tensor:
    return Tensor(_build_full(other.shape, 0.0), requires_grad=other.requires_grad)


def randn(*shape: Any, requires_grad: bool = False, generator: Any = None) -> Tensor:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        dims = tuple(int(dim) for dim in shape[0])
    else:
        dims = tuple(int(dim) for dim in shape)

    def _sample() -> float:
        return _rng.gauss(0.0, 1.0)

    data = _build_with(dims, _sample)
    return Tensor(data, requires_grad=requires_grad)


def count_nonzero(value: Tensor) -> int:
    return sum(1 for item in _flatten(value.tolist()) if item != 0)


def any(value: Tensor) -> bool:  # type: ignore[override]
    return builtins_any(bool(item) for item in _flatten(value.tolist()))


def is_tensor(obj: Any) -> bool:
    return isinstance(obj, Tensor)


def equal(left: Tensor, right: Tensor) -> bool:
    return left.tolist() == right.tolist()


class _TestingNamespace:
    @staticmethod
    def assert_close(left: Any, right: Any, *, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        left_list = _as_list(left)
        right_list = _as_list(right)
        if not _allclose(left_list, right_list, rtol=rtol, atol=atol):
            raise AssertionError(f"Tensors are not close: {left_list!r} != {right_list!r}")


testing = _TestingNamespace()


class _OptimNamespace:  # pragma: no cover - compatibility shim
    class Optimizer:
        def __init__(self) -> None:
            self.state: dict[Any, dict[str, Any]] = {}


optim = _OptimNamespace()


def save(obj: Any, path: Path | str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(obj, handle)


def load(path: Path | str, map_location: Any = None) -> Any:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _to_nested(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_nested(item) for item in value]
    if isinstance(value, bool):
        return bool(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(value or 0.0)


def _infer_shape(data: Any) -> tuple[int, ...]:
    if isinstance(data, list):
        if not data:
            return (0,)
        return (len(data),) + _infer_shape(data[0])
    return ()


def _fill_like(template: Any, value: float) -> Any:
    if isinstance(template, list):
        return [_fill_like(child, value) for child in template]
    return value


def _build_full(shape: Sequence[int], value: float) -> Any:
    if not shape:
        return value
    dim = int(shape[0])
    return [_build_full(shape[1:], value) for _ in range(dim)]


def _build_with(shape: Sequence[int], sampler: Callable[[], float]) -> Any:
    if not shape:
        return sampler()
    dim = int(shape[0])
    return [_build_with(shape[1:], sampler) for _ in range(dim)]


def _sum_nested(data: Any) -> float:
    if isinstance(data, list):
        return sum(_sum_nested(item) for item in data)
    return float(data)


def _combine_nested(left: Any, right: Any, op: Callable[[float, float], Any]) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        return [_combine_nested(l, r, op) for l, r in zip(left, right)]
    if isinstance(left, list):
        return [_combine_nested(l, right, op) for l in left]
    if isinstance(right, list):
        return [_combine_nested(left, r, op) for r in right]
    return op(float(left), float(right))


def _index_nested(data: Any, index: Any) -> Any:
    if isinstance(index, tuple):
        current = data
        for idx in index:
            current = _index_nested(current, idx)
        return current
    if isinstance(index, slice):
        return deepcopy(data[index])
    if isinstance(data, list):
        return deepcopy(data[index])
    raise TypeError("Cannot index into scalar tensor")


def _flatten(data: Any) -> Iterable:
    if isinstance(data, list):
        for item in data:
            yield from _flatten(item)
    else:
        yield data


def _as_list(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_as_list(item) for item in value]
    return value


def _allclose(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _allclose(l, r, rtol=rtol, atol=atol) for l, r in zip(left, right)
        )
    if isinstance(left, list) or isinstance(right, list):
        return False
    if isinstance(left, bool) or isinstance(right, bool):
        return bool(left) == bool(right)
    diff = abs(float(left) - float(right))
    return diff <= atol + rtol * abs(float(right))


__all__ = [
    "Tensor",
    "TensorType",
    "tensor",
    "zeros",
    "zeros_like",
    "randn",
    "manual_seed",
    "count_nonzero",
    "any",
    "is_tensor",
    "equal",
    "testing",
    "optim",
    "save",
    "load",
]
