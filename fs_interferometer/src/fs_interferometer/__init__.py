from __future__ import annotations
from ._core import (
    __doc__,
    __version__,
    calc_fs_interferometer,
    get_fock_space_basis,
    _get_interferometer_on_fock_space,
    registrations,
)

try:
    from fs_interferometer import gpu_ops
except ImportError:
    pass

from .interferometer import fs_interferometer_grad, calc_fs_interferometer, total_size, _get_interferometer_on_fock_space_fwd

__all__ = [
    "__doc__",
    "__version__",
    "calc_fs_interferometer",
    "_get_interferometer_on_fock_space_fwd",
    "total_size",
    "get_fock_space_basis",
    "fs_interferometer_grad",
    "registrations",
]
