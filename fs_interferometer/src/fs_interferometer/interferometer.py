import jax
import math
from functools import partial
import fs_interferometer
import jax.numpy as np


jax.config.update("jax_enable_x64", True)

for name, target in fs_interferometer.registrations().items():
    jax.ffi.register_ffi_target(name, target)

if hasattr(fs_interferometer, "gpu_ops"):
    try:
        for name, target in fs_interferometer.gpu_ops.foo().items():
            jax.ffi.register_ffi_target(name, target, platform="CUDA")
    except ImportError:
        print("GPU operations not available. Please install the GPU version of JAX.")


def total_size(interferometer, cutoff):
    d = len(interferometer)
    sum = 1 + pow(d, 2)
    for i in range(2, cutoff):
        sum += pow(math.comb(d + i - 1, i), 2)
    return sum


def total_helper_idx_size(interferometer, cutoff):
    d = len(interferometer)
    sum = 0
    for i in range(2, cutoff):
        sum += math.comb(d + i - 1, i)
    return sum * (2 + d)


def total_helper_sqrt_size(interferometer, cutoff):
    d = len(interferometer)
    sum = 0
    for i in range(2, cutoff):
        sum += math.comb(d + i - 1, i)
    return sum * (1 + d)


def _get_interferometer_on_fock_space_xla(cutoff, d, interferometer):
    if cutoff[0] < 2:
        raise Exception("Cutoff has to be at least 2", str(cutoff[0]))

    if interferometer.shape[0] != interferometer.shape[1]:
        raise Exception(
            "Interferometer has to be a square matrix",
            str(interferometer.shape[0]) + " x " + str(interferometer.shape[1]),
        )

    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name,
            (
                jax.ShapeDtypeStruct(
                    [total_size(interferometer, cutoff[0])], interferometer.dtype
                ),
            ),
            vmap_method="broadcast_all",
        )(cutoff, d, interferometer)

    return jax.lax.platform_dependent(
        cpu=impl("_get_interferometer_on_fock_space_xla"),
        cuda=impl("fs_interferometer"),
    )


def _get_interferometer_on_fock_space_fwd(cutoff, d, intf):
    ts = total_size(intf, cutoff[0])

    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name,
            (
                jax.ShapeDtypeStruct([ts], intf.dtype),
                jax.ShapeDtypeStruct([cutoff[0]], np.uint64),
                jax.ShapeDtypeStruct(
                    [total_helper_idx_size(intf, cutoff[0])], np.uint32
                ),
                jax.ShapeDtypeStruct(
                    [total_helper_sqrt_size(intf, cutoff[0])], np.float64
                ),
            ),
            vmap_method="broadcast_all",
        )(cutoff, d, intf)

    resj, dimsj, helper_idxj, helper_sqrtj = jax.lax.platform_dependent(
        cpu=impl("_get_interferometer_on_fock_space_fwd"),
        cuda=impl("fs_interferometer_fwd"),
    )
    return resj, dimsj, helper_idxj, helper_sqrtj


def _get_interferometer_on_fock_space_bwd(
    cutoff, interferometer, d, res, dims, helper_idx, helper_sqrt, upstream
):
    def impl(target_name):
        return lambda: jax.ffi.ffi_call(
            target_name,
            (jax.ShapeDtypeStruct(interferometer.shape, res.dtype)),
            vmap_method="broadcast_all",
        )(cutoff, interferometer, d, res, dims, helper_idx, helper_sqrt, upstream)

    return jax.lax.platform_dependent(
        cpu=impl("_get_interferometer_on_fock_space_bwd"),
        cuda=impl("fs_interferometer_bwd"),
    )


def fs_interferometer_grad(cutoff, d, interferometer, upstream):
    if cutoff[0] < 2:
        raise Exception("Cutoff has to be at least 2", str(cutoff[0]))
    if interferometer.shape[0] != interferometer.shape[1]:
        raise Exception(
            "Interferometer has to be a square matrix",
            str(interferometer.shape[0]) + " x " + str(interferometer.shape[1]),
        )
    if interferometer.shape[0] != d:
        raise Exception(
            "d has to be equal to the dimensions of interferometer", str(d[0])
        )
    resj, dimsj, helper_idxj, helper_sqrtj = _get_interferometer_on_fock_space_fwd(
        cutoff, d, interferometer
    )
    grad = _get_interferometer_on_fock_space_bwd(
        cutoff, interferometer, d, resj, dimsj, helper_idxj, helper_sqrtj, upstream
    )
    return grad
