from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
import perm

jax.config.update("jax_enable_x64", True)

for name, target in perm.gpu_ops.foo().items():
    print(name, target)
    jax.ffi.register_ffi_target(name, target, platform="CUDA")

def foo_fwd(a, b):
  assert a.dtype == jnp.complex128
  assert a.shape == b.shape
  assert a.dtype == b.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  c, b_plus_1 = jax.ffi.ffi_call("foo_fwd", (out_type, out_type))(a, b, n=n)
  return c, (a, b_plus_1)


def foo_bwd(res, c_grad):
  a, b_plus_1 = res
  assert c_grad.dtype == jnp.float32
  assert c_grad.shape == a.shape
  assert a.shape == b_plus_1.shape
  assert c_grad.dtype == a.dtype
  assert a.dtype == b_plus_1.dtype
  n = np.prod(a.shape).astype(np.uint64)
  out_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
  return jax.ffi.ffi_call("foo_bwd", (out_type, out_type))(c_grad, a, b_plus_1,
                          n=n)


@jax.custom_vjp
def foo(a, b):
  c, _ = foo_fwd(a, b)
  return c


foo.defvjp(foo_fwd, foo_bwd)

a = jnp.array([[1+2j, 2+3j], [3+4j, 4+5j]], dtype=jnp.complex128)
b = jnp.array([[5+6j, 6+7j], [7+8j, 8+9j]], dtype=jnp.complex128)
c = foo(a, b)
print(c)