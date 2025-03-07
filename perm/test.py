import jax
import math
from functools import partial
import numpy as np
import perm
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
for name, target in perm.registrations().items():
    jax.ffi.register_ffi_target(name, target)

def total_size(interferometer, cutoff):
   d = len(interferometer)
   sum = 1 + pow(d,2)
   for i in range(2, cutoff):
      sum += pow(math.comb(d + i - 1 , i), 2)
   return sum

def total_helper_idx_size(interferometer, cutoff):
   d = len(interferometer)
   sum = 0
   for i in range(2, cutoff):
      sum += math.comb(d + i - 1 , i)
   return sum * (2+ d)

def total_helper_sqrt_size(interferometer, cutoff):
   d = len(interferometer)
   sum = 0
   for i in range(2, cutoff):
      sum += math.comb(d + i - 1 , i)
   return sum * (1 + d)

@partial(jax.custom_vjp)
def _get_interferometer_on_fock_space_xla( cutoff, interferometer):
  '''if cutoff.dtype != jnp.int:
    raise ValueError("Only the float32 dtype is implemented by rms_norm")
    '''
  call = jax.ffi.ffi_call(
    "_get_interferometer_on_fock_space_xla",
    (
    jax.ShapeDtypeStruct([total_size(interferometer, cutoff[0])], interferometer.dtype),
    jax.ShapeDtypeStruct([cutoff[0]], np.uint64)),
    vmap_method="broadcast_all",
  )
  res, dims = call(cutoff, interferometer)
  sliced_res = []
  start_idx = 0
  for d in dims:
     sliced_res.append(np.array(res[start_idx:(start_idx+d*d)]).reshape(d,d))
     start_idx += d*d
  print(sliced_res)
  return sliced_res

def _get_interferometer_on_fock_space_fwd( cutoff, intf):
  ts = total_size(intf, cutoff[0])
  call = jax.ffi.ffi_call(
    "_get_interferometer_on_fock_space_fwd",
    (
    jax.ShapeDtypeStruct([ts], intf.dtype),
    jax.ShapeDtypeStruct([cutoff[0]], np.uint64),
    jax.ShapeDtypeStruct([total_helper_idx_size(intf, cutoff[0]) ], np.uint64),
    jax.ShapeDtypeStruct([total_helper_sqrt_size(intf, cutoff[0])], np.float64)),
    vmap_method="broadcast_all",
  )
  res, dims, helper_idx, helper_sqrt = call(cutoff, intf)
  sliced_res = []
  start_idx = 0
  for d in dims:
     sliced_res.append(np.array(res[start_idx:(start_idx+d*d)]).reshape(d,d))
     start_idx += d*d
  return res, dims, helper_idx, helper_sqrt
def _get_interferometer_on_fock_space_bwd(res, dims, helper_idx, helper_sqrt):
  pass


_get_interferometer_on_fock_space_xla.defvjp(_get_interferometer_on_fock_space_fwd, _get_interferometer_on_fock_space_bwd)

def interferometer():
    return np.array(
        [
            [
                -0.11035524 + 0.43053175j,
                0.16672794 - 0.47819775j,
                0.01831264 - 0.07497556j,
                0.44214383 + 0.23308614j,
                0.49732734 - 0.20707815j,
            ],
            [
                -0.2226116 - 0.39735917j,
                0.30063104 - 0.55866103j,
                0.00329231 - 0.28009116j,
                0.23376629 - 0.22854039j,
                -0.43588261 + 0.12139052j,
            ],
            [
                0.24737463 - 0.24638855j,
                -0.31444021 - 0.28467995j,
                -0.16425914 - 0.33655956j,
                -0.26884137 + 0.68435951j,
                -0.07230938 - 0.10989756j,
            ],
            [
                0.34411568 + 0.00861778j,
                -0.26208576 - 0.04842002j,
                0.64971999 - 0.05882925j,
                0.29164646 + 0.0682574j,
                0.01229276 + 0.54314998j,
            ],
            [
                -0.59541546 - 0.01014536j,
                -0.28295784 + 0.10016806j,
                0.57604147 - 0.13381814j,
                -0.07509227 + 0.08557502j,
                -0.12237335 - 0.42143858j,
            ],
        ], dtype=np.complex128
    )
cutoff = np.array([3], dtype=np.uint64)
print(_get_interferometer_on_fock_space_fwd(cutoff, interferometer()))
