#ifndef KERNELS_H_
#define KERNELS_H_

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime_api.h>

namespace ffi = xla::ffi;

ffi::Error calc_perm_fwd_host(cudaStream_t stream, ffi::Buffer<ffi::U64> cutoff,
                              ffi::Buffer<ffi::U64> d,
                              ffi::Buffer<ffi::C128> interferometer,
                              ffi::ResultBuffer<ffi::C128> y,
                              ffi::ResultBuffer<ffi::U64> y_dims,
                              ffi::ResultBuffer<ffi::U32> helper_idx,
                              ffi::ResultBuffer<ffi::F64> helper_sqrt);

ffi::Error calc_perm_bwd_host(
    cudaStream_t stream, ffi::Buffer<ffi::C128> interferometer,
    ffi::Buffer<ffi::C128> y, ffi::Buffer<ffi::U64> y_dim,
    ffi::Buffer<ffi::U32> helper_idx, ffi::Buffer<ffi::F64> helper_sqrt,
    ffi::Buffer<ffi::C128> upstream_buff, ffi::ResultBuffer<ffi::C128> result);

ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::C128> a,
                      ffi::Buffer<ffi::C128> b, ffi::ResultBuffer<ffi::C128> c,
                      ffi::ResultBuffer<ffi::C128> b_plus_1, size_t n);

ffi::Error FooBwdHost(cudaStream_t stream, ffi::Buffer<ffi::F32> c_grad,
                      ffi::Buffer<ffi::F32> a,
                      ffi::ResultBuffer<ffi::F32> b_plus_1,
                      ffi::ResultBuffer<ffi::F32> a_grad,
                      ffi::ResultBuffer<ffi::F32> b_grad, size_t n);

#endif // KERNELS_H_