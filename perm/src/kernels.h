#ifndef KERNELS_H_
#define KERNELS_H_

#include "xla/ffi/api/ffi.h"
#include <cuda_runtime_api.h>

namespace ffi = xla::ffi;

ffi::Error fs_interferometer_host(cudaStream_t stream,
                                  ffi::Buffer<ffi::U64> cutoff,
                                  ffi::Buffer<ffi::U64> d,
                                  ffi::Buffer<ffi::C128> interferometer,
                                  ffi::ResultBuffer<ffi::C128> y);

ffi::Error fs_interferometer_fwd_host(
    cudaStream_t stream, ffi::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::U64> d,
    ffi::Buffer<ffi::C128> interferometer, ffi::ResultBuffer<ffi::C128> y,
    ffi::ResultBuffer<ffi::U64> y_dims, ffi::ResultBuffer<ffi::U32> helper_idx,
    ffi::ResultBuffer<ffi::F64> helper_sqrt);

ffi::Error calc_perm_bwd_host(cudaStream_t stream, ffi::Buffer<ffi::U64> cutoff,
                              ffi::Buffer<ffi::C128> interferometer,
                              ffi::Buffer<ffi::U64> d, ffi::Buffer<ffi::C128> y,
                              ffi::Buffer<ffi::U64> y_dim,
                              ffi::Buffer<ffi::U32> helper_idx,
                              ffi::Buffer<ffi::F64> helper_sqrt,
                              ffi::Buffer<ffi::C128> upstream_buff,
                              ffi::ResultBuffer<ffi::C128> result);

#endif // KERNELS_H_