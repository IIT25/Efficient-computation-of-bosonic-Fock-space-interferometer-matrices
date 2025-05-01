#include "cuda_helpers.cc.cu"
#include "kernels.h"
#include "matrix.hpp"
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cuComplex.h>
#include <cuda/std/complex>
#include <cuda_runtime_api.h>

__global__ void
fs_interferometer_kernel(unsigned long *cutoff_, unsigned long *d_,
                         cuda::std::complex<double> *interferometer_,
                         cuda::std::complex<double> *y)
{
  unsigned long cutoff = *cutoff_;
  unsigned long d = *d_;

  y[0] = cuda::std::complex<double>(1.0, 0.0);
  int result_size = 1;

  for (int i = 0; i < d * d; i++)
  {
    y[result_size] = interferometer_[i];
    result_size++;
  }

  int size = binomialCoeff(d + cutoff - 1, d);
  Matrix<int> space = Matrix<int>(size, d);
  int current_row = 0;
  for (size_t n = 0; n < cutoff; n++)
  {
    int num_rows = symmetric_subspace_cardinality(d, n);
    Matrix<int> out = space.rowsliceR(current_row, current_row + num_rows);
    partitions(d, n, out);
    current_row += num_rows;
  }
  Matrix<int> basis = Matrix<int>(space.rows, d);
  Matrix<int> first_subpace_indices_space = Matrix<int>(1, space.rows);
  Matrix<double> sqrt_first_occupation_numbers = Matrix<double>(1, space.rows);
  Matrix<int> first_nonzero_space_index = Matrix<int>(1, space.rows);
  Matrix<double> sqrt_space = Matrix<double>(space.rows, space.cols);
  for (size_t i = 0; i < space.rows; i++)
  {
    Matrix<int> current_basis = space.rowidx(i);
    current_basis.sqrt_indexed(&sqrt_space, i);
    bool found_first = false;
    for (size_t j = 0; j < d; j++)
    {
      current_basis[j] -= 1;
      basis(i, j) = get_index_in_fock_subspace(current_basis);
      if (!found_first && current_basis[j] >= 0)
      {
        first_nonzero_space_index[i] = j;
        first_subpace_indices_space[i] = basis(i, j);
        found_first = true;
        sqrt_first_occupation_numbers[i] = sqrt_space(i, j);
      }
      current_basis[j] += 1;
    }
  }
  Matrix<int> cutoffM = Matrix<int>(1, cutoff);
  cutoffM.iota(1);
  Matrix<int> indices = cutoff_fock_space_dim_array(cutoffM, d);
  Matrix<cuda::std::complex<double>> interferometer =
      Matrix<cuda::std::complex<double>>(d, d, interferometer_);
  Matrix<cuda::std::complex<double>> previous_representation = interferometer;
  for (size_t n = 2; n < cutoff; n++)
  {
    Matrix<int> subspace_range = Matrix<int>(1, indices[n] - indices[n - 1]);
    subspace_range.iota(indices[n - 1]);
    Matrix<int> subspace_indices =
        (*basis[subspace_range]).mod(indices[n - 1] - indices[n - 2]);
    Matrix<int> first_subspace_indices =
        (*first_subpace_indices_space[subspace_range]);
    Matrix<int> first_nonzero_indices =
        (*first_nonzero_space_index[subspace_range]);
    Matrix<double> sqrt_occupation_numbers = (*sqrt_space[subspace_range]);
    Matrix<double> sqrt_first_occupation_numbers_indexed =
        (*sqrt_first_occupation_numbers[subspace_range]);
    Matrix<cuda::std::complex<double>> representation =
        Matrix<cuda::std::complex<double>>(first_nonzero_indices.cols,
                                           sqrt_occupation_numbers.rows);
    for (size_t k = 0; k < first_nonzero_indices.cols; k++)
    {
      cuda::std::complex<double> denominator =
          sqrt_first_occupation_numbers_indexed[k];
      Matrix<cuda::std::complex<double>> previous_representation_indexed =
          previous_representation.rowidx(first_subspace_indices[k]);
      for (size_t j = 0; j < sqrt_occupation_numbers.cols; j++)
      {
        cuda::std::complex<double> one_particle_contrib =
            interferometer(first_nonzero_indices[k], j) / denominator;
        for (size_t i = 0; i < sqrt_occupation_numbers.rows; i++)
        {
          representation(k, i) +=
              one_particle_contrib * sqrt_occupation_numbers(i, j) *
              previous_representation_indexed[subspace_indices(i, j)];
        }
      }
    }
    for (int i = 0; i < representation.size(); i++)
    {
      y[result_size] = representation.data[i];
      result_size++;
    }
    previous_representation = representation;
  }
}

ffi::Error fs_interferometer_host(cudaStream_t stream,
                                  ffi::Buffer<ffi::U64> cutoff,
                                  ffi::Buffer<ffi::U64> d,
                                  ffi::Buffer<ffi::C128> interferometer,
                                  ffi::ResultBuffer<ffi::C128> y)
{
  const int block_dim = 1;
  const int grid_dim = 1;
  fs_interferometer_kernel<<<grid_dim, block_dim, /*shared_mem=*/0,
                             stream>>>(
      cutoff.typed_data(), d.typed_data(),
      reinterpret_cast<cuda::std::complex<double> *>(
          interferometer.typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(y->typed_data()));

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") +
                                cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

__global__ void fs_interferometer_fwd_kernel(
    unsigned long *cutoff_, unsigned long *d_,
    cuda::std::complex<double> *interferometer_, cuda::std::complex<double> *y,
    unsigned long *y_dims, int *helper_idx, double *helper_sqrt)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  // calc_helper
  unsigned long cutoff = *cutoff_;
  unsigned long d = *d_;

  int size = binomialCoeff(d + cutoff - 1, d);
  Matrix<int> space = Matrix<int>(size, d);
  int current_row = 0;
  for (size_t n = 0; n < cutoff; n++)
  {
    int num_rows = symmetric_subspace_cardinality(d, n);
    Matrix<int> out = space.rowsliceR(current_row, current_row + num_rows);
    int a = out(0, 0);
    partitions(d, n, out);
    current_row += num_rows;
  }

  Matrix<int> basis = Matrix<int>(space.rows, d);
  Matrix<int> first_subpace_indices_space = Matrix<int>(1, space.rows);
  Matrix<double> sqrt_first_occupation_numbers = Matrix<double>(1, space.rows);
  Matrix<int> first_nonzero_space_index = Matrix<int>(1, space.rows);
  Matrix<double> sqrt_space = Matrix<double>(space.rows, space.cols);

  for (size_t i = 0; i < space.rows; i++)
  {
    Matrix<int> current_basis = space.rowidx(i);
    current_basis.sqrt_indexed(&sqrt_space, i);
    bool found_first = false;
    for (size_t j = 0; j < d; j++)
    {
      current_basis[j] -= 1;
      basis(i, j) = get_index_in_fock_subspace(current_basis);
      if (!found_first && current_basis[j] >= 0)
      {
        first_nonzero_space_index[i] = j;
        first_subpace_indices_space[i] = basis(i, j);
        found_first = true;
        sqrt_first_occupation_numbers[i] = sqrt_space(i, j);
      }
      current_basis[j] += 1;
    }
  }
  // helper prep
  Matrix<int> cutoffM = Matrix<int>(1, cutoff);
  cutoffM.iota(1);
  Matrix<int> indices = cutoff_fock_space_dim_array(cutoffM, d);
  // interferometer prep
  Matrix<cuda::std::complex<double>> first =
      Matrix<cuda::std::complex<double>>(1, 1);

  first(0, 0) = cuda::std::complex<double>(1.0, 0.0);

  y_dims[0] = 1;
  y[0] = cuda::std::complex<double>(1.0, 0.0);
  int result_size = 1;
  int helper_idx_size = 0;
  int helper_sqrt_size = 0;
  y_dims[1] = d;
  Matrix<cuda::std::complex<double>> interferometer =
      Matrix<cuda::std::complex<double>>(d, d, interferometer_);
  for (int i = 0; i < interferometer.size(); i++)
  {
    y[result_size] = interferometer.data[i];
    result_size++;
  }
  Matrix<cuda::std::complex<double>> previous_representation = interferometer;
  for (size_t n = 2; n < cutoff; n++)
  {
    Matrix<int> subspace_range = Matrix<int>(1, indices[n] - indices[n - 1]);
    subspace_range.iota(indices[n - 1]);
    Matrix<int> subspace_indices =
        (*basis[subspace_range]).mod(indices[n - 1] - indices[n - 2]);
    Matrix<int> first_subspace_indices =
        (*first_subpace_indices_space[subspace_range]);
    Matrix<int> first_nonzero_indices =
        (*first_nonzero_space_index[subspace_range]);
    Matrix<double> sqrt_occupation_numbers = (*sqrt_space[subspace_range]);
    Matrix<double> sqrt_first_occupation_numbers_indexed =
        (*sqrt_first_occupation_numbers[subspace_range]);
    Matrix<cuda::std::complex<double>> representation =
        Matrix<cuda::std::complex<double>>(first_nonzero_indices.cols,
                                           sqrt_occupation_numbers.rows);
    y_dims[n] = first_nonzero_indices.cols;
    for (size_t k = 0; k < first_nonzero_indices.cols; k++)
    {
      cuda::std::complex<double> denominator =
          sqrt_first_occupation_numbers_indexed[k];
      Matrix<cuda::std::complex<double>> previous_representation_indexed =
          previous_representation.rowidx(first_subspace_indices[k]);
      for (size_t j = 0; j < sqrt_occupation_numbers.cols; j++)
      {
        cuda::std::complex<double> one_particle_contrib =
            interferometer(first_nonzero_indices[k], j) / denominator;
        for (size_t i = 0; i < sqrt_occupation_numbers.rows; i++)
        {
          representation(k, i) +=
              one_particle_contrib * sqrt_occupation_numbers(i, j) *
              previous_representation_indexed[subspace_indices(i, j)];
        }
      }
    }
    for (int i = 0; i < representation.size(); i++)
    {
      y[result_size] = representation.data[i];
      result_size++;
    }
    for (int i = 0; i < subspace_indices.size(); i++)
    {
      helper_idx[helper_idx_size] = subspace_indices[i];
      helper_idx_size++;
    }
    for (int i = 0; i < first_nonzero_indices.size(); i++)
    {
      helper_idx[helper_idx_size] = first_nonzero_indices[i];
      helper_idx_size++;
    }
    for (int i = 0; i < first_subspace_indices.size(); i++)
    {
      helper_idx[helper_idx_size] = first_subspace_indices[i];
      helper_idx_size++;
    }
    for (int i = 0; i < sqrt_occupation_numbers.size(); i++)
    {
      helper_sqrt[helper_sqrt_size] = sqrt_occupation_numbers[i];
      helper_sqrt_size++;
    }
    for (int i = 0; i < sqrt_first_occupation_numbers_indexed.size(); i++)
    {
      helper_sqrt[helper_sqrt_size] = sqrt_first_occupation_numbers_indexed[i];
      helper_sqrt_size++;
    }
    previous_representation = representation;
  }
}

ffi::Error fs_interferometer_fwd_host(
    cudaStream_t stream, ffi::Buffer<ffi::U64> cutoff, ffi::Buffer<ffi::U64> d,
    ffi::Buffer<ffi::C128> interferometer, ffi::ResultBuffer<ffi::C128> y,
    ffi::ResultBuffer<ffi::U64> y_dims, ffi::ResultBuffer<ffi::U32> helper_idx,
    ffi::ResultBuffer<ffi::F64> helper_sqrt)
{
  const int block_dim = 1;
  const int grid_dim = 1;
  cudaMemset(y_dims->typed_data(), 0, 7 * sizeof(unsigned long));
  fs_interferometer_fwd_kernel<<<grid_dim, block_dim, /*shared_mem=*/0,
                                 stream>>>(
      cutoff.typed_data(), d.typed_data(),
      reinterpret_cast<cuda::std::complex<double> *>(
          interferometer.typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(y->typed_data()),
      reinterpret_cast<unsigned long *>(y_dims->typed_data()),
      reinterpret_cast<int *>(helper_idx->typed_data()),
      reinterpret_cast<double *>(helper_sqrt->typed_data()));
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") +
                                cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

__global__ void calc_fs_interferometer_bwd_kernel(
    cuda::std::complex<double> *interferometer_, const int *cutoff_,
    const int *d_, cuda::std::complex<double> *y_, unsigned long *y_dims_,
    int *helper_idx_, double *helper_sqrt_,
    cuda::std::complex<double> *upstream_buff_,
    cuda::std::complex<double> *result)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  // unwrap buffers
  unsigned long cutoff = *cutoff_;
  unsigned long d = *d_;
  Matrix<cuda::std::complex<double>> interferometer =
      Matrix<cuda::std::complex<double>>(d, d, interferometer_);
  Matrix<unsigned long> y_dims = Matrix<unsigned long>(1, cutoff, y_dims_);
  Matrix<cuda::std::complex<double>> upstream =
      Matrix<cuda::std::complex<double>>(d, d, upstream_buff_);
  Matrix<cuda::std::complex<double>> full_kl_grad =
      Matrix<cuda::std::complex<double>>(interferometer.rows,
                                         interferometer.cols);
  for (int i = tid; i < d * d; i += grid_stride)
  {
    full_kl_grad.data[i] = cuda::std::complex<double>(0.0, 0.0);
  }
  __syncthreads();

  for (int row_index = 0; row_index < d; row_index++)
  {
    for (int col_index = 0; col_index < d; col_index++)
    {
      Matrix<cuda::std::complex<double>> second_subspace_grad =
          Matrix<cuda::std::complex<double>>(interferometer.rows,
                                             interferometer.cols);
      for (int i = tid; i < d * d; i += grid_stride)
      {
        second_subspace_grad[i] = cuda::std::complex<double>(0.0, 0.0);
      }
      __syncthreads();
      second_subspace_grad(row_index, col_index) = 1.0;
      Matrix<cuda::std::complex<double>> previous_subspace_grad =
          second_subspace_grad;
      int res_size = 1;
      int upstream_size = 1 + d * d;
      int helper_idx_size = 0;
      int helper_sqrt_size = 0;
      for (int p = 2; p < cutoff; p++)
      {
        Matrix<cuda::std::complex<double>> previous_subspace_representation =
            Matrix<cuda::std::complex<double>>(y_dims[p - 1], y_dims[p - 1],
                                               y_ + res_size);
        Matrix<cuda::std::complex<double>> upstream_p =
            Matrix<cuda::std::complex<double>>(y_dims[p], y_dims[p],
                                               upstream_buff_ + upstream_size);
        upstream_size += y_dims[p] * y_dims[p];
        res_size += y_dims[p - 1] * y_dims[p - 1];
        Matrix<int> subspace_indices = Matrix<int>(
            y_dims[p], interferometer.rows, helper_idx_ + helper_idx_size);
        helper_idx_size += y_dims[p] * interferometer.rows;
        Matrix<int> first_nonzero_indices =
            Matrix<int>(1, y_dims[p], helper_idx_ + helper_idx_size);
        helper_idx_size += y_dims[p];
        Matrix<int> first_subspace_indices =
            Matrix<int>(1, y_dims[p], helper_idx_ + helper_idx_size);
        helper_idx_size += y_dims[p];
        Matrix<double> sqrt_occupation_numbers = Matrix<double>(
            y_dims[p], interferometer.rows, helper_sqrt_ + helper_sqrt_size);
        helper_sqrt_size += y_dims[p] * interferometer.rows;
        Matrix<double> sqrt_first_occupation_numbers =
            Matrix<double>(1, y_dims[p], helper_sqrt_ + helper_sqrt_size);
        helper_sqrt_size += y_dims[p];
        Matrix<cuda::std::complex<double>> subspace_grad =
            _calculate_subspace_grad(
                row_index, col_index, previous_subspace_representation,
                subspace_indices, first_subspace_indices, first_nonzero_indices,
                sqrt_occupation_numbers, sqrt_first_occupation_numbers,
                interferometer, previous_subspace_grad);
        full_kl_grad(row_index, col_index) +=
            einsum_ij_ij(conj(subspace_grad), upstream_p);
        if (row_index == 0 && col_index == 0)
        {
          upstream_p.print_complex();
        }
        previous_subspace_grad = subspace_grad;
      }
    }
  }
  // full_kl_grad.print_complex();
  Matrix<cuda::std::complex<double>> upstream_1 =
      Matrix<cuda::std::complex<double>>(d, d, upstream_buff_ + 1);
  full_kl_grad.add(upstream_1);
  // convert result to buffer
  for (size_t i = 0; i < full_kl_grad.size(); i++)
  {
    result[i] = full_kl_grad[i];
  }
}
ffi::Error calc_fs_interferometer_bwd_host(
    cudaStream_t stream, ffi::Buffer<ffi::U64> cutoff,
    ffi::Buffer<ffi::C128> interferometer, ffi::Buffer<ffi::U64> d,
    ffi::Buffer<ffi::C128> y, ffi::Buffer<ffi::U64> y_dim,
    ffi::Buffer<ffi::U32> helper_idx, ffi::Buffer<ffi::F64> helper_sqrt,
    ffi::Buffer<ffi::C128> upstream_buff, ffi::ResultBuffer<ffi::C128> result)
{
  const int block_dim = 1;
  const int grid_dim = 1;
  calc_fs_interferometer_bwd_kernel<<<grid_dim, block_dim, /*shared_mem=*/0,
                                      stream>>>(
      reinterpret_cast<cuda::std::complex<double> *>(
          interferometer.typed_data()),
      reinterpret_cast<const int *>(cutoff.typed_data()),
      reinterpret_cast<const int *>(d.typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(y.typed_data()),
      reinterpret_cast<unsigned long *>(y_dim.typed_data()),
      reinterpret_cast<int *>(helper_idx.typed_data()),
      reinterpret_cast<double *>(helper_sqrt.typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(
          upstream_buff.typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(result->typed_data()));
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") +
                                cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

__global__ void
FooFwdKernel(const cuda::std::complex<double> *a,
             const cuda::std::complex<double> *b, cuda::std::complex<double> *c,
             cuda::std::complex<double> *b_plus_1, // intermediate output b+1
             size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < n; i += grid_stride)
  {
    b_plus_1[i] = b[i] + 1.0;
    c[i] = a[i] * b_plus_1[i];
  }
}

ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::C128> a,
                      ffi::Buffer<ffi::C128> b, ffi::ResultBuffer<ffi::C128> c,
                      ffi::ResultBuffer<ffi::C128> b_plus_1, size_t n)
{
  const int block_dim = 128;
  const int grid_dim = 1;

  std::vector<std::complex<double>> a_host(n);
  cudaMemcpy(a_host.data(), a.typed_data(), n * sizeof(std::complex<double>),
             cudaMemcpyDeviceToHost);

  FooFwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      reinterpret_cast<const cuda::std::complex<double> *>(a.typed_data()),
      reinterpret_cast<const cuda::std::complex<double> *>(b.typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(c->typed_data()),
      reinterpret_cast<cuda::std::complex<double> *>(b_plus_1->typed_data()),
      n);

  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") +
                                cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

__global__ void FooBwdKernel(const float *c_grad,   // incoming gradient wrt c
                             const float *a,        // original input a
                             const float *b_plus_1, // intermediate output b+1
                             float *a_grad,         // outgoing gradient wrt a
                             float *b_grad,         // outgoing gradient wrt b
                             size_t n)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride)
  {
    a_grad[i] = c_grad[i] * b_plus_1[i];
    b_grad[i] = c_grad[i] * a[i];
  }
}

ffi::Error FooBwdHost(cudaStream_t stream, ffi::Buffer<ffi::F32> c_grad,
                      ffi::Buffer<ffi::F32> a,
                      ffi::ResultBuffer<ffi::F32> b_plus_1,
                      ffi::ResultBuffer<ffi::F32> a_grad,
                      ffi::ResultBuffer<ffi::F32> b_grad, size_t n)
{
  const int block_dim = 128;
  const int grid_dim = 1;
  FooBwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      c_grad.typed_data(), a.typed_data(), b_plus_1->typed_data(),
      a_grad->typed_data(), b_grad->typed_data(), n);
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess)
  {
    return ffi::Error::Internal(std::string("CUDA error: ") +
                                cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}