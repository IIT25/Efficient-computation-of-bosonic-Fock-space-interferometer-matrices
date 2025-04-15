#ifndef MATRIX_H
#define MATRIX_H

#include "math.h"
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifndef DEBUG
#include <iostream>
#endif

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

/**
 * The class for storing matrices.
 *
 * @param rows The number of rows.
 * @param cols The number of columns.
 * @param data The optional input data. If not provided, new data is allocated.
 */
template <typename Tscalar> class Matrix {
public:
  size_t rows;
  size_t cols;
  size_t stride; // Column stride.
  Tscalar *data;
  bool owner;       // True if data is owned by the instance, otherwise false.
  size_t *refcount; // Number of references

  HOST_DEVICE Matrix(size_t rows, size_t cols)
      : rows(rows), cols(cols), stride(cols) {
    data = new Tscalar[rows * cols];
    owner = true;
    refcount = new size_t;
    (*refcount) = 1;
  }

  HOST_DEVICE Matrix(size_t rows, size_t cols, Tscalar *data)
      : rows(rows), cols(cols), stride(cols), data(data) {
    owner = false;
    refcount = new size_t;
    (*refcount) = 1;
  }

  HOST_DEVICE Matrix(const Matrix &matrix)
      : rows(matrix.rows), cols(matrix.cols), stride(matrix.stride),
        data(matrix.data), owner(matrix.owner), refcount(matrix.refcount) {
    (*refcount)++;
  }

  HOST_DEVICE Matrix()
      : rows(0), cols(0), stride(0), data(nullptr), owner(false),
        refcount(nullptr) {}

  HOST_DEVICE void operator=(const Matrix &matrix) {
    rows = matrix.rows;
    cols = matrix.cols;
    stride = matrix.stride;
    data = matrix.data;
    owner = matrix.owner;
    refcount = matrix.refcount;

    (*refcount)++;
  }

  HOST_DEVICE size_t size() { return rows * cols; }
  HOST_DEVICE Matrix copy() {
    Matrix matrix_copy(rows, cols);

    memcpy(matrix_copy.data, data, size() * sizeof(Tscalar));

    return matrix_copy;
  }

  HOST_DEVICE ~Matrix() {
    bool call_delete = ((*refcount) == 1);

    if (call_delete)
      delete refcount;
    else
      (*refcount)--;

    if (call_delete && owner)
      delete[] data;
  }
  void printdim() {
    std::cout << std::endl
              << "The stored matrix (" << rows << " x " << cols
              << "):" << std::endl;
  }
  HOST_DEVICE void print_complex() {
    printf("The stored matrix (%zu x %zu):\n", rows, cols);
    /*std::cout << std::endl
              << "The stored matrix (" << rows << " x " << cols
              << "):" << std::endl;*/
    for (size_t row_idx = 0; row_idx < rows; row_idx++) {
      for (size_t col_idx = 0; col_idx < cols; col_idx++) {
        size_t element_idx = row_idx * stride + col_idx;
        printf(" ( %f, %f)", data[element_idx].real(),
               data[element_idx].imag());
        // std::cout << " " << data[element_idx];
      }
      printf("\n");
      // std::cout << std::endl;
    }
    printf("----------------\n");
    // std::cout << "------------------------" << std::endl;
  }
  HOST_DEVICE void print_int() {
    printf("The stored matrix (%lu x %lu):\n", rows, cols);
    /*std::cout << std::endl
              << "The stored matrix (" << rows << " x " << cols
              << "):" << std::endl;*/
    for (size_t row_idx = 0; row_idx < rows; row_idx++) {
      for (size_t col_idx = 0; col_idx < cols; col_idx++) {
        size_t element_idx = row_idx * stride + col_idx;
        printf(" %d", data[element_idx]);
        // std::cout << " " << data[element_idx];
      }
      printf("\n");
      // std::cout << std::endl;
    }
    printf("----------------\n");
    // std::cout << "------------------------" << std::endl;
  }
  HOST_DEVICE void print_double() {
    printf("The stored matrix (%lu x %lu):\n", rows, cols);
    /*std::cout << std::endl
              << "The stored matrix (" << rows << " x " << cols
              << "):" << std::endl;*/
    for (size_t row_idx = 0; row_idx < rows; row_idx++) {
      for (size_t col_idx = 0; col_idx < cols; col_idx++) {
        size_t element_idx = row_idx * stride + col_idx;
        printf(" %f", data[element_idx]);
        // std::cout << " " << data[element_idx];
      }
      printf("\n");
      // std::cout << std::endl;
    }
    printf("----------------\n");
    // std::cout << "------------------------" << std::endl;
  }
  HOST_DEVICE Matrix<Tscalar> rowidx(size_t idx) { //
    Matrix<Tscalar> matrix_copy(1, cols);
    /*if (idx >= rows) {
      std::cout << "rowindx operator  cols:" << cols << " rows: " << rows
                << " idx: " << idx << "\n";
    }*/
    memcpy(matrix_copy.data, data + idx * stride, cols * sizeof(Tscalar));

    return matrix_copy;
  }
  HOST_DEVICE void rowidxR(size_t idx, Matrix<Tscalar> *out) { //
    *out = Matrix<Tscalar>(1, cols, data + idx * stride);
    /*if (idx >= rows) {
      std::cout << "rowindxR operator  cols:" << cols << " rows: " << rows
                << " idx: " << idx << "\n";
    }*/
  }

  HOST_DEVICE Matrix<Tscalar> rowsliceR(size_t start, size_t end) {
    Matrix matrix_copy(end - start, cols, data + (start * stride));
    /*if (end >= rows) {
      std::cout << "rowsliceR operator  cols:" << cols << " rows: " << rows
                << " end idx: " << end << std::endl;
    }*/

    return matrix_copy;
  }
  HOST_DEVICE Tscalar &operator[](size_t idx) {
    /*if (idx >= size()) {
      std::cout << "[] operator  cols:" << cols << " rows: " << rows
                << " idx: " << idx << "\n";
    }*/
    return data[idx];
  }
  HOST_DEVICE Matrix *operator[](Matrix<int> idx) {
    Matrix *r;
    if (rows == 1) {
      r = new Matrix(1, idx.cols);
      for (size_t j = 0; j < idx.cols; j++) {
        *((*r).data + j) = *(data + idx[j]);
      }
    } else {
      r = new Matrix(idx.cols, cols);
      for (size_t i = 0; i < idx.cols; i++) {
        for (size_t j = 0; j < cols; j++) {
          *((*r).data + j + i * stride) = *(data + j + idx[i] * stride);
        }
      }
    }

    return r;
  }

  HOST_DEVICE Tscalar &operator()(size_t row, size_t col) {

    return data[row * stride + col];
  }
  HOST_DEVICE void iota(Tscalar start) {
    /*if (data == nullptr) {
      std::cout << "Nullpointer for data" << std::endl;
      return;
    }*/
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        *(data + j + i * stride) = start + j + i * stride;
      }
    }
  }

  void sqrt(Matrix<double> out) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        *(out.data + j + i * stride) = std::sqrt(*(data + j + i * stride));
      }
    }
  }
  HOST_DEVICE void sqrt_indexed(Matrix<double> *out, size_t idx) {
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        out->data[j + (idx + i) * stride] = sqrtf((float)data[j + i * stride]);
      }
    }
  }
  void zeros() { memset(data, 0, rows * cols); }
  Matrix<int> mod(Matrix<int> divisor) {
    Matrix<int> r = Matrix<int>(rows, cols);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        *(r.data + j + i * stride) = (*(data + j + i * stride)) % divisor(i, j);
      }
    }
    return r;
  }
  HOST_DEVICE Matrix<int> mod(int divisor) {
    Matrix<int> r = Matrix<int>(rows, cols);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        *(r.data + j + i * stride) = (*(data + j + i * stride)) % divisor;
      }
    }
    return r;
  }

  HOST_DEVICE Matrix<Tscalar> horizontal_concat(Matrix<Tscalar> m) {
    Matrix<Tscalar> res = Matrix<Tscalar>(1, m.size() + (this->size()));
    memcpy(res.data, this->data, this->size() * sizeof(Tscalar));
    memcpy(res.data + this->size(), m.data, m.size() * sizeof(Tscalar));
    return res;
  }
  HOST_DEVICE void add(Matrix<Tscalar> m) {
    for (int i = 0; i < size(); i++) {
      data[i] += m.data[i];
    }
  }
  Tscalar einsum_ij_ij(Matrix<Tscalar> m) {
    size_t eq_rows = std::min(rows, m.rows);
    size_t eq_cols = std::min(cols, m.cols);
    Tscalar sum = 0;
    for (int row_idx = 0; row_idx < eq_rows; row_idx++) {
      for (int col_idx = 0; col_idx < eq_cols; col_idx++) {
        sum += (*this)(row_idx, col_idx) * m(row_idx, col_idx);
      }
    }
    return sum;
  }
  Matrix<Tscalar> conj() {
    Matrix<Tscalar> res = Matrix<Tscalar>(rows, cols);
    for (int idx = 0; idx < size(); idx++) {
      res[idx] = std::conj(*(data + idx));
    }
    return res;
  }
};

#endif