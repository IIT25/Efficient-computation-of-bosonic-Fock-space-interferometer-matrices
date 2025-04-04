#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstring>
#include <stdexcept>

#ifndef DEBUG
#include <iostream>
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

  Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), stride(cols) {
    data = new Tscalar[rows * cols];
    owner = true;
    refcount = new size_t;
    (*refcount) = 1;
  }

  Matrix(size_t rows, size_t cols, Tscalar *data)
      : rows(rows), cols(cols), stride(cols), data(data) {
    owner = false;
    refcount = new size_t;
    (*refcount) = 1;
  }

  Matrix(const Matrix &matrix)
      : rows(matrix.rows), cols(matrix.cols), stride(matrix.stride),
        data(matrix.data), owner(matrix.owner), refcount(matrix.refcount) {
    (*refcount)++;
  }

  Matrix()
      : rows(0), cols(0), stride(0), data(nullptr), owner(false),
        refcount(nullptr) {}

  void operator=(const Matrix &matrix) {
    rows = matrix.rows;
    cols = matrix.cols;
    stride = matrix.stride;
    data = matrix.data;
    owner = matrix.owner;
    refcount = matrix.refcount;

    (*refcount)++;
  }

  size_t size() { return rows * cols; }
  Matrix copy() {
    Matrix matrix_copy(rows, cols);

    memcpy(matrix_copy.data, data, size() * sizeof(Tscalar));

    return matrix_copy;
  }

  ~Matrix() {
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
  void print() {
    std::cout << std::endl
              << "The stored matrix (" << rows << " x " << cols
              << "):" << std::endl;
    for (size_t row_idx = 0; row_idx < rows; row_idx++) {
      for (size_t col_idx = 0; col_idx < cols; col_idx++) {
        size_t element_idx = row_idx * stride + col_idx;
        std::cout << " " << data[element_idx];
      }
      std::cout << std::endl;
    }
    std::cout << "------------------------" << std::endl;
  }
  Matrix<Tscalar> rowidx(size_t idx) { //
    Matrix<Tscalar> matrix_copy(1, cols);
    if (idx >= rows) {
      std::cout << "rowindx operator  cols:" << cols << " rows: " << rows
                << " idx: " << idx << "\n";
    }
    memcpy(matrix_copy.data, data + idx * stride, cols * sizeof(Tscalar));

    return matrix_copy;
  }
  Matrix<Tscalar> &rowidxR(size_t idx) { //
    Matrix<Tscalar> matrix_copy(1, cols, data + idx * stride);
    if (idx >= rows) {
      std::cout << "rowindxR operator  cols:" << cols << " rows: " << rows
                << " idx: " << idx << "\n";
    }

    return matrix_copy;
  }

  Matrix<Tscalar> rowsliceR(size_t start, size_t end) {
    Matrix matrix_copy(end - start, cols, data + (start * stride));
    /*if (end >= rows) {
      std::cout << "rowsliceR operator  cols:" << cols << " rows: " << rows
                << " end idx: " << end << std::endl;
    }*/

    return matrix_copy;
  }
  Tscalar &operator[](size_t idx) {
    /*if (idx >= size()) {
      std::cout << "[] operator  cols:" << cols << " rows: " << rows
                << " idx: " << idx << "\n";
    }*/
    return data[idx];
  }
  Matrix *operator[](Matrix<int> idx) {
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

  Tscalar &operator()(size_t row, size_t col) {

    return data[row * stride + col];
  }
  void iota(Tscalar start) {
    if (data == nullptr) {
      std::cout << "Nullpointer for data" << std::endl;
      return;
    }
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
  Matrix<int> mod(int divisor) {
    Matrix<int> r = Matrix<int>(rows, cols);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        *(r.data + j + i * stride) = (*(data + j + i * stride)) % divisor;
      }
    }
    return r;
  }

  Matrix<Tscalar> horizontal_concat(Matrix<Tscalar> m) {
    Matrix<Tscalar> res = Matrix<Tscalar>(1, m.size() + (this->size()));
    memcpy(res.data, this->data, this->size() * sizeof(Tscalar));
    memcpy(res.data + this->size(), m.data, m.size() * sizeof(Tscalar));
    return res;
  }
  void add(Matrix<Tscalar> m) {
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
    for (int idx; idx < size(); idx++) {
      res[idx] = std::conj(*(data + idx));
    }
    return res;
  }
};

#endif