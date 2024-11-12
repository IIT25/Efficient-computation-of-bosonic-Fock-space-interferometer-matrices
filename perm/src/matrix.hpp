#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <cstring>

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

#ifdef DEBUG
  void print() {
    std::cout << std::endl << "The stored matrix:" << std::endl;
    for (size_t row_idx = 0; row_idx < rows; row_idx++) {
      for (size_t col_idx = 0; col_idx < cols; col_idx++) {
        size_t element_idx = row_idx * stride + col_idx;
        std::cout << " " << data[element_idx];
      }
      std::cout << std::endl;
    }
    std::cout << "------------------------" << std::endl;
  }
#endif

  Tscalar &operator[](size_t idx) { return data[idx]; }

  Tscalar &operator()(size_t row, size_t col) {
    return data[row * stride + col];
  }
};

#endif