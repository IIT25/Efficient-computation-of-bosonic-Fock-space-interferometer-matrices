#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <ostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

std::vector<std::vector<double>> numpy_to_matrix(
    pybind11::array_t<double, pybind11::array::c_style | py::array::forcecast>
        numpy_array) {
  py::buffer_info bufferinfo = numpy_array.request();

  size_t rows = bufferinfo.shape[0];
  size_t cols = bufferinfo.shape[1];
  std::vector<std::vector<double>> matrix;
  double *data = static_cast<double *>(bufferinfo.ptr);
  for (int i = 0; i < rows; i++) {
    matrix.push_back(std::vector<double>());
    for (int j = 0; j < cols; j++) {
      matrix[i].push_back(*data);
      data++;
    }
  }

  return matrix;
}
std::vector<std::vector<double>>
add_to_matrix(std::vector<std::vector<double>> m) {
  m.push_back(m.back());
  double sum_of_elems;
  for (std::vector<double> &row : m) {
    sum_of_elems = 0;
    for (auto &n : row)
      sum_of_elems += n;
    row.push_back(sum_of_elems);
  }
  return m;
}
std::vector<std::vector<std::vector<double>>>
calc_perm(int cutoff, pybind11::array interferometer) {
  std::vector<std::vector<std::vector<double>>> matrices;
  matrices.push_back({{1.0}});
  std::vector<std::vector<double>> intfrm = numpy_to_matrix(interferometer);
  matrices.push_back(intfrm);
  for (int i = 1; i < cutoff; i++) {
    matrices.push_back(add_to_matrix(matrices[i]));
  }
  return matrices;
}

PYBIND11_MODULE(_core, m) {

  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";
  /*m.def("MatrixAdd", &MatrixAdd, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");*/
  m.def("calc_perm", &calc_perm, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");
#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
