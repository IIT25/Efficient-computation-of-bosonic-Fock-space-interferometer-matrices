#include "kernels.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fs_interferometer, fs_interferometer_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::C128>>(),        // result
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    fs_interferometer_fwd, fs_interferometer_fwd_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::C128>>()         // result
        .Ret<ffi::Buffer<ffi::U64>>()          // result dimensions
        .Ret<ffi::Buffer<ffi::U32>>()          // helper_idx
        .Ret<ffi::Buffer<ffi::F64>>(),         // helper_sqrt
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_fs_interferometer_bwd, calc_fs_interferometer_bwd_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::C128>>(),
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

template <typename T> py::capsule EncapsulateFfiHandler(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}

PYBIND11_MODULE(gpu_ops, m) {
  m.doc() = R"pbdoc(
          fs_interferometeranent calculator plugin
          -----------------------
  
          .. currentmodule:: scikit_build_example
  
          .. autosummary::
             :toctree: _generate
  
             fs_interferometeranent
      )pbdoc";
  m.def("foo", []() {
    py::dict registrations;
    registrations["fs_interferometer"] =
        EncapsulateFfiHandler(fs_interferometer);
    registrations["fs_interferometer_fwd"] =
        EncapsulateFfiHandler(fs_interferometer_fwd);
    registrations["calc_fs_interferometer_bwd"] =
        EncapsulateFfiHandler(calc_fs_interferometer_bwd);
    return registrations;
  });
  m.attr("__version__") = "dev";
}
