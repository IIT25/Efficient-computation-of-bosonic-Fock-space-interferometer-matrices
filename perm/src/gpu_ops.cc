#include "kernels.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_perm_fwd, calc_perm_fwd_host,
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

// Creates symbol FooBwd with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    calc_perm_bwd, calc_perm_bwd_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::C128>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Arg<ffi::Buffer<ffi::U64>>()
        .Arg<ffi::Buffer<ffi::U32>>()
        .Arg<ffi::Buffer<ffi::F64>>()
        .Arg<ffi::Buffer<ffi::C128>>()
        .Ret<ffi::Buffer<ffi::C128>>(),
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooFwd, FooFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::C128>>()            // a
        .Arg<ffi::Buffer<ffi::C128>>()            // b
        .Ret<ffi::Buffer<ffi::C128>>()            // c
        .Ret<ffi::Buffer<ffi::C128>>()            // b_plus_1
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

// Creates symbol FooBwd with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooBwd, FooBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>() // stream
        .Arg<ffi::Buffer<ffi::F32>>()             // c_grad
        .Arg<ffi::Buffer<ffi::F32>>()             // a
        .Arg<ffi::Buffer<ffi::F32>>()             // b_plus_1
        .Ret<ffi::Buffer<ffi::F32>>()             // a_grad
        .Ret<ffi::Buffer<ffi::F32>>()             // b_grad
        .Attr<size_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible}); // cudaGraph enabled

template <typename T> py::capsule EncapsulateFfiHandler(T *fn) {
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}

PYBIND11_MODULE(gpu_ops, m) {
  m.doc() = R"pbdoc(
          Permanent calculator plugin
          -----------------------
  
          .. currentmodule:: scikit_build_example
  
          .. autosummary::
             :toctree: _generate
  
             permanent
      )pbdoc";
  m.def("foo", []() {
    py::dict registrations;
    registrations["calc_perm_fwd"] = EncapsulateFfiHandler(calc_perm_fwd);
    registrations["calc_perm_bwd"] = EncapsulateFfiHandler(calc_perm_bwd);
    registrations["foo_fwd"] = EncapsulateFfiHandler(FooFwd);
    registrations["foo_bwd"] = EncapsulateFfiHandler(FooBwd);
    return registrations;
  });
  m.attr("__version__") = "dev";
}
