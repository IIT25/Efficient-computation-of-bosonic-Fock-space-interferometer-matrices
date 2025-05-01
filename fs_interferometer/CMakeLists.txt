# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.29.8)
set(CMAKE_CXX_STANDARD 17)
# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version. 
project(
  fs_interferometer
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)
 
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
  enable_language(CUDA)
  set(USE_CUDA ON)
  message(STATUS "CUDA enabled")
else()
  message(STATUS "Building without CUDA")
endif()

# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
#set(Python_FIND_VIRTUALENV FIRST)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import site; print(site.getsitepackages()[0] + '/jaxlib/include')"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE src/main.cpp WITH_SOABI)
# set(XLA_DIR C:/Users/erdei/anaconda3/envs/piquasso/Lib/site-packages/jaxlib/include)
target_include_directories(_core PUBLIC "${XLA_DIR}")



target_link_libraries(_core PRIVATE pybind11::headers)


# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
# The install directory is the output (wheel) directory
install(TARGETS _core DESTINATION fs_interferometer)  

if(USE_CUDA)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  pybind11_add_module(
    gpu_ops
    ${CMAKE_CURRENT_LIST_DIR}/src/kernels.cc.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/gpu_ops.cc)
  target_include_directories(gpu_ops PRIVATE ${XLA_DIR})

  install(TARGETS gpu_ops DESTINATION fs_interferometer)
endif()