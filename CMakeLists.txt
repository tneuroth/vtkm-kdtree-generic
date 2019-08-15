set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required (VERSION 2.6)
project(TestKDTreeGeneric LANGUAGES CXX CUDA)

include(ParallelUtils.cmake)
enable_cuda_support()

set(VTKm_DIR "../vtkm-build/lib/cmake/vtkm-1.3")
set(CMAKE_CXX_FLAGS "-O3 -std=c++11" )

set(CUDA_NVCC_FLAGS "-std=c++11 -O4 -lcuda -lcudart")

find_package(VTKm REQUIRED)
find_package(CUDA REQUIRED)

cuda_add_executable(test test.cu)
target_link_libraries(test  vtkm_cont)

