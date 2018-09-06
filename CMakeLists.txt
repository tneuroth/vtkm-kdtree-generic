set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required (VERSION 2.6)
project(TestKDTreeGeneric)

set(VTKm_DIR "../vtkm-build/lib/cmake/vtkm-1.2")
set (CMAKE_CXX_FLAGS "-O3" )

find_package(VTKm  REQUIRED)

add_executable(test test.cpp)

target_link_libraries(test  vtkm_cont)
