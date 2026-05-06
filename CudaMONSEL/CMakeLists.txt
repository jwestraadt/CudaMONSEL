cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(CudaMONSEL
VERSION 1.0
DESCRIPTION "CUDA MONSEL"
LANGUAGES CXX CUDA
)

message(${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(CudaMONSEL/gov/nist/microanalysis)
