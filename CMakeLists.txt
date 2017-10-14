cmake_minimum_required(VERSION 2.6)
SET(THIS_PROJECT "mg_gpgpu")
project(${THIS_PROJECT})

add_subdirectory(mg_gpgpu_core)
add_subdirectory(tests)
#add_subdirectory(benchmarks)
#add_subdirectory(experiments)
