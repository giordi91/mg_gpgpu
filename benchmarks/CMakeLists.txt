#cmake 3.9 needed to properly build visual studio, otherwise 3.8 should suffice
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
SET(THIS_PROJECT "mg_gpgpu_benchmarks")
project(${THIS_PROJECT} LANGUAGES CXX CUDA)
cmake_minimum_required(VERSION 2.6)

message("STATUS" "---- Building ${THIS_PROJECT}")
message("STATUS" "---- current cmake dir is ${CMAKE_CURRENT_SOURCE_DIR}")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/mg_gpgpu_core/include ${CMAKE_SOURCE_DIR}/benchmarks)
INCLUDE_DIRECTORIES()

#if(UNIX AND NOT APPLE)
#	message(STATUS, "---- Currently building on UNIX system")
#endif()

SET(CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#find all the sources
message("STATUS" "---- source dir is:  ${CORE_SOURCE_DIR}")
file(GLOB SOURCES ${CORE_SOURCE_DIR}/*.cu)
message("STATUS" "Found sources:")
foreach(ff ${SOURCES})
    message("STATUS" \t${ff})
endforeach()


add_executable(${THIS_PROJECT} ${SOURCES})

#setting c++ standard
target_compile_features(${THIS_PROJECT} PUBLIC cxx_std_11) 


 #enabling separated compilation
set_property(TARGET ${THIS_PROJECT} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

#setting the dep with the core lib
add_dependencies(${THIS_PROJECT} mg_gpgpu_core)

#linking libraries with our core lib
target_link_libraries(${THIS_PROJECT} PRIVATE mg_gpgpu_core)

#TODO(giordi) set optimization flags for cuda, check if now that cuda is first citizen language
# if is done automatically
#set(CUDA_NVCC_FLAGS -std=c++11 -gencode=arch=compute_61,code=sm_61 -O3 )  