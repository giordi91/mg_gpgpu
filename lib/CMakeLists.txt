#cmake 3.9 needed to properly build visual studio, otherwise 3.8 should suffice
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
SET(THIS_PROJECT "mg_gpgpu_core")
project(${THIS_PROJECT} LANGUAGES CXX CUDA)

message("STATUS" "---- Building ${THIS_PROJECT}")
message("STATUS" "---- current cmake dir is ${CMAKE_CURRENT_SOURCE_DIR}")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)


file(GLOB_RECURSE INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")

SET(CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
#find all the sources
message("STATUS" "---- source dir is:  ${CORE_SOURCE_DIR}")
file(GLOB SOURCES ${CORE_SOURCE_DIR}/*.cpp)
message("STATUS" "Found sources:")
foreach(ff ${SOURCES})
    message("STATUS" \t${ff})
endforeach()

file(GLOB CU_SOURCES ${CORE_SOURCE_DIR}/*.cu)
message("STATUS" "Found cuda sources:")
foreach(ff ${CU_SOURCES})
    message("STATUS" \t${ff})
endforeach()

add_library(${THIS_PROJECT} STATIC
  ${SOURCES}
  ${CU_SOURCES}
  ${INCLUDE_FILES}
  )
 
#setting c++ standard
target_compile_features(${THIS_PROJECT} PUBLIC cxx_std_11)
 

set_target_properties( ${THIS_PROJECT}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
