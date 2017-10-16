#cmake 3.9 needed to properly build visual studio, otherwise 3.8 should suffice
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

SET(THIS_PROJECT "mg_gpgpu_tests")
#enabling cuda
project(${THIS_PROJECT} LANGUAGES CXX CUDA)

message("STATUS" "---- Building ${THIS_PROJECT}")
message("STATUS" "---- current cmake dir is ${CMAKE_CURRENT_SOURCE_DIR}")

#finding the includes
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib/include )


#find all the sources
SET(CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
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

