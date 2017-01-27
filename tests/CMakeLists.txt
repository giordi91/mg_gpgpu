cmake_minimum_required(VERSION 2.6)
SET(THIS_PROJECT "mg_gpgpu_tests")
project(${THIS_PROJECT})
find_package(CUDA QUIET REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
#list(APPEND CUDA_NVCC_FLAGS "--relocatable-device-code=true ")
#find_package(CUDA QUIET REQUIRED)
message("STATUS" "---- current cmake dir is ${CMAKE_SOURCE_DIR}")

set(CUDA_NVCC_FLAGS -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -rdc=true  -O3 -use_fast_math -g -lineinfo)  

add_subdirectory("../mg_gpgpu_core" "${CMAKE_CURRENT_BINARY_DIR}")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/../mg_gpgpu_core/include)

get_property(dirs  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

if(UNIX AND NOT APPLE)
	message(STATUS, "---- Currently building on UNIX system")
endif()

SET(CORE_SOURCE_DIR ${CMAKE_SOURCE_DIR})
#find all the sources
message("STATUS" "---- source dir is:  ${CORE_SOURCE_DIR}")
file(GLOB SOURCES ${CORE_SOURCE_DIR}/*.cpp)
message("STATUS" "Found sources:")
foreach(ff ${SOURCES})
    message("STATUS" \t${ff})
endforeach()


#adding library
add_executable(${THIS_PROJECT} ${SOURCES}) 
target_link_libraries(${THIS_PROJECT}  mg_gpgpu_core gmock gmock_main pthread )
add_dependencies(${THIS_PROJECT} mg_gpgpu_core)
#shared flags
set(SHARED_COMPILE_FLAGS "-std=c++11 -Wall -m64 -lgmock -lgmockfes_main -lpthread")
set(SHARED_LINK_FLAGS "-L/usr/local/cuda/lib64")

set_target_properties(${THIS_PROJECT} PROPERTIES 
		COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} ${SPECIFIC_COMPILE_FLAGS}"
		LINK_FLAGS "${SHARED_LINK_FLAGS} ${SPECIFIC_COMPILE_FLAGS}")



