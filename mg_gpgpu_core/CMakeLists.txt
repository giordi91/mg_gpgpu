cmake_minimum_required(VERSION 2.6)
SET(THIS_PROJECT "mg_gpgpu_core")


project(${THIS_PROJECT})
find_package(CUDA QUIET REQUIRED)

message("STATUS" "---- current cmake dir is ${CMAKE_CURRENT_SOURCE_DIR}")


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(UNIX AND NOT APPLE)
	message(STATUS, "---- Currently building on UNIX system")
endif()
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


set(CUDA_NVCC_FLAGS  -gencode=arch=compute_50,code=sm_50  -O3 )  
#adding library
cuda_add_library(${THIS_PROJECT} STATIC ${SOURCES} ${CU_SOURCES}) 
