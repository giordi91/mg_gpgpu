cmake_minimum_required(VERSION 2.6)
SET(THIS_PROJECT "mg_gpgpu_core")


project(${THIS_PROJECT})
find_package(CUDA QUIET REQUIRED)

message("STATUS" "---- current cmake dir is ${CMAKE_CURRENT_SOURCE_DIR}")


INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)


if(UNIX AND NOT APPLE)
	message(STATUS, "---- Currently building on UNIX system")
endif()
set(CUDA_NVCC_FLAGS -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_50,code=compute_50 -rdc=true  -O3 -use_fast_math -g -lineinfo)  

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


#adding library
cuda_add_library(${THIS_PROJECT} STATIC ${SOURCES} ${CU_SOURCES}) 
target_link_libraries(${THIS_PROJECT} cuda cudart )

#shared flags
set(SHARED_COMPILE_FLAGS "-std=c++11 -Wall -m64 -D_DEBUG")

set_target_properties(${THIS_PROJECT} PROPERTIES COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} ${SPECIFIC_COMPILE_FLAGS}")



