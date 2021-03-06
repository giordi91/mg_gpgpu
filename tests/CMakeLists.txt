cmake_minimum_required(VERSION 2.6)
SET(THIS_PROJECT "mg_gpgpu_tests")
project(${THIS_PROJECT})
find_package(CUDA QUIET REQUIRED)

message("STATUS" "---- current cmake dir is ${CMAKE_CURRENT_SOURCE_DIR}")


#add_subdirectory("../mg_gpgpu_core" "${CMAKE_CURRENT_BINARY_DIR}")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/mg_gpgpu_core/include)
INCLUDE_DIRECTORIES("/usr/local/include")

get_property(dirs  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

if(UNIX AND NOT APPLE)
	message(STATUS, "---- Currently building on UNIX system")
endif()


SET(CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
#find all the sources
message("STATUS" "---- source dir is:  ${CORE_SOURCE_DIR}")
file(GLOB SOURCES ${CORE_SOURCE_DIR}/*.cu)
message("STATUS" "Found sources:")
foreach(ff ${SOURCES})
    message("STATUS" \t${ff})
endforeach()


set(CUDA_NVCC_FLAGS -std=c++11 -gencode=arch=compute_61,code=sm_61 -O3 -Xptxas -dlcm=cg)  
#adding library
cuda_add_executable(${THIS_PROJECT} ${SOURCES}) 
target_link_libraries(${THIS_PROJECT}  mg_gpgpu_core gmock gmock_main pthread /usr/local/cuda/lib64/libcudart.so )
add_dependencies(${THIS_PROJECT} mg_gpgpu_core)


