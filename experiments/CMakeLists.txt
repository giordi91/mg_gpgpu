cmake_minimum_required(VERSION 2.6)
SET(THIS_PROJECT "mg_gpgpu_experiments")
project(${THIS_PROJECT})
find_package(CUDA QUIET REQUIRED)

message("STATUS" "---- current cmake dir is ${CMAKE_CURRENT_SOURCE_DIR}")


#add_subdirectory("../mg_gpgpu_core" "${CMAKE_CURRENT_BINARY_DIR}")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/mg_gpgpu_core/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/experiments)

get_property(dirs  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

if(UNIX AND NOT APPLE)
	message(STATUS, "---- Currently building on UNIX system")
endif()

SET(CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})


set(CUDA_NVCC_FLAGS -std=c++11 -gencode=arch=compute_61,code=sm_61 -O3 )  
#adding library
cuda_add_executable(simple_copy_buffer simple_copy_buffer.cu)
target_link_libraries(simple_copy_buffer  mg_gpgpu_core pthread /usr/local/cuda/lib64/libcudart.so )
add_dependencies(simple_copy_buffer mg_gpgpu_core)


#cuda_compile_ptx(
#        cuda_ptx_files
#        simple_copy_buffer.cu
#        OPTIONS   -gencode=arch=compute_61,code=sm_61   --source-in-ptx --ptxas-options=-v  --generate-line-info
#
#)
#add_custom_target(ptx ALL --generate-line-info
#        DEPENDS ${cuda_ptx_files} simple_copy_buffer.cu
#        SOURCES simple_copy_buffer.cu)
set(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/simple_copy_buffer.cu)
add_custom_target(genptx ALL COMMAND /usr/local/cuda/bin/nvcc ${SOURCES}
        -I${CMAKE_SOURCE_DIR}/mg_gpgpu_core/include -std=c++11
        -gencode=arch=compute_61,code=sm_61  -ptx --source-in-ptx --ptxas-options=-v  --generate-line-info
        )

add_custom_target( gen_sass ALL COMMAND /usr/local/cuda/bin/cuobjdump -sass $<TARGET_FILE:simple_copy_buffer>  &>simple_copy_buffer.ass)