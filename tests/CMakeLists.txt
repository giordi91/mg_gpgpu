
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
SET(THIS_PROJECT "mg_gpgpu_tests")
project(${THIS_PROJECT} LANGUAGES CXX CUDA)

message("STATUS" "---- current cmake dir is ${CMAKE_CURRENT_SOURCE_DIR}")


#add_subdirectory("../mg_gpgpu_core" "${CMAKE_CURRENT_BINARY_DIR}")
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/mg_gpgpu_core/include  ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
#INCLUDE_DIRECTORIES("/usr/local/include")





 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11

 

 
add_executable(${THIS_PROJECT} parallel_reduce_test.cu)
target_compile_features(${THIS_PROJECT} PUBLIC cxx_std_11) 
set_property(TARGET ${THIS_PROJECT} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(${THIS_PROJECT} mg_gpgpu_core)

target_link_libraries(${THIS_PROJECT} PRIVATE mg_gpgpu_core)












#get_property(dirs  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#  message(STATUS "dir='${dir}'")
#endforeach()

#if(UNIX AND NOT APPLE)
#	message(STATUS, "---- Currently building on UNIX system")
#	set(EXTRA_LIB pthread)
#endif()


#SET(CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
##find all the sources
#message("STATUS" "---- source dir is:  ${CORE_SOURCE_DIR}")
#file(GLOB SOURCES ${CORE_SOURCE_DIR}/*.cu)
#message("STATUS" "Found sources:")
#foreach(ff ${SOURCES})
#    message("STATUS" \t${ff})
#endforeach()


#set(CUDA_NVCC_FLAGS -std=c++11 -gencode=arch=compute_61,code=sm_61 -O3 -Xptxas -dlcm=cg)  
##adding library
#cuda_add_executable(${THIS_PROJECT} ${SOURCES}) 
#target_link_libraries(${THIS_PROJECT}  mg_gpgpu_core  ${EXTRA_LIB} ${CUDA_LIBRARIES} )
#add_dependencies(${THIS_PROJECT} mg_gpgpu_core)


