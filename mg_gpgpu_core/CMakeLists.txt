cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
SET(THIS_PROJECT "mg_gpgpu_core")
project(${THIS_PROJECT} LANGUAGES CXX CUDA)

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
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
target_compile_features(${THIS_PROJECT} PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties( ${THIS_PROJECT}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
#add_executable(particle_test test.cu)
 
#set_property(TARGET particle_test 
#             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(particle_test PRIVATE particles)


#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
#file(GLOB_RECURSE INCLUDE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
#message("STATUS" "Found includes:")
#foreach(ff ${INCLUDE_FILES})
#    message("STATUS" \t${ff})
#endforeach()

#if(UNIX AND NOT APPLE)
	#message(STATUS, "---- Currently building on UNIX system")
#endif()
#SET(CORE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
#find all the sources
#message("STATUS" "---- source dir is:  ${CORE_SOURCE_DIR}")
#file(GLOB SOURCES ${CORE_SOURCE_DIR}/*.cpp)
#message("STATUS" "Found sources:")
#foreach(ff ${SOURCES})
#    message("STATUS" \t${ff})
#endforeach()
#file(GLOB CU_SOURCES ${CORE_SOURCE_DIR}/*.cu)
#message("STATUS" "Found cuda sources:")
#foreach(ff ${CU_SOURCES})
#    message("STATUS" \t${ff})
#endforeach()


#set(CUDA_NVCC_FLAGS -std=c++11 -gencode=arch=compute_61,code=sm_61 -O3 )  
#adding library
#add_library(${THIS_PROJECT} STATIC ${INCLUDE_FILES} ${SOURCES} ${CU_SOURCES}) 
