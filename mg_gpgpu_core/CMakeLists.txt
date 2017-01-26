cmake_minimum_required(VERSION 2.6)
SET(THIS_PROJECT "mg_gpgpu_core")
project(${THIS_PROJECT})

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


#adding library
add_library(${THIS_PROJECT} STATIC ${SOURCES}) 

#shared flags
set(SHARED_COMPILE_FLAGS "-std=c++11 -Wall -m64 -D_DEBUG")

set_target_properties(${THIS_PROJECT} PROPERTIES COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} ${SPECIFIC_COMPILE_FLAGS}")



