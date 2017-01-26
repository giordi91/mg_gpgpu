cmake_minimum_required(VERSION 2.6)
project(mg_gpu_core)



message("STATUS" "---- current cmake dir is ${CMAKE_SOURCE_DIR}")


INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)


if(UNIX AND NOT APPLE)
	message(STATUS, "---- Currently building on UNIX system")
endif()

SET(CORE_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)
#find all the sources
message("STATUS" "---- source dir is:  ${CORE_SOURCE_DIR}")
file(GLOB SOURCES ${CORE_SOURCE_DIR}/*.cpp)
message("STATUS" "Found sources:")
foreach(ff ${SOURCES})
    message("STATUS" \t${ff})
endforeach()


#adding library
add_library(${CMAKE_PROJECT_NAME} STATIC ${SOURCES}) 

#shared flags
set(SHARED_COMPILE_FLAGS "-std=c++11 -Wall -m64 -D_DEBUG")

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} ${SPECIFIC_COMPILE_FLAGS}")



