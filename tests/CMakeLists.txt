cmake_minimum_required(VERSION 2.6)
SET(THIS_PROJECT "mg_gpgpu_tests")
project(${THIS_PROJECT})

message("STATUS" "---- current cmake dir is ${CMAKE_SOURCE_DIR}")

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
target_link_libraries(${THIS_PROJECT} mg_gpgpu_core)
add_dependencies(${THIS_PROJECT} mg_gpgpu_core)
#shared flags
set(SHARED_COMPILE_FLAGS "-std=c++11 -Wall -m64 -lgmock -lgmockfes_main -lpthread")
set(SHARED_LINK_FLAGS "-lgmock -lgmock_main -lpthread ")

set_target_properties(${THIS_PROJECT} PROPERTIES 
		COMPILE_FLAGS "${SHARED_COMPILE_FLAGS} ${SPECIFIC_COMPILE_FLAGS}"
		LINK_FLAGS "${SHARED_LINK_FLAGS} ${SPECIFIC_COMPILE_FLAGS}")



