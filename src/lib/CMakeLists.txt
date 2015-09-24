cmake_minimum_required(VERSION 2.8)
project(drwnLibs)
if (DEFINED ENV{DRWN_DIR})
    set(DRWN_DIR $ENV{DRWN_DIR})
else()
    set(DRWN_DIR /home/ebi/Projects/darwin)
endif()


find_package(Eigen3)
include_directories(${EIGEN3_INCLUDE_DIR})


include_directories(${DRWN_DIR}/include)
add_subdirectory(base ${CMAKE_BINARY_DIR}/base)
add_subdirectory(io ${CMAKE_BINARY_DIR}/io)
add_subdirectory(ml ${CMAKE_BINARY_DIR}/ml)
add_subdirectory(pgm ${CMAKE_BINARY_DIR}/pgm)
add_subdirectory(vision ${CMAKE_BINARY_DIR}/vision)
