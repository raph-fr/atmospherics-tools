cmake_minimum_required(VERSION 3.15)
project(FastFeedback)

include(ExternalProject)

set(OSCPROB_SOURCE_DIR ${CMAKE_BINARY_DIR}/oscprob-src)

ExternalProject_Add(OscProb
    DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
    GIT_REPOSITORY https://github.com/joaoabcoelho/OscProb
    SOURCE_DIR ${OSCPROB_SOURCE_DIR}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${MAKE}
    INSTALL_COMMAND "${oscprob_lib_install_commands}"
)

message(STATUS "Source dir -> ${OSCPROB_SOURCE_DIR}")

find_package(ROOT REQUIRED COMPONENTS RIO Net)

add_library(OscProbLib STATIC IMPORTED GLOBAL)
set_property(TARGET OscProbLib 
    PROPERTY IMPORTED_LOCATION 
    ${OSCPROB_SOURCE_DIR}/lib/libOscProb.so
)

set(OSCPROB_INCLUDE_DIR ${OSCPROB_SOURCE_DIR}/inc/)

add_subdirectory(src)
target_link_libraries(Oscillogram PRIVATE OscProbLib)
