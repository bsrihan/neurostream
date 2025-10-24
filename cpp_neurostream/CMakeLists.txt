cmake_minimum_required(VERSION 3.15)
project(neurostream_cpp VERSION 0.1.0 LANGUAGES CXX)

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Find Eigen3 (matrix operations for re-referencing)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find hiredis (Redis client)
find_path(HIREDIS_INCLUDE_DIR hiredis/hiredis.h)
find_library(HIREDIS_LIBRARY hiredis)

if(NOT HIREDIS_INCLUDE_DIR OR NOT HIREDIS_LIBRARY)
    message(FATAL_ERROR "hiredis not found. Install with: brew install hiredis (macOS) or apt install libhiredis-dev (Linux)")
endif()

message(STATUS "Found hiredis: ${HIREDIS_LIBRARY}")

# Source files (only include what exists for now)
set(SOURCES
    src/main.cpp
    src/redis_io.cpp
    src/rereference.cpp
    # src/filter.cpp        # TODO: implement
    # src/features.cpp      # TODO: implement
)

# Main executable
add_executable(neurostream_processor ${SOURCES})
target_include_directories(neurostream_processor PRIVATE ${HIREDIS_INCLUDE_DIR})
target_link_libraries(neurostream_processor
    Eigen3::Eigen
    ${HIREDIS_LIBRARY}
)

# Installation
install(TARGETS neurostream_processor DESTINATION bin)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "C++ flags (Release): ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags (Debug): ${CMAKE_CXX_FLAGS_DEBUG}")
