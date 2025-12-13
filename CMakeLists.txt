cmake_minimum_required(VERSION 3.15)

project(ggbond
    VERSION 0.1.0
    LANGUAGES C CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Add GGML as vendor library
add_subdirectory(vendor/ggml)

# Add GGML target
if(NOT TARGET ggml)
    add_library(ggml INTERFACE)
    target_include_directories(ggml INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/include
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/src
    )
    target_sources(ggml INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/src/ggml.c
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/src/ggml-alloc.c
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/src/ggml-backend.c
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/src/ggml-quants.c
    )
endif()

# Create a simple Python extension module using pybind11
pybind11_add_module(ggmllib
    src/ggmllib.cpp
)

# Link with GGML
target_link_libraries(ggmllib PRIVATE ggml)

# Include GGML headers
target_include_directories(ggmllib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml/include
)

# Set Python extension properties
set_target_properties(ggmllib PROPERTIES
    SUFFIX ".so"
    PREFIX ""
    POSITION_INDEPENDENT_CODE ON
)

# Set the output filename to ggml.so
set_target_properties(ggmllib PROPERTIES
    OUTPUT_NAME "ggml"
)

# Install the Python module
install(TARGETS ggmllib
    DESTINATION ggbond
)

# Install Python package files
install(DIRECTORY ggbond/
    DESTINATION ggbond
    FILES_MATCHING PATTERN "*.py"
)