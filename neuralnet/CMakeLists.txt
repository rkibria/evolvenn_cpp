cmake_minimum_required(VERSION 3.0)

add_library(neuralnet STATIC
    src/neuralnet.cpp
    )

target_include_directories(neuralnet PUBLIC include)

add_subdirectory(tests)
