cmake_minimum_required(VERSION 3.0)

add_library(population STATIC
    src/population.cpp
    )

target_include_directories(population PUBLIC include)

# add_subdirectory(tests)
