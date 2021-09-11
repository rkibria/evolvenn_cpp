cmake_minimum_required(VERSION 3.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

project(evolvenn)

project(evolvenn LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost)
find_package(TBB)

add_subdirectory(neuralnet)
add_subdirectory(population)

add_executable(evolvenn
    main.cpp
    )

target_link_libraries(evolvenn
    neuralnet
    population
    Boost::boost
    Boost::serialization
    TBB::tbb
    )
