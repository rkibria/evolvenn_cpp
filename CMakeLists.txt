cmake_minimum_required(VERSION 3.0)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

project(evolvenn)

find_package(Boost)
find_package(TBB)

add_subdirectory(neuralnet)

add_executable(evolvenn
    main.cpp
    )

target_link_libraries(evolvenn
    neuralnet
    Boost::boost
    Boost::serialization
    TBB::tbb
    )
