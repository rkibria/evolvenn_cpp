cmake_minimum_required(VERSION 3.0)

set(UNIT_TEST_LIST
    basics
    )

foreach(NAME IN LISTS UNIT_TEST_LIST)
    list(APPEND UNIT_TEST_SOURCE_LIST ${NAME}_test.cpp)
endforeach()
 
set(TARGET_NAME neuralnet_tests)

add_executable(${TARGET_NAME}
  main.cpp
  ${UNIT_TEST_SOURCE_LIST})

target_link_libraries(${TARGET_NAME} PUBLIC neuralnet)

target_include_directories(${TARGET_NAME} PUBLIC .)

add_test(
    NAME ${TARGET_NAME}
    COMMAND ${TARGET_NAME} -o report.xml -r junit
    )
