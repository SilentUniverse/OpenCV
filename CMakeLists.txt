cmake_minimum_required(VERSION 3.14)
project(opencv_test)
set(CMAKE_CXX_STANDARD 14)
set(SOURCE_FILES main.cpp steger.cpp main.h)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(opencv_test ${SOURCE_FILES})
target_link_libraries(opencv_test ${OpenCV_LIBS} )

