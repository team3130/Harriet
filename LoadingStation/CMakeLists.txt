cmake_minimum_required (VERSION 2.8)

project(LoadingStation)

set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3 REQUIRED)
# We need thread support
find_package(Threads REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS} ./)

add_executable(LoadingStation LoadingStation.cpp)

target_link_libraries(LoadingStation ${OpenCV_LIBS} ntcore ${CMAKE_THREAD_LIBS_INIT})

