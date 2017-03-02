cmake_minimum_required (VERSION 2.8)

project(HarrietRPi)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3 REQUIRED)
find_package(Threads REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ./)

#add_executable(loadingstation LoadingStation/LoadingStation.cpp)
#target_link_libraries(loadingstation ${OpenCV_LIBS} ntcore ${CMAKE_THREAD_LIBS_INIT})

#add_executable(gearlift GearLift/GearLift.cpp)
#target_link_libraries(gearlift ${OpenCV_LIBS} ntcore ${CMAKE_THREAD_LIBS_INIT})

#add_executable(calibration calibration/calibration.cpp)
#target_link_libraries(calibration ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_executable(highgoal HighGoal.cpp)
target_link_libraries(highgoal ${OpenCV_LIBS} ntcore ${CMAKE_THREAD_LIBS_INIT})

add_executable(test_video test_video.cpp)
target_link_libraries(test_video ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

add_executable(takepics takepics.cpp)
target_link_libraries(takepics ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})

