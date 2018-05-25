cmake_minimum_required(VERSION 2.8.11)
project(SynCamStereo)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
add_definitions("-std=c++11")
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)



add_executable(mainFaceBlink src/Queue.cpp src/mainFaceBlink_zoom.cpp)
target_link_libraries(mainFaceBlink ${OpenCV_LIBS})
target_link_libraries(mainFaceBlink ${PCL_LIBRARIES})
target_link_libraries(mainFaceBlink ${dlib_LIBS} stdc++fs)

add_executable(mainCapture src/mainCapture.cpp)
target_link_libraries(mainCapture ${OpenCV_LIBS})
target_link_libraries(mainCapture ${PCL_LIBRARIES})


