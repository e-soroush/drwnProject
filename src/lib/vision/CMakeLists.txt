project(drwnVision)
find_package(OpenCV REQUIRED)
set (SRC_LIST "")
aux_source_directory(. SRC_LIST)
add_library(${PROJECT_NAME} ${SRC_LIST})
target_link_libraries(${PROJECT_NAME} ${OPENCV_LIBS} drwnPGM drwnML)
