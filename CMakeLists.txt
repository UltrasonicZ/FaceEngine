cmake_minimum_required(VERSION 3.5)

project(testSSD_Prj LANGUAGES CXX)
set(TARGET face_engine)

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

#寻找OpenCV库
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

# 调用openmp
FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
    message("OPENMP FOUND")  
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  


#打印调试信息
MESSAGE(STATUS "Project: ${PROJECT_NAME}")
MESSAGE(STATUS "OpenCV library status:")
MESSAGE(STATUS "    version: ${OpenCV_VERSION}")
MESSAGE(STATUS "    libraries: ${OpenCV_LIBS}")
MESSAGE(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(
    inc/
    data/models
    )

set(SOURCE_LIST ${CMAKE_SOURCE_DIR}/src/facerecg_interface.cpp)

list(APPEND SOURCE_LIST
    ${CMAKE_SOURCE_DIR}/src/ncnnssd.cpp
    ${CMAKE_SOURCE_DIR}/src/face_alignment.cpp
    #${CMAKE_SOURCE_DIR}/src/rgbalive.cpp
    ${CMAKE_SOURCE_DIR}/src/fas.cpp
    ${CMAKE_SOURCE_DIR}/src/eye_exist.cpp
    ${CMAKE_SOURCE_DIR}/src/mouth_exist.cpp
    ${CMAKE_SOURCE_DIR}/src/nose_exist.cpp
    ${CMAKE_SOURCE_DIR}/src/eyebrow_exist.cpp
    ${CMAKE_SOURCE_DIR}/src/chin_exist.cpp
    ${CMAKE_SOURCE_DIR}/src/forehead_exist.cpp
    ${CMAKE_SOURCE_DIR}/src/occ_exist.cpp
    ${CMAKE_SOURCE_DIR}/src/fas_structure.cpp
    
    ${CMAKE_SOURCE_DIR}/src/tool.cpp
    ${CMAKE_SOURCE_DIR}/src/utils.cpp
    )

file(GLOB_RECURSE HEADER_LIST ${CMAKE_SOURCE_DIR}/*.h*)

add_library(${TARGET} SHARED
    ${HEADER_LIST}
    ${SOURCE_LIST}
)
    
set(ncnn_DIR "/home/storm/software/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
# set(ncnn_DIR "/home/storm/software/faceiqa/fosaferncnn/build-linux/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")


find_package(ncnn REQUIRED)
target_link_libraries(${TARGET} 
    ${OpenCV_LIBS}
    ncnn
    )

option(CAMERA "Enable camera" OFF)
if (CAMERA)
    message("Camera is Open")
    add_executable(testGZ 
        ${CMAKE_SOURCE_DIR}/examples/testCamera.cpp
        ${SOURCE_LIST}
        )
else()
    add_executable(testGZ 
        # ${CMAKE_SOURCE_DIR}/examples/test_picture.cc
     
        ${CMAKE_SOURCE_DIR}/examples/testNCNN.cpp
        # ${CMAKE_SOURCE_DIR}/examples/test3D.cpp
        ${SOURCE_LIST}
        )
endif()

target_link_libraries(testGZ 
    ${OpenCV_LIBS}
    ncnn
    )
