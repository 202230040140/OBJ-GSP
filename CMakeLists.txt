cmake_minimum_required(VERSION 3.18)

project(OBJ_GSP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED CONDA_PREFIX)
  set(CONDA_PREFIX "$ENV{CONDA_PREFIX}")
endif()

if(NOT CONDA_PREFIX)
  message(FATAL_ERROR "Set CONDA_PREFIX to the obj-gsp-cpp environment path.")
endif()

set(OpenCV_DIR "${CONDA_PREFIX}/Library/cmake/x64/vc15/lib")
find_package(OpenCV REQUIRED)
find_package(OpenMP)

add_executable(obj_gsp
  main.cpp
  Debugger/ColorMap.cpp
  Debugger/ErrorController.cpp
  Debugger/ImageDebugger.cpp
  Debugger/TimeCalculator.cpp
  Feature/FeatureController.cpp
  Feature/ImageData.cpp
  Feature/MultiImage.cpp
  Mesh/MeshGrid.cpp
  Mesh/MeshOptimization.cpp
  Stitching/APAP_Stitching.cpp
  Stitching/NISwGSP_Stitching.cpp
  Stitching/Parameter.cpp
  Util/Blending.cpp
  Util/EdgeDetection.cpp
  Util/InputParser.cpp
  Util/RuntimeConfig.cpp
  Util/Statistics.cpp
  Util/Thin.cpp
  Util/Transform.cpp
  lsd.cpp
)

target_include_directories(obj_gsp PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CONDA_PREFIX}/Library/include"
  "${CONDA_PREFIX}/Library/include/eigen3"
)

target_link_directories(obj_gsp PRIVATE
  "${CONDA_PREFIX}/Library/lib"
)

target_link_libraries(obj_gsp PRIVATE
  ${OpenCV_LIBS}
  "${CONDA_PREFIX}/Library/lib/gflags.lib"
  "${CONDA_PREFIX}/Library/bin/vl.lib"
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(obj_gsp PRIVATE OpenMP::OpenMP_CXX)
endif()

file(GLOB OBJ_GSP_RUNTIME_DLLS
  "${CONDA_PREFIX}/Library/bin/*.dll"
  "${CONDA_PREFIX}/*.dll"
)

add_custom_command(TARGET obj_gsp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${OBJ_GSP_RUNTIME_DLLS}
          $<TARGET_FILE_DIR:obj_gsp>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CONDA_PREFIX}/Library/bin/deflate.dll"
          "$<TARGET_FILE_DIR:obj_gsp>/libdeflate.dll"
)

target_compile_definitions(obj_gsp PRIVATE
  _CRT_SECURE_NO_WARNINGS
  NOMINMAX
)
