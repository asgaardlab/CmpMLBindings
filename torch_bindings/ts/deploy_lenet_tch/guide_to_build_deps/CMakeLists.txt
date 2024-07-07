cmake_minimum_required(VERSION 3.14)
project (torch-js)

set(PYTORCH_VERSION 1.10.2)
set(PYTORCH_VISION_VERSION 0.11.3)
execute_process(COMMAND cp -r $ENV{LIBTORCH_GPU} ${CMAKE_CURRENT_SOURCE_DIR}/build)
find_package(CUDA)

# Use C++14
set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/libtorch/)
find_package(Torch REQUIRED)

if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/torchvision")
  execute_process(
    COMMAND git clone --depth 1 --branch v${PYTORCH_VISION_VERSION} https://github.com/pytorch/vision
  )
  execute_process(
    COMMAND git -C "${CMAKE_CURRENT_BINARY_DIR}/vision/" apply "${CMAKE_CURRENT_SOURCE_DIR}/patches/0001-Remove-native-image-support.patch"
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=${CUDA_FOUND} -DCMAKE_PREFIX_PATH=${CMAKE_CURRENT_BINARY_DIR}/libtorch -S "${CMAKE_CURRENT_BINARY_DIR}/vision/" -B "${CMAKE_CURRENT_BINARY_DIR}/vision/build/"
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --build "${CMAKE_CURRENT_BINARY_DIR}/vision/build" --config Release
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} --install "${CMAKE_CURRENT_BINARY_DIR}/vision/build" --prefix "${CMAKE_CURRENT_BINARY_DIR}/torchvision"
  )
endif()

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH};${CMAKE_CURRENT_BINARY_DIR}/torchvision/)
find_package(TorchVision)

if(WIN32)
elseif(APPLE)
  set(CMAKE_BUILD_RPATH @loader_path;@executable_path)
else()
  set(CMAKE_BUILD_RPATH $ORIGIN)
endif()

include_directories(${CMAKE_JS_INC})

file(GLOB SOURCE_FILES "src/*.cc" "src/*.h")
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${CMAKE_JS_SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "" SUFFIX ".node")

# Include N-API wrappers
execute_process(COMMAND node -p "require('node-addon-api').include"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE NODE_ADDON_API_DIR
        )
string(REPLACE "\n" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
string(REPLACE "\"" "" NODE_ADDON_API_DIR ${NODE_ADDON_API_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${NODE_ADDON_API_DIR})
add_definitions(-DNAPI_VERSION=3)

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})

target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/torchvision/include")
target_link_libraries(${PROJECT_NAME} TorchVision::TorchVision)

target_link_libraries(${PROJECT_NAME} ${CMAKE_JS_LIB})

if(WIN32)
  # use move for windows since the libs are too large to be copied on github actions vm
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND powershell mv
          "\"${CMAKE_CURRENT_BINARY_DIR}/libtorch/lib/*.dll\""
          \"$<TARGET_FILE_DIR:torch-js>\")
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E rm
          "$<TARGET_FILE_DIR:torch-js>/cudnn_adv_train64_8.dll"
          "$<TARGET_FILE_DIR:torch-js>/cudnn_cnn_train64_8.dll"
          "$<TARGET_FILE_DIR:torch-js>/cudnn_ops_train64_8.dll")
else()
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_BINARY_DIR}/libtorch/lib"
          $<TARGET_FILE_DIR:torch-js>)
endif()

if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/torchvision/bin")
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_BINARY_DIR}/torchvision/bin"
          $<TARGET_FILE_DIR:torch-js>)
elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/torchvision/lib")
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_BINARY_DIR}/torchvision/lib"
          $<TARGET_FILE_DIR:torch-js>)
elseif(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/torchvision/lib64")
  add_custom_command(TARGET torch-js POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${CMAKE_CURRENT_BINARY_DIR}/torchvision/lib64"
          $<TARGET_FILE_DIR:torch-js>)
endif()
