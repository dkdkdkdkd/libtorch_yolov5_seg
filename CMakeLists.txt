cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(deeplab_example)

# OpenCV 찾기
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# LibTorch 경로 설정 (설치 경로에 따라 수정이 필요합니다)
list(APPEND CMAKE_PREFIX_PATH "/home/jiho/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_executable(run_cpu run_cpu.cc)
add_executable(run_gpu run_gpu.cc)
# OpenCV와 LibTorch 라이브러리를 deeplab_example 타겟에 연결
target_link_libraries(run_cpu ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(run_gpu ${TORCH_LIBRARIES} ${OpenCV_LIBS})

