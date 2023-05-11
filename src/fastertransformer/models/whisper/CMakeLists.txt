cmake_minimum_required(VERSION 3.8)

add_library(WhisperEncoder STATIC WhisperEncoder.cc Conv1dLayer.cc)
set_property(TARGET WhisperEncoder PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET WhisperEncoder PROPERTY CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(WhisperEncoder PUBLIC 
    -lcudart "${CUDNN_LIBRARY_PATH}"
    activation_kernels 
    cublasMMWrapper
    cuda_utils
    layernorm_kernels
    )
