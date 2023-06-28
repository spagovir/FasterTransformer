cmake_minimum_required(VERSION 3.8)

add_library(WhisperEncoder STATIC WhisperEncoder.h WhisperEncoder.cc Conv1dLayer.cc WhisperKernels.cu WhisperEncoderLayer.cc)
set_property(TARGET WhisperEncoder PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET WhisperEncoder PROPERTY CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
add_executable(Conv1dTests Conv1dTests.cc)
target_link_libraries(WhisperEncoder PUBLIC 
    -lcudart "${CUDNN_LIBRARY_PATH}"
    activation_kernels 
    cublasMMWrapper
    cuda_utils
    layernorm_kernels
    )
target_link_libraries(Conv1dTests PUBLIC 
    -lcublasLt
    -lcublas
    -lcudart "${CUDNN_LIBRARY_PATH}"
    activation_kernels 
    cublasMMWrapper
    cuda_utils
    layernorm_kernels
    )

add_library(WhisperDecoder STATIC WhisperDecoder.cc WhisperContextDecoder.cc)
target_link_libraries(WhisperDecoder PUBLIC
    -lcudart "${CUDNN_LIBRARY_PATH}"
    activation_kernels
    cublasMMWrapper
    cuda_utils
    layernorm_kernels)

