#pragma once

#include "cudnn.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasAlgoMap.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "cuda.h"
#include <type_traits>
#include <cstdint>

namespace fastertransformer 
{   
    class WhisperCudaContext 
    {   cublasLtHandle_t         cublasltHandle_ // inited
    ;   cublasHandle_t           cublasHandle_ // inited
    ;   std::mutex*              cublas_wrapper_mutex_ // ininted
    ;   cublasAlgoMap*           cublas_algo_map_ // inited
    ;   public:
    ;   struct cudaDeviceProp    prop_ // inited
    ;   cudnnHandle_t       cudnn_handle
    ;   cudaStream_t        stream_ // set
    ;   cublasMMWrapper*    cublas_wrapper // set
    ;   IAllocator*         iallocator
    ;   // WhisperCudaContext takes ownership of iallocator and will free it when destroyed
        WhisperCudaContext(cublasHandle_t _cublasHandle, cudaStream_t _stream, IAllocator *_iallocator)
        :   cublasHandle_(_cublasHandle)
        ,   stream_(_stream)
        {   check_cuda_error(cublasLtCreate(&cublasltHandle_))
        ;   cublas_algo_map_            =   new cublasAlgoMap(GEMM_CONFIG, "")
        ;   cublas_wrapper_mutex_       = new std::mutex()
        ;   check_cuda_error(cudaGetDeviceProperties(&prop_, 0))
        ;   cublasSetStream(cublasHandle_, stream_)
        ;   iallocator                   = _iallocator 
        ;   cublas_wrapper = new
            cublasMMWrapper
            (   cublasHandle_
            ,   cublasltHandle_
            ,   stream_
            ,   cublas_algo_map_
            ,   cublas_wrapper_mutex_
            ,   iallocator
            )
        ;   cudnnCreate(&cudnn_handle)
        ;   }
    ;   ~WhisperCudaContext()
        {   delete cublas_wrapper
        ;   delete cublas_wrapper_mutex_
        ;   delete cublas_algo_map_
        ;   delete iallocator
        ;   cublas_wrapper_mutex_   = nullptr
        ;   cublas_algo_map_        = nullptr
        ;   cublas_wrapper          = nullptr
        ;   iallocator = nullptr
        ;   }
    ;   }
;   }   ;