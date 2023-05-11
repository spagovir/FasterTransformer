#include "cudnn.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "cuda.h"
#include <type_traits>

namespace fastertransformer 
{   template<AllocatorType AT>
    class WhisperCudaContext 
    {   cublasLtHandle_t         cublasltHandle_ // inited
    ;   cublasHandle_t           cublasHandle_ // inited
    ;   std::mutex*              cublas_wrapper_mutex_ // ininted
    ;   cublasAlgoMap*           cublas_algo_map_ // inited
    ;   struct cudaDeviceProp    prop_ // inited
    ;   public:
        cudnnHandle_t       cudnn_handle
    ;   cudaStream_t        stream // set
    ;   cublasMMWrapper     cublas_wrapper // set
    ;   Allocator<AT>       allocator // set
    ;   IAllocator*         iallocator
    ;   WhisperCudaContext(cublasHandle_t _cublasHandle, cudaStream_t _stream)
        :   cublasHandle_(_cublasHandle)
        ,   stream(_stream)
        {   check_cuda_error(cublasLtCreate(&cublasltHandle_))
        ;   cublas_algo_map_            = new cublasAlgoMap(GEMM_CONFIG, "")
        ;   cublas_wrapper_mutex_       = new std::mutex()
        ;   check_cuda_error(cudaGetDeviceProperties(&prop_, 0))
        ;   cublasSetStream(cublasHandle_, stream)
        ;   allocator                   = Allocator<AT>()
        ;   if(std::is_base_of<IAllocator,Allocator<AT>>()) iallocator = &allocator
        ;   cublas_wrapper = 
            cublasMMWrapper
            (   cublasHandle_
            ,   cublasltHandle_
            ,   stream
            ,   cublas_algo_map_
            ,   cublas_wrapper_mutex_
            ,   &allocator
            )
        ;   check_cuda_error(cudnnCreate(&cudnn_handle))
        ;   }
    ;   ~WhisperCudaContext()
        {   delete cublas_wrapper_mutex_
        ;   delete cublas_algo_map_
        ;   cublas_wrapper_mutex_   = nullptr
        ;   cublas_algo_map_        = nullptr
        ;   }
    ;   }
;   }   ;