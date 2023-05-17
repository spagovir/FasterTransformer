#pragma once
#include "src/fastertransformer/layers/DenseWeight.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "cudnn.h"
namespace fastertransformer {

template<typename T1, typename T2=T1>
class Conv1dLayer : public BaseLayer 
    { size_t  stride
    ; size_t  padding
    ; size_t  kernel_size
    ; cudnnHandle_t& cudnn_handle
    ; public : 
       Conv1dLayer
      ( cudaStream_t     stream
      , cublasMMWrapper* cublas_wrapper
      , IAllocator*      allocator
      , size_t           _stride
      , size_t           _padding
      , size_t           _kernel_size
      , cudnnHandle_t&   _cudnn_handle
      ) 
      : BaseLayer
        ( stream
        , cublas_wrapper
        , allocator
        , true
        )
      , stride(_stride)
      , padding(_padding)
      , kernel_size(_kernel_size)
      , cudnn_handle(_cudnn_handle)
      {}
    ; void forward(Tensor input_tensor, Tensor output_tensor, DenseWeight<T1,T2> weight)
    ; void allocateBuffer() override
    ; void freeBuffer() override
    ; }
; }