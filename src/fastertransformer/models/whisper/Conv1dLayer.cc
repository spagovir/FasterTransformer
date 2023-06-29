#include "src/fastertransformer/models/whisper/Conv1dLayer.h"
#include "src/fastertransformer/utils/conv1d.h"
#include "src/fastertransformer/kernels/activation_kernels.h"
#include <cstddef>

namespace fastertransformer 
{   template<typename T1,typename T2>
    void Conv1dLayer<T1,T2>::forward(   
        Tensor input_tensor, // batch, width, input_channels
        Tensor output_tensor, // batch, width, output_channels
        DenseWeight<T1,T2> weights) 
    {   
        uint32_t batch    = input_tensor.shape[0]
    ;   uint32_t in_width = input_tensor.shape[1]
    ;   uint32_t in_chan  = input_tensor.shape[2]
    ;   FT_CHECK(batch == output_tensor.shape[0])
    ;   uint32_t out_width = output_tensor.shape[1]
    ;   FT_CHECK(out_width = (in_width + padding * 2 + stride - kernel_size) / stride)
    ;   uint32_t out_chan = output_tensor.shape[2]
    ;   conv1d
        (   output_tensor.getPtr<T2>()
        ,   input_tensor.getPtr<T1>()
        ,   weights.kernel
        ,   batch
        ,   in_width
        ,   padding
        ,   in_chan        
        ,   out_chan    
        ,   kernel_size
        ,   stride
        ,   cudnn_handle
        )
    ;   invokeAddBiasGeluV2<T2>
        (   output_tensor.getPtr<T2>()
        ,   weights.bias
        ,   nullptr
        ,   nullptr
        ,   batch * out_width
        ,   out_chan
        ,   stream_) 
    ;   }
    ;   template<typename T1,typename T2>
        void Conv1dLayer<T1,T2>::allocateBuffer() {}
    ;   template<typename T1,typename T2>
        void Conv1dLayer<T1,T2>::freeBuffer() {}
    ;   template class Conv1dLayer<float>
    ;
} ;

