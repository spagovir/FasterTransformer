#include "src/fastertransformer/models/whisper/Conv1dLayer.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/layers/DenseWeight.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_utils.h"
namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext
{

void conv1d_from_th
(   size_t  in_chan
,   size_t  out_chan
,   size_t  ker_size 
,   size_t  stride
,   size_t  padding
,   th::Tensor& const   weight 
,   th::Tensor& const   bias    
,   ft::WhisperCudaContext<ft::AllocatorType::TH>&  const   context
,   ft::Conv1dLayer<float>*    ft_layer
,   ft::DenseWeight<float>*    ft_weight
)
{   // FT_CHECK(weights.size(0) == out_chan)
;   // FT_CHECK(weights.size(1) == in_chan)
;   // FT_CHECK(weights.size(2) == ker_size)
;   ft_layer = new ft::Conv1dLayer<float>
    (   context.stream
    ,   &context.cublas_wrapper
    ,   &context.allocator   
    ,   stride
    ,   padding
    ,   ker_size
    ,   context.cudnn_handle
    )
;   ft_weight->kernel   = get_ptr<float>(weight)
;   ft_weight->bias     = get_ptr<float>(bias)
;   }
}