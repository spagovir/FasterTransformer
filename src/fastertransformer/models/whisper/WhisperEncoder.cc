#include "src/fastertransformer/models/whisper/WhisperEncoder.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cstddef>
#include <type_traits>

namespace fastertransformer
{   template<typename T, AllocatorType AT>
    WhisperEncoder<T,AT>::WhisperEncoder
    ( WhisperCudaContext<AT>    &context
    , bool                      is_free_buffer_after_forward
    , WhisperConfig             config
    )
    :   context_(context)
    ,   is_free_buffer_after_forward_(is_free_buffer_after_forward)
    ,   buffers_allocated_(false)
    ,   config_(config)
    {   if(! is_free_buffer_after_forward) allocateBuffer()
    ;   conv1 = Conv1dLayer<T>
        (   context_.stream
        ,   context_.cublas_wrapper
        ,   context_.iallocator
        ,   1
        ,   1
        ,   3
        ,   context_.cudnn_handle )
    ;   conv2 = Conv1dLayer<T>
        (   context_.stream
        ,   context_.cublas_wrapper
        ,   context_.iallocator
        ,   2
        ,   1
        ,   3
        ,   context_.cudnn_handle)
    ;} 
    
    template<typename T, AllocatorType AT>
        void WhisperEncoder<T,AT>::allocateBuffer() {
            allocateBuffer(config_.batch_size, config_.max_source_positions);
        } ;
    template<typename T, AllocatorType AT>
    void WhisperEncoder<T,AT>::allocateBuffer(size_t batch, size_t in_seq)
    {   conv1_out_buffer = 
            (T*) context_.iallocator.malloc
                (   sizeof(T) 
                *   in_seq
                *   batch
                *   config_.d_model)
    ;   conv2_out_buffer = 
            (T*) context_.iallocator->malloc
            (   sizeof(T)
            *   (in_seq + 1)/2
            *   batch
            *   config_.d_model
            )
    ;   buffers_allocated_ = true
    ;   }
;   template<typename T, AllocatorType AT>
    void WhisperEncoder<T,AT>::freeBuffer()
    {   context_.iallocator.free(conv1_out_buffer)
    ;   context_.iallocator.free(conv2_out_buffer)
    ;   conv1_out_buffer = nullptr
    ;   conv2_out_buffer = nullptr
    ;   buffers_allocated_ = false
    ;   }
;   template<typename T, AllocatorType AT>
    void WhisperEncoder<T,AT>::forward
    (   TensorMap   &input_tensors
    ,   TensorMap   &output_tensors
    ,   WhisperEncoderWeight<T> weight)
    // input_tensors:
    //      "encoder_input" : [batch, max_sequence_length, num_mel_bins]
    //      "input_lengths" : [batch]
    // output_tensors:
    //      "encoder_output" : [batch, seq/2, d_model]
    {   Tensor &in_tensor = input_tensors.at("encoder_input")
    ;   Tensor &out_tensor = output_tensors.at("encoder_output")
    ;   size_t batch = in_tensor.shape[0]
    ;   size_t seq = in_tensor.shape[1]
    ;   FT_CHECK(config.num_mel_bins == in_tensor.shape[2])
    ;   if(!buffers_allocated_) allocateBuffer(batch,seq)
    ;   else
        {   FT_CHECK(batch == config.batch_size)
        ;   FT_CHECK(seq == config.max_source_positions)
        ;   }
    ;   FT_CHECK(batch == out_tensor.shape[0])
    ;   FT_CHECK((seq+1)/2 == out_tensor.shape[1])
    ;   FT_CHECK(config.d_model == out_tensor.shape[2])
    ;   Tensor conv1_out_tensor = Tensor
            (   MEMORY_GPU
            ,   getTensorType<T>()
            ,   {batch, seq, config.d_model}
            ,   conv1_out_buffer)
    ;   conv1.forward
        (   in_tensor.getPtr<T>()
        ,   conv1_out_tensor
        ,   weight.conv1)
    ;   Tensor conv2_out_tensor = Tensor
        (   MEMORY_GPU
        ,   getTensorType<T>()
        ,   {batch, (seq + 1)/2, config.d_model}
        ,   conv2_out_buffer
        )
    ;   conv2.forward
        (   conv1_out_tensor
        ,   conv2_out_tensor
        ,   weight.conv2)
    ;   memcpy
        (   out_tensor.getPtr<void>()
        ,   conv2_out_tensor.getPtr<void>()
        ,   conv2_out_tensor.sizeBytes())
    ;   if(is_free_buffer_after_forward_)   freeBuffer()
    ;   }
    template<typename T, AllocatorType AT>
    WhisperEncoder<T,AT>::~WhisperEncoder<T, AT>()
    {   if(buffers_allocated_)  freeBuffer()
    ;   
    }
;   }