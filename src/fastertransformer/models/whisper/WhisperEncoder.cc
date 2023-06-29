#include "src/fastertransformer/models/whisper/WhisperEncoder.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/models/whisper/WhisperKernels.h"
#include <cstddef>
#include <memory>
#include <type_traits>

namespace fastertransformer
{   template<typename T>
    WhisperEncoder<T>::WhisperEncoder
    ( WhisperCudaContext        *context
    , bool                      is_free_buffer_after_forward
    , WhisperConfig             config
    )
    :   BaseLayer
        (   context->stream_
        ,   context->cublas_wrapper
        ,   context->iallocator
        ,   is_free_buffer_after_forward)
    ,   context_(context)
    ,   buffers_allocated_(false)
    ,   config_(config)

    ,   conv1(Conv1dLayer<T>
        (   context->stream_
        ,   context->cublas_wrapper
        ,   context->iallocator
        ,   1 
        ,   1
        ,   3
        ,   context->cudnn_handle ))
    ,   conv2
        (   context->stream_
        ,   context->cublas_wrapper
        ,   context->iallocator
        ,   2 //2
        ,   1
        ,   3
        ,   context_->cudnn_handle)
    ,   attn_block(
        config,
        context,
        is_free_buffer_after_forward
        )
    {   if(! is_free_buffer_after_forward) allocateBuffer()
    ;} 
    
    template<typename T>
        void WhisperEncoder<T>::allocateBuffer() {
            allocateBuffer(config_.batch_size, config_.max_source_positions);
        } ;
    template<typename T>
    void WhisperEncoder<T>::allocateBuffer(uint32_t batch, uint32_t in_seq)
    {   conv1_out_buffer = 
            (T*) context_->iallocator->malloc
                (   sizeof(T) 
                *   in_seq 
                *   batch
                *   config_.d_model)
    ;   residual = 
            (T*) context_->iallocator->malloc
            (   sizeof(T)
            *   ((in_seq + 1)/2)
            *   batch
            *   config_.d_model
            )
    ;   buffers_allocated_ = true
    ;   }
;   template<typename T>
    void WhisperEncoder<T>::freeBuffer()
    {   context_->iallocator->free((void**) &conv1_out_buffer)
    ;   context_->iallocator->free((void**) &residual)
    ;   conv1_out_buffer = nullptr
    ;   residual = nullptr
    ;   buffers_allocated_ = false
    ;   }
;   template<typename T>
    void WhisperEncoder<T>::forward
    (   TensorMap   &input_tensors
    ,   TensorMap   &output_tensors
    ,   WhisperEncoderWeight<T> weight)
    // input_tensors:
    //      "encoder_input" : [batch, max_sequence_length, num_mel_bins]
    // output_tensors:
    //      "encoder_output" : [batch, max_sequence_length / 2 + 1, d_model]
    {   Tensor &in_tensor = input_tensors.at("encoder_input")
    ;   Tensor &out_tensor = output_tensors.at("encoder_output")
    ;   uint32_t batch = in_tensor.shape[0]
    ;   uint32_t seq = in_tensor.shape[1]
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
        (   in_tensor
        ,   conv1_out_tensor
        ,   weight.conv1)
    ;   Tensor conv2_out_tensor = Tensor
        (   MEMORY_GPU
        ,   getTensorType<T>()
        ,   {batch, (seq+1)/2, config.d_model} //(seq+1)/2
        ,   residual
        )
    ;   conv2.forward
        (   conv1_out_tensor
        ,   conv2_out_tensor
        ,   weight.conv2)
    
    ;   invokeEmbedSinusoid(conv2_out_tensor, context_->stream_)   
    ;   sync_check_cuda_error()
    /* 
    ;   cudaMemcpy
        (   out_tensor.getPtr<void>()
        ,   conv2_out_tensor.getPtr<void>()
        ,   conv2_out_tensor.sizeBytes()
        ,   cudaMemcpyDefault)
    */
    ;   for(uint32_t i = 0; i < config.encoder_layers; i++) //config_.encoder_layers; i++)
        {   attn_block.forward
            (   conv2_out_tensor
            ,   weight.layers[i]
            ,   (i == (config_.encoder_layers - 1)) ? weight.layernorm : weight.layers[i+1].layernorm1
            ,   out_tensor.getPtr<T>()
            ,   i == 0)
        ;   }
    /*
    ;   cudaMemcpy(
            out_tensor.getPtr<void>(), 
            residual, 
            conv2_out_tensor.sizeBytes(),
            cudaMemcpyDefault)
    */
    
    ;   if(is_free_buffer_after_forward_)   freeBuffer()
    ;   }
    template<typename T> 
    std::vector<uint32_t> WhisperEncoder<T>::out_size(uint32_t batch, uint32_t seq)
    {   return {batch, (seq+1)/2, config.d_model};} //(seq+1)/2

    template<typename T>
    WhisperEncoder<T>::~WhisperEncoder()
    {   if(buffers_allocated_)  freeBuffer()
    ;   
    }
;   template class WhisperEncoder<float>
;   }

