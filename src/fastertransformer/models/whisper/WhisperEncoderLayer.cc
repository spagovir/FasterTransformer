#include "src/fastertransformer/models/whisper/WhisperEncoderLayer.h"
#include "src/fastertransformer/models/whisper/WhisperKernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/utils/Tensor.h"

namespace fastertransformer 
{
template<typename T>
WhisperEncoderLayer<T>::WhisperEncoderLayer( WhisperConfig      config
                                           , WhisperCudaContext *context
                                           , bool is_free_buffer_after_forward)
    :   BaseLayer( context->stream_
                 , context->cublas_wrapper
                 , context->iallocator
                 , is_free_buffer_after_forward )
    ,   self_attn{ config.batch_size
                 , (config.max_source_positions + 1)/2
                 , config.encoder_attention_heads
                 , config.d_model
                 , 1.0
                 , context->stream_
                 , context->cublas_wrapper
                 , context->iallocator
                 , is_free_buffer_after_forward
                 }
    ,   ffn      { config.batch_size
                 , (config.max_source_positions +1)/2
                 , 1
                 , config.d_model
                 , 1
                 , config.encoder_ffn_dim
                 , context->stream_
                 , context->cublas_wrapper
                 , context->iallocator
                 , is_free_buffer_after_forward}
    ,   max_batch(config.batch_size)
    ,   max_seq(config.max_source_positions / 2 + 1)
    ,   d_model(config.d_model)
    {
        if(!is_free_buffer_after_forward) allocateBuffer();
    }
    
template<typename T> 
WhisperEncoderLayer<T>::~WhisperEncoderLayer()
{
    if(buffers_allocated) freeBuffer();
}

template<typename T> 
void WhisperEncoderLayer<T>::allocateBuffer(size_t batch, size_t seq)
{
    size_t size = batch * seq * d_model * sizeof(T);
    pre_buffer = (T*) allocator_->malloc(size);
    attn_mask = (T*) allocator_->malloc(batch * seq * seq * sizeof(T));
    invokeCausalAttnMask(attn_mask, batch, seq, stream_);
    buffers_allocated = true;
}

template<typename T> 
void WhisperEncoderLayer<T>::allocateBuffer()
{
    allocateBuffer(max_batch,max_seq);
}

template<typename T>
void WhisperEncoderLayer<T>::freeBuffer()
{
    allocator_->free((void**)&pre_buffer);
    allocator_->free((void**)&attn_mask);
    buffers_allocated = false;
}

template<typename T> 
void WhisperEncoderLayer<T>::forward(Tensor residual, WhisperEncoderLayerWeight<T> weight)
{
    size_t batch;
    size_t seq;
    T* residualPtr = residual.getPtr<T>();
    if(is_free_buffer_after_forward_)
    {
        batch = residual.shape[0];
        seq = residual.shape[1];
    }
    else {
        batch = max_batch;
        seq = max_seq;
    }

    invokeGeneralLayerNorm(
        pre_buffer,
        residualPtr,
        weight.layernorm1.gamma,
        weight.layernorm1.beta,
        LAYERNORM_EPS,
        batch * seq,
        d_model,
        nullptr,
        0,
        stream_
    );
    Tensor attn_queries = Tensor(
        MemoryType::MEMORY_GPU, 
        getTensorType<T>(),
        {batch * seq, d_model},
        (void*) pre_buffer
        );
    Tensor attn_mask_tensor = Tensor(
        MemoryType::MEMORY_GPU,
        getTensorType<T>(),
        {batch, 1, seq, seq},
        (void*) attn_mask
    );
    TensorMap attn_inputs = TensorMap(
        {
            {"input_queries", attn_queries},
            {"attention_mask", attn_mask_tensor}
        }
    );
    //this is wrong we need an add not overwritte version for attn. 
    TensorMap attn_outputs = TensorMap(
        {{"hidden_features", Tensor(
            MemoryType::MEMORY_GPU,
            getTensorType<T>(),
            {batch * seq, d_model},
            (void*) residualPtr
        )}}
    );
    invokeGeneralLayerNorm(
        pre_buffer,
        residualPtr,
        weight.layernorm2.gamma,
        weight.layernorm2.beta,
        LAYERNORM_EPS,
        batch*seq,
        d_model,
        nullptr,
        0,
        stream_
    );
    
}

template class WhisperEncoderLayer<float>;
}