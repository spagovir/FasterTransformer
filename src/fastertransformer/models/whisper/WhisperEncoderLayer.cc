#include "src/fastertransformer/models/whisper/WhisperEncoderLayer.h"
#include "src/fastertransformer/models/whisper/WhisperEncoderLayerWeight.h"
#include "src/fastertransformer/models/whisper/WhisperKernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"

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
                 , config.d_model / config.encoder_attention_heads
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
    ,   buffers_allocated(false)
    {
        if(!is_free_buffer_after_forward) allocateBuffer();
    }
    
template<typename T> 
WhisperEncoderLayer<T>::~WhisperEncoderLayer()
{
    if(buffers_allocated) freeBuffer();
}

template<typename T> 
void WhisperEncoderLayer<T>::allocateBuffer(uint32_t batch, uint32_t seq)
{
    uint32_t size = batch * seq * d_model * sizeof(T);
    attn_mask = (T*) allocator_->malloc(batch * seq * seq * sizeof(T));
    float f = 1.0f;
    k_bias = (T*) allocator_->malloc(d_model * sizeof(T));
    //invokeCausalAttnMask(attn_mask, batch, seq, stream_);
    invokeEncoderAttnMask(attn_mask, batch, seq, stream_);
    sync_check_cuda_error();
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
    allocator_->free((void**)&attn_mask);
    allocator_->free((void**) &k_bias);
    buffers_allocated = false;
}

template<typename T> 
void WhisperEncoderLayer<T>::forward(Tensor residual, WhisperEncoderLayerWeight<T> weight, LayerNormWeight<T> next_ln_weight, T* lno_buffer, bool is_first)
{
    uint32_t batch;
    uint32_t seq;
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
    if(!buffers_allocated) allocateBuffer(batch,seq);
    if(weight.self_attn.key_weight.bias == nullptr) weight.self_attn.key_weight.bias = k_bias;
    if(is_first)
        invokeGeneralLayerNorm(
            lno_buffer,
            residualPtr,
            weight.layernorm1.gamma,
            weight.layernorm1.beta,
            LAYERNORM_EPS,
            batch * seq,
            d_model,
            nullptr,
            0,
            stream_
        ), sync_check_cuda_error();
    
    Tensor attn_queries = Tensor(
        MemoryType::MEMORY_GPU, 
        getTensorType<T>(),
        {batch * seq, d_model},
        (void*) lno_buffer
        );
    Tensor attn_mask_tensor = Tensor(
        MemoryType::MEMORY_GPU,
        getTensorType<T>(),
        {batch, 1, seq, seq},
        (void*) attn_mask
    );
    TensorMap attn_inputs = TensorMap(
        {
            {"input_query", attn_queries},
            {"attention_mask", attn_mask_tensor}
        }
    );
    TensorMap attn_outputs = TensorMap(
        {{"hidden_features", Tensor(
            MemoryType::MEMORY_GPU,
            getTensorType<T>(),
            {batch * seq, d_model},
            (void*) lno_buffer
        )}}
    );
    


    self_attn.forward(&attn_outputs, &attn_inputs, &weight.self_attn);
    sync_check_cuda_error();
    invokeGeneralAddBiasResidualPreLayerNorm(
        residualPtr,
        lno_buffer,
        lno_buffer,
        residualPtr,
        (T*) nullptr,
        weight.layernorm2.gamma,
        weight.layernorm2.beta,
        weight.self_attn.attention_output_weight.bias,
        LAYERNORM_EPS,
        batch * seq,
        d_model,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0,
        stream_
    );
    sync_check_cuda_error();
    TensorMap ffn_input = TensorMap(
        {{"ffn_input", Tensor(MemoryType::MEMORY_GPU,
        getTensorType<T>(),
        {batch * seq, d_model},
        lno_buffer)}}
    );
    TensorMap ffn_output = TensorMap(
        {{"ffn_output", Tensor(
            MemoryType::MEMORY_GPU,
            getTensorType<T>(),
            {batch * seq, d_model},
            lno_buffer
        )}}
    );
    ffn.forward(&ffn_output,&ffn_input,&weight.ffn);
    sync_check_cuda_error();
    invokeGeneralAddBiasResidualPreLayerNorm(
        residualPtr,
        lno_buffer,
        lno_buffer,
        residualPtr,
        (T*) nullptr,
        next_ln_weight.gamma,
        next_ln_weight.beta,
        weight.ffn.output_weight.bias,
        LAYERNORM_EPS,
        batch * seq,
        d_model,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        0,
        stream_
    );
    sync_check_cuda_error();
    if(is_free_buffer_after_forward_) freeBuffer();
}

template class WhisperEncoderLayer<float>;
}