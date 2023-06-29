#pragma once

#include "src/fastertransformer/models/whisper/WhisperEncoderWeight.h"
#include "src/fastertransformer/models/whisper/Conv1dLayer.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/models/whisper/WhisperEncoderLayer.h"
namespace fastertransformer {
template <typename T>
class WhisperEncoder : public BaseLayer {
    WhisperConfig config;
    public: 
    WhisperEncoder
        ( WhisperCudaContext        *context
        , bool                      is_free_buffer_after_forward
        , WhisperConfig             config
        )
    
    ;   void    allocateBuffer()    override
    ;   void    freeBuffer()        override
    ;   void    forward
        (   TensorMap   &input_tensors
        ,   TensorMap   &output_tensors
        ,   WhisperEncoderWeight<T>    weight)
    ;   ~WhisperEncoder()
    ;   std::vector<uint32_t> out_size(uint32_t batch, uint32_t seq)
    ;   protected:
        bool    is_free_buffer_after_forward_
    ;   bool    buffers_allocated_
    ;   T* conv1_out_buffer
    ;   T* residual
    ;   T* attn_mask
    ;   WhisperConfig           config_
    ;   WhisperCudaContext      *context_
    ;   void allocateBuffer(uint32_t batch, uint32_t in_seq)
    ;   Conv1dLayer<T>   conv1
    ;   Conv1dLayer<T>   conv2
    ;   WhisperEncoderLayer<T> attn_block
    ;   }
;   }