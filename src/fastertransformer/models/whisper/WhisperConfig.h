#pragma once
#include <stddef.h>

namespace fastertransformer
{
struct WhisperConfig { 
    size_t batch_size;
    size_t vocab_size;
    size_t num_mel_bins;
    size_t encoder_layers;
    size_t encoder_attention_heads;
    size_t decoder_layers;
    size_t decoder_attention_heads;
    size_t encoder_ffn_dim;
    size_t decoder_ffn_dim;
    size_t max_source_positions;
    size_t max_target_positions;
    size_t d_model;
    WhisperConfig() 
    : batch_size(1)
    , vocab_size(51865)
    , num_mel_bins(80)
    , encoder_layers(4) //6 on standard
    , encoder_attention_heads(6) //huggingface says 4
    , decoder_layers(4) //6 on standard
    , decoder_attention_heads(4)
    , decoder_ffn_dim(1536)
    , encoder_ffn_dim(1536)
    , max_source_positions(3000)
    , max_target_positions(2048)
    , d_model(384)
    {}
;   }
; }

