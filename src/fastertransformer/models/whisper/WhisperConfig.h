#pragma once
#include <stddef.h>
#include <cstdint>

namespace fastertransformer
{
#define LAYERNORM_EPS 0.00001
struct WhisperConfig { 
    uint32_t batch_size;
    uint32_t vocab_size;
    uint32_t num_mel_bins;
    uint32_t encoder_layers;
    uint32_t encoder_attention_heads;
    uint32_t decoder_layers;
    uint32_t decoder_attention_heads;
    uint32_t encoder_ffn_dim;
    uint32_t decoder_ffn_dim;
    uint32_t max_source_positions;
    uint32_t max_target_positions;
    uint32_t d_model;
    uint32_t max_beams;
    uint32_t eos_token_id;
    WhisperConfig() 
    : batch_size(1)
    , vocab_size(51865)
    , num_mel_bins(80)
    , encoder_layers(4) //6 on standard
    , encoder_attention_heads(6) //huggingface says 4
    , decoder_layers(4) //6 on standard
    , decoder_attention_heads(6)
    , decoder_ffn_dim(1536)
    , encoder_ffn_dim(1536)
    , max_source_positions(3000)
    , max_target_positions(448)
    , d_model(384)
    , max_beams(5)
    , eos_token_id(50256)
    {}
;   }
; }

