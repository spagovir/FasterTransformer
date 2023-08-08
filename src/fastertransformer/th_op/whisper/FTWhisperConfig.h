#include "src/fastertransformer/th_op/th_utils.h"
#include <ATen/core/ivalue.h>
#include <cstdint>
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext 
{
    struct FTWhisperConfig:public th::CustomClassHolder
    {
        ft::WhisperConfig config;
        FTWhisperConfig(
            int64_t batch_size = 1,
            int64_t vocab_size = 51865,
            int64_t num_mel_bins = 80,
            int64_t encoder_layers = 6,
            int64_t encoder_attention_heads = 4,
            int64_t decoder_layers = 6,
            int64_t decoder_attention_heads = 4,
            int64_t decoder_ffn_dim = 1536,
            int64_t encoder_ffn_dim = 1536, 
            int64_t max_source_positions = 3000,
            int64_t max_target_positions = 2048,
            int64_t d_model = 384, 
            int64_t max_beams = 5,
            int64_t eos_token_id = 50256
        ):
        config
        (
            {
                .batch_size=static_cast<uint32_t>(batch_size),
                .vocab_size=static_cast<uint32_t>(vocab_size),
                .num_mel_bins=static_cast<uint32_t>(num_mel_bins),
                .encoder_layers=static_cast<uint32_t>(encoder_layers),
                .encoder_attention_heads=static_cast<uint32_t>(encoder_attention_heads),
                .decoder_layers=static_cast<uint32_t>(decoder_layers),
                .decoder_attention_heads=static_cast<uint32_t>(decoder_attention_heads),
                .encoder_ffn_dim=static_cast<uint32_t>(encoder_ffn_dim),
                .decoder_ffn_dim=static_cast<uint32_t>(decoder_ffn_dim),
                .max_source_positions=static_cast<uint32_t>(max_source_positions),
                .max_target_positions=static_cast<uint32_t>(max_target_positions),
                .d_model=static_cast<uint32_t>(d_model),
                .max_beams=static_cast<uint32_t>(max_beams),
                .eos_token_id=static_cast<uint32_t>(eos_token_id)
            }
        )
        {};
    };
}
