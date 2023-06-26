#include "src/fastertransformer/models/whisper/WhisperContextDecoder.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/models/whisper/WhisperKernels.h"

namespace fastertransformer
{
    template<typename T>
    void WhisperForConditionalGeneration<T>::forward(
        TensorMap &output_tensors,
        TensorMap &input_tensors,
        WhisperEncoderWeight<T> encoder_weight,
        WhisperDecoderWeight<T> decoder_weight
    )
    /*
    input_tensors:
        "encoder_outputs" : [batch, seq, d_model]
        "decoder_inputs" : size_t[batch,target_seq]
        "decoder_input_lengths" : size_t[batch]
        "top_k": Optional size_t[1]
        "top_p": Optional size_t[1]
        "beam_search_diversity_rate": Optional size_t[1]
        "temperature": Optional [1]
        "beam_width": Optional size_t[1]
        "input_lengths:" Optional size_t[batch]
    output_tensors:
        "output_ids" : size_t[batch, beam, max_target_positions]
        "output_logprobs" : [batch, beam, max_target_positions, vocab_size]
    */
    {
        size_t batch = input_tensors.at("encoder_outputs").shape[0];
        size_t seq = input_tensors.at("encoder_inputs").shape[1];
        size_t beam = input_tensors.at("beam_width").getPtr<size_t>()[0];
        Tensor encoderOutputTensor = input_tensors.at("encoder_outputs");
        invokeRepeat<T>(decoder_input_buf, encoderOutputTensor, 1, config_.max_beams, context_->stream_);
        size_t out_seq = output_tensors.at("output_ids").shape[1];
        size_t output_beams_lda = batch * config_.max_beams * config_.vocab_size;
        // output_id_beams : seq x batch x beam x vocab_size
        for(int idx = 0; idx + 1 < out_seq; idx ++)
        {
            Tensor decoder_output_logits = 
                Tensor(
                    MEMORY_GPU,
                    getTensorType<T>(),
                    {batch, config_.max_beams, config_.vocab_size},
                    logits_buffer);
            size_t size_per_head = config_.d_model / config_.decoder_attention_heads;
            size_t x = 16 / sizeof(T);
            size_t size_per_head_x = size_per_head/x;
            TensorMap decoder_outputs = 
                TensorMap(
                    {
                        {
                            "output_logits",
                            decoder_output_logits
                        },
                        {   
                            "cache_indirection",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<size_t>(),
                                {batch, config_.max_beams, config_.max_target_positions},
                                cache_indir)
                        },
                        {
                            "self_key_cache",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {config_.decoder_layers, batch * config_.max_beams, config_.decoder_attention_heads, size_per_head_x, out_seq, x},
                                self_key_cache
                            )
                        },
                        {   
                            "self_value_cache",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {config_.decoder_layers, batch * config_.max_beams, config_.decoder_attention_heads, out_seq, size_per_head},
                                self_value_cache
                            )
                        },
                        {
                            "cross_key_cache",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {config_.decoder_layers, batch * config_.max_beams, config_.decoder_attention_heads, size_per_head_x, out_seq, x},
                                cross_key_cache
                            )
                        },
                        {
                            "cross_value_cache",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {config_.decoder_layers, batch * config_.max_beams, config_.decoder_attention_heads, out_seq, size_per_head},
                                cross_value_cache
                            )
                        }
                    }
                );
            TensorMap decoder_inputs =
                TensorMap(
                    {
                        {
                            "encoder_outputs",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {batch * beam, seq, config_.d_model},
                                decoder_input_buf
                            )
                        },
                        {

                        }
                    }
                );
        }

    }
    template class WhisperForConditionalGeneration<float>;
}