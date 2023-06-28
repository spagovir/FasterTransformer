#include "src/fastertransformer/th_op/whisper/FTWhisperDecoder.h"
#include "src/fastertransformer/th_op/whisper/VectorReader.h"
#include <ATen/core/TensorBody.h>
#include <vector>

namespace torch_ext
{
    FTWhisperDecoder::FTWhisperDecoder(std::vector<th::Tensor> weights)
    {
        /*
        weights:
            0: token_embed
            1: pos_embed
            2 + layer * 24 for layer in layers:
                +0: pre_self_attn_ln 
                +2: self_attn
                +9: pre_cross_attn_ln
                +11: cross_attn
                +18: pre_ffn_ln
                +20: ffn
            where:
                attn:
                    +0 key.weight
                    +1 query.weight
                    +2 query.bias
                    +3 value.weight
                    +4 value.bias
                    +5 output.weight
                    +6 output.bias
                ln:
                    +0 gamma
                    +1 beta
                ffn:
                    +0 ntermediate.weight
                    +1 intermediate.bias
                    +2 output.weight
                    +3 output.bias
                
        */
        context = new ft::WhisperCudaContext
        (   at::cuda::getCurrentCUDABlasHandle()
        ,   at::cuda::getCurrentCUDAStream()
        ,   new ft::Allocator<ft::AllocatorType::TH>()
        );   
        ft::WhisperConfig config;
        decoder = new ft::WhisperContextDecoder<float>(context, config,true);
        VectorReader<float> reader(&weights);
        weight_.token_embed = reader.read();
        weight_.pos_embed = reader.read();
        for(auto layer = weight_.layers.begin();
            layer != weight_.layers.end();
            layer++)
        {
            layer->pre_self_attn_layernorm.gamma = reader.read();
            layer->pre_self_attn_layernorm.beta = reader.read();

            layer->self_attn.key_weight.kernel = reader.read();
            layer->self_attn.query_weight.kernel = reader.read();
            layer->self_attn.query_weight.bias = reader.read();
            layer->self_attn.value_weight.kernel = reader.read();
            layer->self_attn.value_weight.bias = reader.read();
            layer->self_attn.attention_output_weight.kernel = reader.read();
            layer->self_attn.attention_output_weight.bias = reader.read();

            layer->pre_cross_attn_layernorm.gamma = reader.read();
            layer->pre_cross_attn_layernorm.beta = reader.read();

            layer->cross_attn.key_weight.kernel = reader.read();
            layer->cross_attn.query_weight.kernel = reader.read();
            layer->cross_attn.query_weight.bias = reader.read();
            layer->cross_attn.value_weight.kernel = reader.read();
            layer->cross_attn.value_weight.bias = reader.read();
            layer->cross_attn.attention_output_weight.kernel = reader.read();
            layer->cross_attn.attention_output_weight.bias = reader.read();

            layer->pre_ffn_layernorm.gamma = reader.read();
            layer->pre_ffn_layernorm.beta = reader.read();

            layer->ffn.intermediate_weight.kernel = reader.read();
            layer->ffn.intermediate_weight.bias = reader.read();
            layer->ffn.output_weight.kernel = reader.read();
            layer->ffn.output_weight.bias = reader.read();
        }

        context->cublas_wrapper->setFP32GemmConfig();
    }

    th::Tensor FTWhisperDecoder::forward(th::Tensor encoder_output, th::Tensor inputs, th::Tensor input_lengths, float temperature = 0)
    {

    }
}