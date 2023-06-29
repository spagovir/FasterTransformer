#include "src/fastertransformer/th_op/whisper/FTWhisperDecoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/th_op/whisper/VectorReader.h"
#include "src/fastertransformer/models/whisper/WhisperKernels.h"
#include <ATen/core/TensorBody.h>
#include <torch/types.h>
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
        decoder = new ft::WhisperContextDecoder<float>(context, config,true);
        VectorReader<float> reader(&weights);
        weight_.token_embed = reader.read();
        weight_.pos_embed = reader.read();
        for(size_t i = 0; i<config.decoder_layers; i++)
        {
            ft::WhisperDecoderLayerWeight<float> layer;
            layer.pre_self_attn_layernorm.gamma = reader.read();
            layer.pre_self_attn_layernorm.beta = reader.read();

            layer.self_attn.key_weight.kernel = reader.read();
            layer.self_attn.key_weight.bias = reader.read();
            layer.self_attn.query_weight.kernel = reader.read();
            layer.self_attn.query_weight.bias = reader.read();
            layer.self_attn.value_weight.kernel = reader.read();
            layer.self_attn.value_weight.bias = reader.read();
            layer.self_attn.attention_output_weight.kernel = reader.read();
            layer.self_attn.attention_output_weight.bias = reader.read();

            layer.pre_cross_attn_layernorm.gamma = reader.read();
            layer.pre_cross_attn_layernorm.beta = reader.read();

            layer.cross_attn.key_weight.kernel = reader.read();
            layer.cross_attn.key_weight.bias = reader.read();
            layer.cross_attn.query_weight.kernel = reader.read();
            layer.cross_attn.query_weight.bias = reader.read();
            layer.cross_attn.value_weight.kernel = reader.read();
            layer.cross_attn.value_weight.bias = reader.read();
            layer.cross_attn.attention_output_weight.kernel = reader.read();
            layer.cross_attn.attention_output_weight.bias = reader.read();

            layer.pre_ffn_layernorm.gamma = reader.read();
            layer.pre_ffn_layernorm.beta = reader.read();

            layer.ffn.intermediate_weight.kernel = reader.read();
            layer.ffn.intermediate_weight.bias = reader.read();
            layer.ffn.output_weight.kernel = reader.read();
            layer.ffn.output_weight.bias = reader.read();

            weight_.layers.push_back(layer);
        }

        context->cublas_wrapper->setFP32GemmConfig();
    }

    th::Tensor FTWhisperDecoder::forward(th::Tensor encoder_output, th::Tensor inputs, th::Tensor input_lengths, double temperature)
    {
        uint32_t beams = 5; 
        uint32_t batch = encoder_output.size(0);
        uint32_t *end_ids = (uint32_t*) context->iallocator->malloc(sizeof(uint32_t) * batch);
        //cuMemsetD32((size_t) end_ids, end_id, batch);
        ft::invokeGenericMemset<uint32_t>(end_ids, end_id, batch, context->stream_);
        ft::TensorMap decoder_inputs(
            {
                {
                    "encoder_outputs",
                    convert_tensor<float>(encoder_output)
                },
                {
                    "decoder_inputs",
                    convert_tensor<uint32_t>(inputs)
                },
                {
                    "input_lengths",
                    convert_tensor<uint32_t>(input_lengths),
                },
                {
                    "beam_width",
                    ft::Tensor(
                        ft::MEMORY_CPU,
                        ft::getTensorType<uint32_t>(),
                        {1},
                        &beams
                    )
                },
                {
                    "end_id",
                    ft::Tensor(
                        ft::MEMORY_GPU,
                        ft::getTensorType<uint32_t>(),
                        {batch},
                        end_ids
                    )
                }
            }
        );

        th::Tensor output_ids = 
        th::empty(
            {
                batch,
                config.max_target_positions
            },
            torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false)
        );
        ft::TensorMap decoder_output_tensors(
            {
                {"output_ids",
                convert_tensor<uint32_t>(output_ids)}
            }
        );
        std::cout<<"begin forward";
        decoder->forward(decoder_output_tensors, decoder_inputs, weight_);
        std::cout<<"end forward";
        context->iallocator->free((void**) &end_ids);
        return output_ids;
    }
    
    FTWhisperDecoder::~FTWhisperDecoder()
    {
        delete decoder;
        delete context;
    }
}


static th::jit::class_<torch_ext::FTWhisperDecoder> ftWhisperDecoderTh 
=   th::jit::class_<torch_ext::FTWhisperDecoder>("FasterTransformer", "FTWhisperDecoder")
    .def(torch::jit::init<std::vector<th::Tensor>>())
    .def("forward", &torch_ext::FTWhisperDecoder::forward)
;