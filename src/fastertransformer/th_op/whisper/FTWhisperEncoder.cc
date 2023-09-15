#include "src/fastertransformer/th_op/whisper/FTWhisperEncoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/th_op/whisper/VectorReader.h"
#include <c10/core/ScalarType.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <ostream>

namespace torch_ext 
{
    FTWhisperEncoder::FTWhisperEncoder(c10::intrusive_ptr<FTWhisperConfig> th_config, std::vector<th::Tensor> weights)
    {   ft::WhisperConfig config = th_config->config
    ;   std::cout << config.d_model << std::endl;
    ;   context = 
        new ft::WhisperCudaContext
        (   at::cuda::getCurrentCUDABlasHandle()
        ,   at::cuda::getCurrentCUDAStream()
        ,   new ft::Allocator<ft::AllocatorType::TH>()
        )
    ;   encoder = new ft::WhisperEncoder<float>
        (   context 
        ,   true
        ,   config)
    ;   VectorReader<float> reader(&weights);
    ;   weight.conv1.kernel = reader.read()
    ;   weight.conv1.bias   = reader.read()
    ;   weight.conv2.kernel = reader.read()
    ;   weight.conv2.bias   = reader.read()
    ;   for(int i = 0; i < config.encoder_layers; i++)
        {   ft::WhisperEncoderLayerWeight<float> lweight
        ;   lweight.self_attn.key_weight.kernel = reader.read()
        ;   lweight.self_attn.value_weight.kernel = reader.read()
        ;   lweight.self_attn.value_weight.bias = reader.read()
        ;   lweight.self_attn.query_weight.kernel = reader.read()
        ;   lweight.self_attn.query_weight.bias = reader.read()
        ;   lweight.self_attn.attention_output_weight.kernel = reader.read()
        ;   lweight.self_attn.attention_output_weight.bias = reader.read()
        ;   lweight.layernorm1.gamma = reader.read()
        ;   lweight.layernorm1.beta = reader.read()
        ;   lweight.ffn.intermediate_weight.kernel = reader.read()
        ;   lweight.ffn.intermediate_weight.bias = reader.read()
        ;   lweight.ffn.output_weight.kernel = reader.read()
        ;   lweight.ffn.output_weight.bias = reader.read()
        ;   lweight.layernorm2.gamma = reader.read()
        ;   lweight.layernorm2.beta = reader.read()
        ;   weight.layers.push_back(lweight)
        ;   }
    ;   weight.layernorm.gamma = reader.read()
    ;   weight.layernorm.beta = reader.read()
    ;   context->cublas_wrapper->setFP32GemmConfig()
    ;   };

    th::Tensor FTWhisperEncoder::forward(th::Tensor input_ids) //, th::Tensor input_lengths)
    {   std::vector<uint32_t> size = encoder->out_size(input_ids.size(0), input_ids.size(1));
    ;   std::cout << size[0] << ", " << size[1] << ", " << size[2] << std::endl;
    ;   th::Tensor output_tensor 
        =   th::empty
            ({  static_cast<long>(size.at(0))
            , static_cast<long>(size.at(1))
            , static_cast<long>(size.at(2))}
            , torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false))
    ;   ft::Tensor output_ft_tensor = convert_tensor<float>(output_tensor,ft::MEMORY_GPU)
    ;   ft::Tensor input_ids_ft  = convert_tensor<float>(input_ids,ft::MEMORY_GPU)
    // ;   ft::Tensor input_lengths_ft = convert_tensor<float>(input_lengths, ft::MEMORY_GPU)
    ;   ft::TensorMap input_map
    ;   ft::TensorMap output_map
    ;   input_map.insert({"encoder_input",input_ids_ft})
    // ;   input_map.insert({"input_lengths", input_lengths_ft})
    ;   output_map.insert({"encoder_output", output_ft_tensor})
    ;   std::cout << "forward reached" << std::endl;
    ;   encoder->forward(input_map, output_map, weight)
    ;   cudaStreamSynchronize(context->stream_)
    ;   return output_tensor
    ;   } 
;   FTWhisperEncoder::~FTWhisperEncoder()
    {   delete encoder
    ;   delete context
    ;   encoder = nullptr
    ;   context = nullptr
    ;   }
;   }

static th::jit::class_<torch_ext::FTWhisperEncoder> ftWhisperEncoderTh 
=   th::jit::class_<torch_ext::FTWhisperEncoder>("FasterTransformer", "FTWhisperEncoder")
    .def(torch::jit::init<c10::intrusive_ptr<torch_ext::FTWhisperConfig>, std::vector<th::Tensor>>())
    .def("forward", &torch_ext::FTWhisperEncoder::forward)
;