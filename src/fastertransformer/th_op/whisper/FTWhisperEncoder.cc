#include "src/fastertransformer/th_op/whisper/FTWhisperEncoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include <c10/core/ScalarType.h>
#include <cuda_runtime_api.h>

namespace torch_ext 
{
    FTWhisperEncoder::FTWhisperEncoder(std::vector<th::Tensor> weights) 
    {   context = 
        new ft::WhisperCudaContext
        (   (std::cout << "initializing blas",at::cuda::getCurrentCUDABlasHandle())
        ,   (std::cout << "getting stream", at::cuda::getCurrentCUDAStream())
        ,   (std::cout << "creating allocator",new ft::Allocator<ft::AllocatorType::TH>())
        )
    ;   encoder = new ft::WhisperEncoder<float>
        (   context 
        ,   true
        ,   ft::WhisperConfig())
    ;   weight.conv1.kernel = get_ptr<float>(weights.at(0))
    ;   weight.conv1.bias   = get_ptr<float>(weights.at(1))
    ;   weight.conv2.kernel = get_ptr<float>(weights.at(2))
    ;   weight.conv2.bias   = get_ptr<float>(weights.at(3))
    ;   };

    th::Tensor FTWhisperEncoder::forward(th::Tensor input_ids, th::Tensor input_lengths)
    {   std::vector<size_t> size = encoder->out_size(input_ids.size(0), input_ids.size(1));
    ;   th::Tensor output_tensor 
        =   th::empty
            ({  static_cast<long>(size.at(0))
            , static_cast<long>(size.at(1))
            , static_cast<long>(size.at(2))}
            , torch::dtype(torch::kFloat32).device(torch::kCUDA).requires_grad(false))
    ;   ft::Tensor output_ft_tensor = convert_tensor<float>(output_tensor,ft::MEMORY_GPU)
    ;   ft::Tensor input_ids_ft  = convert_tensor<float>(input_ids,ft::MEMORY_GPU)
    ;   ft::Tensor input_lengths_ft = convert_tensor<float>(input_lengths, ft::MEMORY_GPU)
    ;   ft::TensorMap input_map
    ;   ft::TensorMap output_map
    ;   input_map.insert({"encoder_input",input_ids_ft})
    ;   input_map.insert({"input_lengths", input_lengths_ft})
    ;   output_map.insert({"encoder_output", output_ft_tensor})
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
    .def(torch::jit::init<std::vector<th::Tensor>>())
    .def("forward", &torch_ext::FTWhisperEncoder::forward)
;