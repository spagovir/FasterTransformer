#include "src/fastertransformer/models/whisper/WhisperContextDecoder.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <vector>
#include "src/fastertransformer/th_op/whisper/FTWhisperConfig.h"
namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext{
    class FTWhisperDecoder:public th::CustomClassHolder
    {
        ft::WhisperContextDecoder<float> *decoder;
        ft::WhisperDecoderWeight<float> weight_;
        ft::WhisperCudaContext *context;
        ft::WhisperConfig config;
        uint32_t end_id;
        public:
        FTWhisperDecoder(c10::intrusive_ptr<FTWhisperConfig> config, std::vector<th::Tensor> weights);
        ~FTWhisperDecoder();
        th::Tensor forward(th::Tensor encoder_outputs, th::Tensor inputs, th::Tensor input_lengths, double temperature = 0);
    };
}
