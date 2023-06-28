#include "src/fastertransformer/models/whisper/WhisperContextDecoder.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include <ATen/core/TensorBody.h>
#include <ATen/core/ivalue.h>
#include <vector>
namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext{
    class FTWhisperDecoder:public th::CustomClassHolder
    {
        ft::WhisperContextDecoder<float> *decoder;
        ft::WhisperDecoderWeight<float> weight_;
        ft::WhisperCudaContext *context;
        public:
        FTWhisperDecoder(std::vector<th::Tensor> weights);
        ~FTWhisperDecoder();
        th::Tensor forward(th::Tensor encoder_outputs, th::Tensor inputs, th::Tensor input_lengths, float temperature = 0);
    };
}
