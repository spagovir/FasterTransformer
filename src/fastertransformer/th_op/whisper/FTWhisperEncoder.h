#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/models/whisper/WhisperEncoder.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/th_op/whisper/FTWhisperConfig.h"
#include <c10/util/intrusive_ptr.h>

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {
    class FTWhisperEncoder :public th::CustomClassHolder
    {   
        ft::WhisperEncoder<float>* encoder;
        ft::WhisperCudaContext* context;
        ft::WhisperEncoderWeight<float> weight;
        public:
        FTWhisperEncoder(c10::intrusive_ptr<FTWhisperConfig> th_config, std::vector<th::Tensor> weights); // I'm lazy and only going to init for whisper tiny rn
        th::Tensor forward(th::Tensor input_ids, th::Tensor input_lengths);
        ~FTWhisperEncoder();

    };
}
