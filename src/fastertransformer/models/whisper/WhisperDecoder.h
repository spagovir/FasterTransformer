#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/DecoderCrossAttentionLayer.h"
#include "src/fastertransformer/layers/attention_layers/DecoderSelfAttentionLayer.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/models/whisper/WhisperDecoderWeight.h"
namespace fastertransformer
{
    template<typename T> 
    class WhisperDecoder : public BaseLayer
    {
        WhisperConfig config_;
        DecoderSelfAttentionLayer<T> self_attn_;
        DecoderCrossAttentionLayer<T> cross_attn_;
        GeluFfnLayer<T> ffn;
        T* residual_buf;
        T* lno_buf;
        size_t* sequence_lengths;
        size_t* encoder_sequence_lengths;
        void allocateBuffer(size_t n, size_t encoder_seq);
        bool is_buffers_allocated_;
        public:
        WhisperDecoder(
            WhisperConfig config,
            WhisperCudaContext *context,
            bool is_free_after_forward
        );
        void forward(TensorMap &output_tensors, TensorMap &input_tensors, WhisperDecoderWeight<T> weight);
        void allocateBuffer() override;
        void freeBuffer() override;

    };
}