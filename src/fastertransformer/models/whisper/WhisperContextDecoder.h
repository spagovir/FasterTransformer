#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/models/whisper/WhisperEncoder.h"
#include "src/fastertransformer/models/whisper/WhisperDecoder.h"
#include "src/fastertransformer/models/whisper/WhisperEncoderWeight.h"
namespace fastertransformer 
{
    template<typename T>
    class WhisperForConditionalGeneration
    {
        WhisperCudaContext *context_;
        WhisperConfig config_;
        WhisperEncoder<T> encoder;
        WhisperDecoder<T> decoder;
        BaseBeamSearchLayer<T> beamsearch;
        T* encoder_output_buf;
        T* decoder_input_buf;
        T* cumulative_log_probs;
        T* self_key_cache;
        T* self_value_cache;
        T* cross_key_cache;
        T* cross_value_cache;
        T* cache_indir;
        T* logits_buffer;
        T* output_id_beams;

        public:
        void forward(TensorMap &output_tensors, TensorMap &input_tensors, WhisperEncoderWeight<T> encoder_weight, WhisperDecoderWeight<T> decoder_weight);

    };
}