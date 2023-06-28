#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/models/whisper/WhisperEncoder.h"
#include "src/fastertransformer/models/whisper/WhisperDecoder.h"
#include "src/fastertransformer/models/whisper/WhisperEncoderWeight.h"
namespace fastertransformer 
{
    template<typename T>
    class WhisperContextDecoder
    {
        WhisperCudaContext *context_;
        WhisperConfig config_;
        WhisperDecoder<T> decoder;
        DynamicDecodeLayer<T> sampler;
        bool is_free_buffer_after_forward_;
        bool is_buffers_allocated_;
        // T* encoder_output_buf;
        T* decoder_input_buf;
        float* cumulative_log_probs;
        T* self_key_cache;
        T* self_value_cache;
        T* cross_key_cache;
        T* cross_value_cache;
        size_t* cache_indir1; //[batch, beam, seq]
        size_t* cache_indir2;
        T* logits_buffer;
        int* sequence_lengths;
        bool* finished;
        size_t* output_id_beams;

        void allocateBuffer(size_t batch, size_t beam, size_t seq, size_t out_seq);
        void freeBuffer();

        public:
        void forward(TensorMap &output_tensors, TensorMap &input_tensors, WhisperEncoderWeight<T> encoder_weight, WhisperDecoderWeight<T> decoder_weight);
        WhisperContextDecoder(WhisperCudaContext *context, WhisperConfig config, bool is_free_buffer_after_forward);    
        ~WhisperContextDecoder();

    };
}