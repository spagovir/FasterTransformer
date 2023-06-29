#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/models/whisper/WhisperDecoder.h"
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
        uint32_t* parent_ids_buf;
        uint32_t* cache_indir1; //[batch, beam, seq]
        uint32_t* cache_indir2;
        T* logits_buffer;
        int* sequence_lengths;
        bool* finished;
        uint32_t* output_id_beams;

        void allocateBuffer(uint32_t batch, uint32_t beam, uint32_t seq, uint32_t out_seq);
        void freeBuffer();

        public:
        void forward(TensorMap &output_tensors, TensorMap &input_tensors, WhisperDecoderWeight<T> decoder_weight);
        WhisperContextDecoder(WhisperCudaContext *context, WhisperConfig config, bool is_free_buffer_after_forward);    
        ~WhisperContextDecoder();

    };
}