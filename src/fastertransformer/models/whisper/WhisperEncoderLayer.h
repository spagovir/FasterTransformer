#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/attention_layers/UnfusedAttentionLayer.h"
#include "src/fastertransformer/layers/FfnLayer.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/models/whisper/WhisperEncoderLayerWeight.h"
namespace fastertransformer
{
template<typename T> 
class WhisperEncoderLayer : public BaseLayer
{   
    UnfusedAttentionLayer<T> self_attn;
    GeluFfnLayer<T> ffn;
    T* attn_mask;
    bool buffers_allocated;
    size_t max_batch;
    size_t max_seq;
    size_t d_model;
    T* k_bias;
    void allocateBuffer(size_t batch, size_t seq);
    public:
    WhisperEncoderLayer( WhisperConfig config
                       , WhisperCudaContext *context
                       , bool is_free_buffer_after_forward);
    ~WhisperEncoderLayer();
    void forward( Tensor residual
                , WhisperEncoderLayerWeight<T> weight
                , LayerNormWeight<T> next_ln_weight
                , T* lno_buffer
                , bool is_first);
    void allocateBuffer() override;
    void freeBuffer() override;
    

};
}
