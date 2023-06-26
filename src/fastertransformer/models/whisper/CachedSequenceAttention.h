#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
namespace fastertransformer
{
template <typename T>
class CachedSequenceAttention : public BaseLayer 
{
    T* q_buf_;
    T* k_buf_;
    T* v_buf_;
    T* q_buf_2_;
    T* k_buf_2_;
    T* qk_buf_;
    T* qk_buf_2_;
    T* qkv_buf_;
    public:
    void forward(TensorMap outputTensors, TensorMap inputTensors, AttentionWeight<T> weight);
}
}