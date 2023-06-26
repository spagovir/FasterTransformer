#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"

namespace fastertransformer
{
template<typename T>
struct WhisperDecoderLayerWeight 
{
    LayerNormWeight<T> pre_self_attn_layernorm; 
    AttentionWeight<T> self_attn;
    LayerNormWeight<T> pre_cross_attn_layernorm;
    AttentionWeight<T> cross_attn;
    LayerNormWeight<T> pre_ffn_layernorm;
    FfnWeight<T> ffn; 
};
}