#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/layers/FfnWeight.h"
using namespace fastertransformer;

struct WhisperEncoderLayerWeight {
    LayerNormWeight<float> layernorm1; 
    AttentionWeight<float> self_attn;
    LayerNormWeight<float> layernorm2; 
    FfnWeight<float> ffn;
    // WhisperEncoderLayerWeight(LayerNormWeight<float> _layernorm1, AttentionWeight<float> _self_attn, LayerNormWeight<float> _layernorm2, FfnWeight<float> _ffn)
    // : layernorm1(_layernorm1), self_attn(_self_attn), layernorm2(_layernorm2), ffn(_ffn) {};
};