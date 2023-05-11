#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/layers/FfnWeight.h"
using namespace fastertransformer;

template <typename T>
struct WhisperEncoderLayerWeight {
    LayerNormWeight<T> layernorm1; 
    AttentionWeight<T> self_attn;
    LayerNormWeight<T> layernorm2; 
    FfnWeight<T> ffn;
};