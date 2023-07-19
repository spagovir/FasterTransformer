#pragma once;
#include "src/fastertransformer/models/whisper/WhisperDecoderLayerWeight.h"
#include <vector>
namespace fastertransformer {
template<typename T>
struct WhisperDecoderWeight {
    T* token_embed; //vector of length vocab_size
    T* pos_embed; //vector of length max_seq_len
    std::vector<WhisperDecoderLayerWeight<T>> layers;
    LayerNormWeight<T> ln;
};
}