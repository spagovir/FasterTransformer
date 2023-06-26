#include "src/fastertransformer/models/whisper/CachedSequenceAttention.h"

namespace fastertransformer
{
template<typename T>
void CachedSequenceAttention<T>::forward(TensorMap output_tensors, TensorMap input_tensors, AttentionWeight<T> weight)
/*
input_tensors:
    input_queries : [batch, q_seq, d_model]
    input_keys : [batch, k_seq, d_model]
    input_lengths: size_t[batch] (optional, mask is autoregressive if none)
    step : size_t[1]
    transpose_key_cache: (optional), transpose key_cache to batch, num_heads, size_per_head/x, seq, x if not null

output_tensors:
    hidden_units : [batch, q_seq, d_model]
    key_cache: [batch, num_heads, k_seq, size_per_head]
    value_cache: [batch, num_heads, k_seq, size_per_head]
*/
{
    Tensor input_queries = input_tensors.at("input_queries");
    Tensor input_keys = input_tensors.at("input_keys");
    size_t batch = input_queries.shape[0];
    size_t q_seq = input_queries.shape[1];
    size_t d_model = input_queries.shape[2];
    size_t 
}

template class CachedSequenceAttention<float>;
}