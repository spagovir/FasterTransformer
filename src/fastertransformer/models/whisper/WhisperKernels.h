#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
namespace fastertransformer {

    void invokeEmbedSinusoid(Tensor out_tensor, cudaStream_t stream, size_t max_time = 10000);
    void invokeCausalAttnMask(float* out, size_t batch, size_t seq, cudaStream_t stream);
    void invokeEncoderAttnMask(float* out, size_t batch, size_t seq, cudaStream_t stream);
}