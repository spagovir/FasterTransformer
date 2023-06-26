#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
namespace fastertransformer {

    void invokeEmbedSinusoid(Tensor out_tensor, cudaStream_t stream, size_t max_time = 10000);
    void invokeCausalAttnMask(float* out, size_t batch, size_t seq, cudaStream_t stream);
    void invokeEncoderAttnMask(float* out, size_t batch, size_t seq, cudaStream_t stream);
    void invokeEmbed(float* out, int* in, float* weight, int n, int d_model, cudaStream_t stream);
    template <typename T>
    void invokeRepeat(T* out, Tensor in, size_t axis, size_t m, cudaStream_t stream);
}