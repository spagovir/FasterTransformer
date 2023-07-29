#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cstdint>
namespace fastertransformer {

    void invokeEmbedSinusoid(Tensor out_tensor, cudaStream_t stream, uint32_t max_time = 10000);
    void invokeCausalAttnMask(float* out, uint32_t batch, uint32_t seq, cudaStream_t stream);
    void invokeEncoderAttnMask(float* out, uint32_t batch, uint32_t seq, cudaStream_t stream);
    void invokeEmbed(float* out, int* in, float* weight, int n, int d_model, cudaStream_t stream);
    template <typename T>
    void invokeRepeat(T* out, Tensor in, uint32_t axis, uint32_t m, cudaStream_t stream);
    template<typename T>
    void invokeCopyTransposeRepeat(T* out, T* in, int a, int b, int r, cudaStream_t stream);
    template<typename T1,typename T2=T1> 
    void invokeCopyTransposeMaxBy(T1* out, T1* in, T2* by, int a, int b, int r, cudaStream_t stream);
    template<typename T> 
    void invokeGenericMemset(T *out, T val, int n, cudaStream_t stream);
    void invokeDecoderPosEmbed(float* out, float* weight, int n, int step, int d_model, cudaStream_t stream);
    template<typename T> 
    void invokeBatchPosEmbed(T* out, T* weight, int batch, int seq, int d_model, cudaStream_t stream);
    void invokeStepSequenceLength(int* out, int n, cudaStream_t stream);
    void invokePaddingInitialize(int* padding_count, int* input_lengths, int max_input_length, int batch, int beam, cudaStream_t stream);
}