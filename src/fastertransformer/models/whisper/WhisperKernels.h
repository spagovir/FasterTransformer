#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
namespace fastertransformer {

    void invokeEmbedSinusoid(Tensor out_tensor, cudaStream_t stream, size_t max_time = 10000);

}