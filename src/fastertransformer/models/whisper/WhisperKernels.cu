#include "src/fastertransformer/models/whisper/WhisperKernels.h"

namespace fastertransformer
{
// assume chan % 2 = 0
__global__ void embedSinusoid( float* out
                   , int batch
                   , int length
                   , int chan
                   , int size
                   , int max_time)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < size)
    {   
        int chanHalfIdx = id % (chan/2);
        int chanIdx = id % chan;
        int seqIdx = (id / chan) % length;
        //int batchIdx = id / (chan * length);
        float scaled_time = seqIdx * expf(-logf((float)max_time) 
                                         / ((float) (chan / 2 - 1))
                                         * chanHalfIdx);
        if(chanIdx < chan/2)
        {   out[id] += sinf(scaled_time);}
        else{   out[id] += cosf(scaled_time);}
    }
}

void invokeEmbedSinusoid(Tensor out_tensor, cudaStream_t stream, size_t max_time)
{
    int n = (int) out_tensor.size();
    dim3 block, grid;
    block.x = std::min<int>((int) n, 1024);
    grid.x = ceil(n/1024);
    embedSinusoid<<<grid, block, 0, stream>>>( out_tensor.getPtr<float>()
                   , (int) out_tensor.shape[0]
                   , (int) out_tensor.shape[1]
                   , (int) out_tensor.shape[2]
                   , n
                   , (int) max_time);
}
}