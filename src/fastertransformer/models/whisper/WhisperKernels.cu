#include "src/fastertransformer/models/whisper/WhisperKernels.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <algorithm>

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
    grid.x = ceil(((float)n)/1024);
    embedSinusoid<<<grid, block, 0, stream>>>( out_tensor.getPtr<float>()
                   , (int) out_tensor.shape[0]
                   , (int) out_tensor.shape[1]
                   , (int) out_tensor.shape[2]
                   , n
                   , (int) max_time);
}

__global__ void causalAttnMask(float* out, int batch, int length, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        int qIdx = (idx/length)%length;
        int kIdx = idx % length;
        if(qIdx >= kIdx) out[idx] = 1.0f; else out[idx] = 0.0f;
    }
}

__global__ void encoderAttnMask(float* out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n) out[idx] = 1.0f; else out[idx] = 0.0f;
}

void invokeEncoderAttnMask(float* out, size_t batch, size_t seq, cudaStream_t stream)
{ 
    int n = batch * seq * seq;
    dim3 block,grid;
    block.x = std::min<int>(1024,n);
    grid.x = ceil(((float) n)/1024);
    encoderAttnMask<<<grid,block,0,stream>>>(out,n);
}

void invokeCausalAttnMask(float* out, size_t batch, size_t seq, cudaStream_t stream)
{
    int n = batch * seq * seq;
    dim3 block,grid;
    block.x = std::min<int>(1024, n);
    grid.x = ceil(((float) n)/1024);
    causalAttnMask<<<grid,block,0,stream>>>(out, batch, seq, n);

}

/*
in[n]
out[n, d_model]
*/
__global__ void embed(float* out, int* in, float* weight, int n, int d_model)
{
    if(blockIdx.x < n && threadIdx.x < d_model)
        out[blockIdx.x * d_model + threadIdx.x] = weight[in[blockIdx.x]*d_model + threadIdx.x];
}

/*
Assume d_model < 1024
*/
void invokeEmbed(float* out, int* in, float* weight, int n, int d_model, cudaStream_t stream)
{
    dim3 block,grid;
    block.x = d_model;
    grid.x = n;
    embed<<<grid,block,0,stream>>>(out,in,weight,n,d_model);

}

template<typename T>
__global__ void repeat(T* out, T* in, int len, int m, int n, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx;
    if(idx < len)
    {
        out[idx] = in[idx/(n *k) + (idx % k)];
    }
}

template<typename T>
void invokeRepeat(T* out, Tensor in, size_t axis, size_t n, cudaStream_t stream)
{
    int m = 1;
    for(size_t i = 0; i < axis; i++)
        m *= in.shape[i];
    int k = 1;
    for(size_t i = axis; i < in.shape.size(); i++)
        k *= in.shape[i];
    int len = n * m * k; 
    dim3 grid, block;
    block.x = std::min<int>(len, 1024);
    grid.x = (len-1)/1024 + 1;
    repeat<<<grid,block,0,stream>>>(out,in.getPtr<T>(), len, m, n, k);
    
    


}

// mergesorts an array of integers of size < 1024,
// returning a sorted array of the indices of those integers.
/*
__global__ void pointerMergeSort(size_t* values, size_t* out, int n){
    __shared__ size_t buffer[1024];
    int idx = threadIdx.x;
    buffer[idx] = (size_t) idx; 
    size_t* from = &buffer;
    size_t* to = out;
    for(int i = 1; i<= n; i>>=1)
    {
        int left_idx = (i>>1) * idx;
        int right_idx = (i>>1) * idx + i;
        for(int j=0; j<i>>1; j++)
        {
            int target_idx = (i>>1) * idx + j;
            if(target_idx < n)
            {
                if  (   right_idx >= n 
                    ||  right_idx >= (i>>1) * (idx+1) 
                    ||  left_idx < (i>>1) * idx + i 
                        &&  values[from[left_idx]] < values[from[right_idx]])
                {
                    to[target_idx] = from[left_idx++];
                }
                else to[target_idx] = from[right_idx++];
            }
        }
        size_t* c = to;
        to = from;
        from = c;
        __syncthreads();
    }
    out[idx] = from[idx];
}
*/
/*
__global__ void oddEvenSort(size_t* values, size_t* out_indices, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n)
    {
        const int even_left_idx = idx*2;
        const int even_right_idx = idx*2 + 1;
        const int odd_left_idx = even_right_idx;
        const int odd_right_idx = idx*2 + 2;
        bool check_on_even = even_right_idx < n;
        bool check_on_odd = odd_right_idx < n;

        out_indices[idx] = idx;
        for(int i = 0; i < n; i++)
        {
            if  (i&1) // odd case
            {
                if  (   check_on_odd 
                    &&  values[out_indices[odd_left_idx]] 
                        >   values[out_indices[odd_right_idx]]
                    )
                {
                    size_t c = out_indices[odd_left_idx];
                    out_indices[odd_left_idx] = out_indices[odd_right_idx];
                    out_indices[odd_right_idx] = c;
                }
            }
            else if // even case
                    (   check_on_even
                    &&  values[out_indices[even_left_idx]]
                        >   values[out_indices[even_right_idx]]
                    ) 
                {
                    size_t c = out_indices[even_left_idx];
                    out_indices[even_left_idx] = out_indices[even_right_idx];
                    out_indices[even_right_idx] = c;
                } 
        }
    }
    
}
*/

/*
Copies a vector of dimensions [a,b] to one of [b,a,r];
*/
template<typename T> 
__global__ void copyTransposeRepeat(T* out, T* in, int a, int b, int r, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        int aIdx = (idx / r) % a;
        int bIdx = idx / r / a; 
        out[idx] = in[aIdx * b + bIdx];
    }
}

template<typename T>
void invokeCopyTransposeRepeat(T* out, T* in, int a, int b, int r, cudaStream_t stream)
{
    int n = a * b * r;
    dim3 block, grid;
    block.x = std::min<int>(n, 1024);
    grid.x = (n-1)/1024 + 1;
    copyTransposeRepeat<T><<<grid,block,0,stream>>>(out, in, a, b, r, n);
}

template<float> repeat(float* out, float* in, int len, int m, int n, int k);
template<float> invokeRepeat(float* out, Tensor in, size_t axis, size_t m, cudaStream_t stream);

template<size_t> copyTransposeRepeat(size_t* out, size_t* in, int a, int b, int r, int n);
template<size_t> invokeCopyTransposeRepeat(size_t* out, size_t* in, int a, int b, int r, cudaStream_t stream);

}
