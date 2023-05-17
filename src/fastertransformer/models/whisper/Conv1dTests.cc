#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/conv1d.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include <cstdlib>
#include <new>
#include <random>
namespace ft = fastertransformer;

int main(int argc, char **argv)
{   printf("main entered\n")
;   cublasHandle_t handle
;   cublasCreate(&handle)
;   cudaStream_t stream
;   cudaStreamCreate(&stream)
;   ft::Allocator<ft::AllocatorType::CUDA> *allocator = new ft::Allocator<ft::AllocatorType::CUDA>(ft::getDevice())
;   ft::WhisperCudaContext *context = new ft::WhisperCudaContext(handle, stream, allocator)
;   const int seq = 3000
;   const int stride = 2
;   const int pad = 1
;   const int kernel_size = 3
;   const int in_chan = 80
;   const int out_chan = 384
;   const int k1_dims = out_chan*in_chan*3
;   const int k2_dims = out_chan*out_chan*3
;   float *k1 = (float*) context->iallocator->malloc(k1_dims *4)
;   float *k2 = (float*) context->iallocator->malloc(k2_dims*4)
;   auto engine = std::mersenne_twister_engine<size_t,32,624,397,31,0x9908b0df,11, 0xffffffff,7,0x9d2c5680,15,0xefc60000,18,1812433253>()
;   std::uniform_real_distribution<float> dist = std::uniform_real_distribution<float>(0,1)
;   cudnnHandle_t& handle1 = context->cudnn_handle
;   cudnnHandle_t& handle2 = context->cudnn_handle
;   printf("hellow\n")
;   for(int i = 0; i < k1_dims; i++)
    {   k1[i] = dist(engine);}
;   for(int i = 0; i < k2_dims; i++)
    {   k2[i] = dist(engine);}
;   printf("hello")

;   float *t0 = (float*) context->iallocator->malloc(seq*in_chan*sizeof(float))
;   float *t1 = (float*) context->iallocator->malloc(seq*out_chan*sizeof(float))
;   float *t2 = new float[((seq-1)/stride+1) * out_chan]
;   for(int i = 0; i< seq*in_chan; i++)
    {   t0[i] = dist(engine);  }
;   printf("beginning conv1d 1: %f\n", k1[0])
;   ft::conv1d(t1, t0, k1, 1, seq, pad, in_chan, out_chan, kernel_size, 1, handle1)
;   printf("beginning conv1d 2: %f\n", t1[20]);
;   ft::conv1d(t2, t1, k2, 1, seq, pad, out_chan, out_chan, kernel_size, stride, handle2) 
;   printf("conv2 run. output: %f\n", t2[1])
;   delete context
;   delete[] k2
;   delete[] k1
;   delete[] t0
;   delete[] t1
;   delete[] t2
;   return 1
;   }