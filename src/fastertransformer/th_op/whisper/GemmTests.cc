#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/utils/cuda_utils.h"
namespace ft = fastertransformer;
namespace th = torch;

using namespace torch_ext;

int main() 
{
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ft::Allocator<fastertransformer::AllocatorType::TH> allocator;
    ft::WhisperCudaContext context(handle, stream, &allocator);
    th::Tensor A = th::eye(2, th::TensorOptions().device(c10::kCUDA).dtype(th::kFloat32));
    th::Tensor B = th::eye(2, th::TensorOptions().device(c10::kCUDA).dtype(th::kFloat32));
    th::Tensor C = th::eye(2, th::TensorOptions().device(c10::kCUDA).dtype(th::kFloat32));
    context.cublas_wrapper->Gemm(
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        2,
        2,
        2,
        get_ptr<float>(A),
        2,
        get_ptr<float>(B),
        2,
        get_ptr<float>(C),
        2);
    ft::sync_check_cuda_error();
    PRINT_TENSOR(C);
    return 0;
    
}