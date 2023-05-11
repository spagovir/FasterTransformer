#include "src/fastertransformer/models/whisper/WhisperEncoderWeight.h"
#include "src/fastertransformer/models/whisper/Conv1dLayer.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
namespace fastertransformer {
template <typename T, AllocatorType AT>
class WhisperEncoder : public BaseLayer {
    WhisperConfig config;
    public: 
    WhisperEncoder
        ( WhisperCudaContext<AT>    &context
        , bool                      is_free_buffer_after_forward
        , WhisperConfig             config
        )
    
    ;   void    allocateBuffer()    override
    ;   void    freeBuffer()        override
    ;   void    forward
        (   TensorMap   &input_tensors
        ,   TensorMap   &output_tensors
        ,   WhisperEncoderWeight<T>    weight)
    ;   ~WhisperEncoder()
    ;   protected:
        bool    is_free_buffer_after_forward_
    ;   bool    buffers_allocated_
    ;   T* conv1_out_buffer
    ;   T* conv2_out_buffer
    ;   WhisperConfig           config_
    ;   WhisperCudaContext<AT>  &context_
    ;   void allocateBuffer(size_t batch, size_t in_seq)
    ;   Conv1dLayer<T>   conv1
    ;   Conv1dLayer<T>   conv2
    ;   }
;   }