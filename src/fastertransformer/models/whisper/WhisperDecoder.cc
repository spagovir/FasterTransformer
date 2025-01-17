#include "src/fastertransformer/models/whisper/WhisperDecoder.h"
#include "src/fastertransformer/kernels/layernorm_int8_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/models/whisper/WhisperKernels.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
namespace fastertransformer
{
    template<typename T> 
    WhisperDecoder<T>::WhisperDecoder(
        WhisperConfig config,
        WhisperCudaContext *context,
        bool is_free_after_forward
    ):
        BaseLayer(
            context->stream_,
            context->cublas_wrapper,
            context->iallocator,
            is_free_after_forward),
        config_(config),
        self_attn_(
            config.batch_size * config.max_beams, 
            config.decoder_attention_heads,
            config.d_model/config.decoder_attention_heads,
            context->stream_,
            context->cublas_wrapper,
            context->iallocator,
            is_free_after_forward,
            false,
            0),
        cross_attn_(
            config.batch_size * config.max_beams,
            config.decoder_attention_heads,
            config.d_model/config.decoder_attention_heads,
            context->stream_,
            context->cublas_wrapper,
            context->iallocator,
            is_free_after_forward
        ),
        ffn(
            config.batch_size * config.max_beams,
            config.max_target_positions,
            1,
            config.d_model,
            1,
            config.decoder_ffn_dim,
            context->stream_,
            context->cublas_wrapper,
            context->iallocator,
            is_free_after_forward
        ),
        is_buffers_allocated_(false) {};


    template<typename T>
    void WhisperDecoder<T>::allocateBuffer(uint32_t n, uint32_t encoder_seq)
    {
        residual_buf = (float*) allocator_->malloc(n * config_.d_model * sizeof(T));
        lno_buf = (float*) allocator_->malloc(n * config_.d_model * sizeof(T));
        // sequence_lengths = (uint32_t*) allocator_->malloc(n * sizeof(uint32_t));
        encoder_sequence_lengths = (uint32_t*) allocator_->malloc(n*sizeof(uint32_t));
        //cuMemsetD32((size_t) sequence_lengths, config_.max_target_positions, n);
        //cuMemsetD32((size_t) encoder_sequence_lengths, encoder_seq, n);
        //allocator_->memSet(sequence_lengths, config_.max_target_positions, sizeof(uint32_t));
        //allocator_->memSet(encoder_sequence_lengths, encoder_seq, sizeof(uint32_t));
        //invokeGenericMemset(sequence_lengths, config_.max_target_positions,n,stream_);
        invokeGenericMemset(encoder_sequence_lengths, encoder_seq, n, stream_);
        is_buffers_allocated_ = true;
    }
    
    template<typename T> 
    void WhisperDecoder<T>::allocateBuffer(){
        assert(false);
    }

    template<typename T> 
    void WhisperDecoder<T>::freeBuffer(){
        allocator_->free((void**) &residual_buf);
        allocator_->free((void**) &lno_buf);
        // allocator_->free((void**) &sequence_lengths);
        allocator_->free((void**) &encoder_sequence_lengths);
        is_buffers_allocated_ = false;
    }

    void usr_prompt(){ std::cout << "press enter to continue \n"; while(std::cin.get()!='\n');}

    template<typename T>
    void WhisperDecoder<T>::forward(TensorMap &output_tensors, TensorMap &input_tensors, WhisperDecoderWeight<T> weight)
    /*
    input_tensors:
    encoder_outputs : [batch * beam, seq, d_model]
    input_ids : int[out_seq, batch, beam]
    step: [1] on CPU
    cache_indirection: uint32_t[batch,beam,seq]
    sequence_lengths: uint32_t[batch * beam]
    padding_offsets: uint32_t[batch * beam]

    output_tensors:
    output_logits : [batch, beam, vocab_size]
    self_key_cache : [layers, batch * beam, num_heads, size_per_head/x, seq, x] // x = 16/sizeof(T) (eg, packed 128 bit floats)
    self_value_cache : [layers, batch * beam, num_heads, seq, size_per_head]
    cross_key_cache: "
    cross_value_cache: "
    */
    {


        Tensor encoder_outputs = input_tensors.at("encoder_outputs");
        Tensor input_ids = input_tensors.at("input_ids");
        int n = encoder_outputs.shape[0] ;
        if(!is_buffers_allocated_) allocateBuffer(n, encoder_outputs.shape[1]);
        uint32_t batch = input_ids.shape[1];
        uint32_t beam = input_ids.shape[2];
        uint32_t cache_lda = ((uint32_t) n) * config_.max_target_positions * config_.d_model;
        uint32_t cross_cache_lda = ((uint32_t) n) * encoder_outputs.shape[1] * config_.d_model;

        // std::cout << "validating decoder inputs: \n";
        // for(uint32_t i = 0; i < batch * beam; i ++)
        // {
            // std::cout << "encoder output ( " << i << "): \n"; 
            // printMatrix(encoder_outputs.getPtr<T>() + i * config_.d_model * 1500, 10, 10, config_.d_model, true); 
        // }
        // std::cout << "input ids: \n";
        // print_to_screen(input_ids.getPtr<uint32_t>(), batch * beam);
        // std::cout << "step: " << *input_tensors.at("step").getPtr<uint32_t>() << "\n; ";

        // std::cout << "press enter to continue. \n";
        // while(std::cin.get() != '\n');




        // std::cout << "decoder inputs: \n";
        // print_to_screen(input_tensors.at("input_ids").getPtr<int>() + (*input_tensors.at("step").getPtr<int>()-1) * batch * beam, batch * beam);
        invokeEmbeddingLookupPosEncodingPadCount(
            residual_buf,
            weight.token_embed,
            weight.pos_embed,
            input_tensors.at("input_ids").getPtr<int>(),
            input_tensors.at("padding_offsets").getPtr<int>(),
            batch * beam,
            config_.d_model,
            1.0f,
            *input_tensors.getPtr<int>("step") - 1,
            batch * beam,
            0,
            stream_
        );
        // std::cout << "post embed;";
        // print_to_screen(residual_buf, 10);
        // std::cout << "validating embed outputs: \n";
        // printMatrix(residual_buf, batch*beam, 10, config_.d_model, true);
        // std::cout << "press enter to continue. \n";
        // while(std::cin.get() != '\n');
        for(uint32_t l = 0; l < config_.decoder_layers; l++)
        {
            if(l == 0)
            {
                invokeGeneralLayerNorm(
                    lno_buf,
                    residual_buf,
                    weight.layers.at(l).pre_self_attn_layernorm.gamma,
                    weight.layers.at(l).pre_self_attn_layernorm.beta,
                    LAYERNORM_EPS,
                    n,
                    config_.d_model,
                    nullptr,
                    0,
                    stream_
                );
            }

            // std::cout << "validating self-attn ln outputs: \n";
            // printMatrix(lno_buf, n, 10, config_.d_model, true);
            // usr_prompt();
            
            // uint32_t step_minus_1 = *input_tensors.at("step").getPtr<uint32_t>() - 1; 
            // Tensor step_minus_1_tensor = Tensor(MEMORY_CPU, getTensorType<uint32_t>(), {1}, &step_minus_1);

            TensorMap self_attn_inputs =
                TensorMap(
                    {   {   "input_query"
                        ,   Tensor( MEMORY_GPU
                                  , getTensorType<T>()
                                  , {(uint32_t) n, config_.d_model}
                                  , lno_buf)}
                    ,   {   "sequence_lengths",
                    input_tensors.at("sequence_lengths")}
                    ,   {"step", input_tensors.at("step")}
                    ,   {"cache_indirection", input_tensors.at("cache_indirection")}
                    }
                );
            uint32_t x = 16/sizeof(T);
            uint32_t size_per_head = config_.d_model/config_.decoder_attention_heads;
            TensorMap self_attn_outputs = 
                TensorMap(
                        {   {   "hidden_features"
                            ,   Tensor  (   MEMORY_GPU
                                        ,   getTensorType<T>()
                                        ,   {(uint32_t) n, config_.d_model}
                                        ,   lno_buf)}
                        ,   {   "key_cache"
                                ,   Tensor  (   MEMORY_GPU
                                            ,   getTensorType<T>()
                                            ,   {(uint32_t) n, config_.decoder_attention_heads, size_per_head/x, config_.max_target_positions, x}
                                            ,   output_tensors.at("self_key_cache").getPtr<T>() + l * cache_lda
                                            )}
                        ,   {   "value_cache"
                                ,   Tensor  (   MEMORY_GPU
                                            ,   getTensorType<T>()
                                            ,   {(uint32_t) n, config_.decoder_attention_heads, config_.max_target_positions, size_per_head}
                                            ,   output_tensors.at("self_value_cache").getPtr<T>() + l * cache_lda)}
                    }
                );
            self_attn_.forward(&self_attn_outputs, &self_attn_inputs, &weight.layers[l].self_attn);
            sync_check_cuda_error();


            invokeGeneralAddBiasResidualPreLayerNorm(
                residual_buf,
                lno_buf,
                lno_buf,
                residual_buf,
                weight.layers[l].pre_cross_attn_layernorm.gamma,
                weight.layers[l].pre_cross_attn_layernorm.beta,
                weight.layers[l].self_attn.attention_output_weight.bias,
                LAYERNORM_EPS,
                n,
                config_.d_model,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                0,
                stream_
            );
            sync_check_cuda_error();

            // std::cout << "validating cross_attn ln output: \n";
            // printMatrix(lno_buf, n, 10, config_.d_model, true);
            // usr_prompt();

            TensorMap cross_attn_inputs = 
                TensorMap(
                    {   {   "input_query"
                        ,   Tensor  (   MEMORY_GPU
                                    ,   getTensorType<T>()
                                    ,   {(uint32_t) n, config_.d_model}
                                    ,   lno_buf)}
                    ,   {   "encoder_sequence_length"
                        ,   Tensor  (   MEMORY_GPU
                                    ,   getTensorType<uint32_t>()
                                    ,   {(uint32_t) n}
                                    ,   encoder_sequence_lengths)}
                    ,   {   "encoder_output"
                        ,   input_tensors.at("encoder_outputs")},
                        {   "step"
                        ,   input_tensors.at("step")}}
                );
            TensorMap cross_attn_outputs = 
                TensorMap(
                    {   {   "hidden_features"
                        ,   Tensor  (   MEMORY_GPU
                                    ,   getTensorType<T>()
                                    ,   {(uint32_t) n, config_.d_model}
                                    ,   lno_buf)}
                    ,   {   "key_cache"
                        ,   Tensor  (   MEMORY_GPU
                                    ,   getTensorType<T>()
                                    ,   {(uint32_t) n, config_.decoder_attention_heads, size_per_head/x, config_.max_target_positions, x}
                                    ,   output_tensors.at("cross_key_cache").getPtr<T>() + l * cross_cache_lda)}
                    ,   {   "value_cache"
                        ,   Tensor  (   MEMORY_GPU
                                    ,   getTensorType<T>()
                                    ,   {(uint32_t) n, config_.decoder_attention_heads, config_.max_target_positions, size_per_head}
                                    ,   output_tensors.at("cross_value_cache").getPtr<T>() + l * cross_cache_lda)}}
                );
            cross_attn_.forward(&cross_attn_outputs, &cross_attn_inputs, &weight.layers[l].cross_attn);
            sync_check_cuda_error();
            // std::cout << "validating cross_attn output: \n";
            // printMatrix(lno_buf, n, 10, config_.d_model, true);
            // std::cout << "key_cache: \n";
            // printMatrix(output_tensors.at("cross_key_cache").getPtr<T>() + l * cross_cache_lda, 10, x, x, true);
            // std::cout << "val_cache: \n";
            // printMatrix(output_tensors.at("cross_value_cache").getPtr<T>() + l *cross_cache_lda, 10, x, size_per_head, true);
            // usr_prompt();
            
            invokeGeneralAddBiasResidualPreLayerNorm(
                residual_buf,
                lno_buf,
                lno_buf,
                residual_buf,
                weight.layers[l].pre_ffn_layernorm.gamma,
                weight.layers[l].pre_ffn_layernorm.beta,
                weight.layers[l].cross_attn.attention_output_weight.bias,
                LAYERNORM_EPS,
                n,
                config_.d_model,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                0,
                stream_
            );
            sync_check_cuda_error();
            // std::cout << "validating final ln output: \n";
            // printMatrix(lno_buf, n, 10, config_.d_model, true);
            // usr_prompt();
            TensorMap ffn_inputs =
                TensorMap(
                    {   {   "ffn_input"
                        ,   Tensor  (   MEMORY_GPU
                                    ,   getTensorType<T>()
                                    ,   {(uint32_t) n, config_.d_model}
                                    ,   lno_buf)}}
                );
            TensorMap ffn_outputs = 
                TensorMap(
                    {   {   "ffn_output"
                        ,   Tensor  (   MEMORY_GPU
                                    ,   getTensorType<T>()
                                    ,   {(uint32_t) n, config_.d_model}
                                    ,   lno_buf)}}
                );
            ffn.forward(&ffn_outputs, &ffn_inputs, &weight.layers[l].ffn);
            sync_check_cuda_error();

            // std::cout << "validating ffn output: \n";
            // printMatrix(lno_buf, n, 10, config_.d_model, true);
            // usr_prompt();
            
            if(l + 1 == config_.decoder_layers){
                invokeGeneralAddBiasResidualPreLayerNorm(
                residual_buf,
                lno_buf,
                lno_buf,
                residual_buf,
                weight.ln.gamma,
                weight.ln.beta,
                weight.layers[l].ffn.output_weight.bias,
                LAYERNORM_EPS,
                n,
                config_.d_model,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                0,
                stream_
            );
            }
            else invokeGeneralAddBiasResidualPreLayerNorm(
                residual_buf,
                lno_buf,
                lno_buf,
                residual_buf,
                weight.layers[l+1].pre_self_attn_layernorm.gamma,
                weight.layers[l+1].pre_self_attn_layernorm.beta,
                weight.layers[l].ffn.output_weight.bias,
                LAYERNORM_EPS,
                n,
                config_.d_model,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                0,
                stream_
            );
            sync_check_cuda_error();


        }

        // std::cout << "hidden units pre unembed: \n";
        // printMatrix(lno_buf, batch * beam, 10, config_.d_model, true);
        // while(std::cin.get() != '\n');

        cublas_wrapper_->Gemm(
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            config_.vocab_size,
            n,
            config_.d_model,
            weight.token_embed,
            config_.d_model,
            lno_buf,
            config_.d_model,
            output_tensors.at("output_logits").getPtr<T>(),
            config_.vocab_size
        );
        sync_check_cuda_error();

        if(is_free_buffer_after_forward_) freeBuffer();
    }

    template class WhisperDecoder<float>;
}