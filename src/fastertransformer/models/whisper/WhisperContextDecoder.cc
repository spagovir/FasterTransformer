#include "src/fastertransformer/models/whisper/WhisperContextDecoder.h"
#include "src/fastertransformer/models/whisper/WhisperConfig.h"
#include "src/fastertransformer/models/whisper/WhisperCudaContext.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/models/whisper/WhisperKernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/cuda_utils.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include <cfloat>
#include <iostream>

namespace fastertransformer
{
    template<typename T>
    void WhisperContextDecoder<T>::forward(
        TensorMap &output_tensors,
        TensorMap &input_tensors,
        WhisperDecoderWeight<T> decoder_weight
    )
    /*
    input_tensors:
        "encoder_outputs" : [batch, seq, d_model]
        "decoder_inputs" : uint32_t[batch,target_seq]
        "decoder_input_lengths" : uint32_t[batch]
        NOT SUPPORTED YET "top_k": Optional uint32_t[1]
        "temperature": Optional [1]
        "beam_width": Optional uint32_t[1] CPU
        "input_lengths:" uint32_t[batch]
        "end_id": uint32_t[batch] GPU 

        // note: top_k and temperature only used if beam_width == null or 1;
        // if top_k is set temperature must also be set (and vice versa)
    output_tensors:
        "output_ids" : uint32_t[batch, max_target_positions]
        NOT_SUPPORTED "output_logprobs" : [batch, beam, max_target_positions, vocab_size]
    */
    {
        std::cout << "entered decoder \n";
        uint32_t batch = input_tensors.at("encoder_outputs").shape[0];
        uint32_t seq = input_tensors.at("encoder_outputs").shape[1];
        uint32_t beam = input_tensors.isExist("beam_width")? input_tensors.at("beam_width").getPtr<uint32_t>()[0] : 1;
        uint32_t out_seq = output_tensors.at("output_ids").shape[1];
        uint32_t output_beams_lda = batch * beam;
        uint32_t max_input_length = input_tensors.at("decoder_inputs").shape[1];
        std::cout << "inputs (" << max_input_length << "):\n";
        print_to_screen(input_tensors.at("decoder_inputs").getPtr<int>(), 10);
        if(!is_buffers_allocated_) allocateBuffer(batch,beam, seq, out_seq);
        sync_check_cuda_error();
        // repeat encoder output for each beam
        Tensor encoderOutputTensor = input_tensors.at("encoder_outputs");
        invokeRepeat<T>(decoder_input_buf, encoderOutputTensor, 1, beam, context_->stream_);
        // output_id_beams : seq x batch x beam
        // initialize output_id_beams from inputs:
        invokeCopyTransposeRepeat<uint32_t>(output_id_beams, input_tensors.at("decoder_inputs").getPtr<uint32_t>(), batch, max_input_length, beam, context_->stream_);
        std::cout << "after copy transpose repeat: \n:";
        printMatrix((int*) output_id_beams, 10, batch * beam, batch * beam, true);
        // while(std::cin.get() != '\n');
        // initialize buffers used in beam search
        invokeDecodingInitialize<float>(finished, sequence_lengths, nullptr, cumulative_log_probs, nullptr, batch, beam, 1, context_->stream_);
        // setup dynamic decode
        TensorMap dynamic_decode_setup_args = 
            TensorMap(
                {
                }
            );
        if(input_tensors.isExist("temperature") && input_tensors.isExist("top_k"))
        {
            dynamic_decode_setup_args.insert(
                {
                    "runtime_top_k",
                    input_tensors.at("top_k")
                }
            );
            dynamic_decode_setup_args.insert(
                {
                    "temperature",
                    input_tensors.at("temperature")
                }
            );
        }
        sampler.setup(
            batch,
            beam,
            &dynamic_decode_setup_args
        );
        // create ping-pong cache indirection buffer
        Tensor cache_indirs[2] = 
        {
            Tensor(
                MEMORY_GPU,
                getTensorType<uint32_t>(),
                {batch, beam, out_seq},
                cache_indir1
            ),
            Tensor(
                MEMORY_GPU,
                getTensorType<uint32_t>(),
                {batch,beam,out_seq},
                cache_indir2
            )
        };

        // initialize some tensors that are used in all iterations.

        Tensor decoder_output_logits = 
            Tensor(
                MEMORY_GPU,
                getTensorType<T>(),
                {batch, config_.max_beams, config_.vocab_size},
                logits_buffer);
        
        Tensor finished_tensor = 
            Tensor(
                MEMORY_GPU,
                getTensorType<bool>(),
                {batch * beam},
                finished
            );
        Tensor cum_log_probs_tensor =
            Tensor(
                MEMORY_GPU,
                getTensorType<float>(),
                {batch * beam},
                cumulative_log_probs
            );
        for(int idx = 1; idx < out_seq; idx ++)
        {
            if(idx == max_input_length) invokePaddingInitialize(padding_count, input_tensors.at("decoder_input_lengths").getPtr<int>(), max_input_length, batch, beam, context_->stream_);
            int src_idx = idx % 2; 
            int tgt_idx = 1 - src_idx;
            Tensor step = Tensor(MEMORY_CPU, getTensorType<uint32_t>(), {1}, &idx);
            uint32_t size_per_head = config_.d_model / config_.decoder_attention_heads;
            uint32_t x = 16 / sizeof(T);
            uint32_t size_per_head_x = size_per_head/x;
            TensorMap decoder_outputs = 
                TensorMap(
                    {
                        {
                            "output_logits",
                            decoder_output_logits
                        },
                        {
                            "self_key_cache",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {config_.decoder_layers, batch * config_.max_beams, config_.decoder_attention_heads, size_per_head_x, out_seq, x},
                                self_key_cache
                            )
                        },
                        {   
                            "self_value_cache",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {config_.decoder_layers, batch * config_.max_beams, config_.decoder_attention_heads, out_seq, size_per_head},
                                self_value_cache
                            )
                        },
                        {
                            "cross_key_cache",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {config_.decoder_layers, batch * config_.max_beams, config_.decoder_attention_heads, size_per_head_x, out_seq, x},
                                cross_key_cache
                            )
                        },
                        {
                            "cross_value_cache",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {config_.decoder_layers, batch * config_.max_beams, config_.decoder_attention_heads, out_seq, size_per_head},
                                cross_value_cache
                            )
                        }
                    }
                );
            TensorMap decoder_inputs =
                TensorMap(
                    {
                        {
                            "encoder_outputs",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<T>(),
                                {batch * beam, seq, config_.d_model},
                                decoder_input_buf
                            )
                        },
                        {
                            "input_ids",
                            Tensor(
                                MEMORY_GPU,
                                getTensorType<uint32_t>(),
                                {batch, beam},
                                output_id_beams + (idx-1) * output_beams_lda
                            )
                        },
                        {
                            "step",
                            step
                        },
                        {   
                            "cache_indirection",
                            cache_indirs[src_idx]
                        },
                        {   "sequence_lengths",
                        Tensor(MEMORY_GPU,
                        getTensorType<uint32_t>(),
                        {beam*batch},
                        sequence_lengths)},
                        {"padding_offsets",
                        Tensor(MEMORY_GPU,
                        getTensorType<uint32_t>(),
                        {beam*batch},
                        padding_count)}
                    }
                );
            decoder.forward(decoder_outputs, decoder_inputs, decoder_weight);
            std::cout << "pre-sampler (" << idx << "): ";
            print_to_screen(finished, batch * beam);
            print_to_screen(logits_buffer, 10);
            // std::cout << "logits: \n";
            // printMatrix(logits_buffer, beam * batch, 10, config_.d_model, true);
            // while(std::cin.get() != '\n');
            if(idx<max_input_length)
            {
                std::cout << "no beam search; \n";
                invokeStepSequenceLength(sequence_lengths, beam * batch, context_->stream_);
            }
            else{
                if(beam>1)
                {
                    std::cout << "yes beam search: \n";
                    uint32_t ite = 0;
                    TensorMap dynamic_decode_input_tensors =
                        TensorMap(
                            {
                                {
                                    "logits",
                                    decoder_output_logits
                                },
                                {"step", step},
                                {
                                    "max_input_length",
                                    Tensor(
                                        MEMORY_CPU,
                                        getTensorType<uint32_t>(),
                                        {1},
                                        &max_input_length
                                    )
                                },
                                {
                                    "end_id",
                                    input_tensors.at("end_id")
                                },
                                {
                                    "ite",
                                    Tensor(
                                        MEMORY_CPU,
                                        getTensorType<uint32_t>(),
                                        {1},
                                        &ite
                                    )
                                },
                                {"src_cache_indirection", cache_indirs[src_idx]},
                                {"local_batch_size",
                                Tensor(
                                    MEMORY_CPU,
                                    getTensorType<uint32_t>(),
                                    {1},
                                    &batch
                                )}
                            }
                        );
                    TensorMap dynamic_decode_output_tensors =
                        TensorMap(
                            {
                                {
                                    "output_ids",
                                    Tensor(
                                        MEMORY_GPU,
                                        getTensorType<uint32_t>(),
                                        {out_seq, batch, beam},
                                        output_id_beams
                                    )
                                },
                                {
                                    "finished",
                                    finished_tensor
                                },
                                {
                                    "cum_log_probs",
                                    cum_log_probs_tensor
                                },
                                {
                                    "tgt_cache_indirection",
                                    cache_indirs[tgt_idx]
                                },
                                {"parent_ids",
                                Tensor(MEMORY_GPU,
                                getTensorType<uint32_t>(),
                                {out_seq, batch * beam},
                                parent_ids_buf)},
                                {
                                    "sequence_length",
                                    Tensor(MEMORY_GPU,
                                    getTensorType<int>(),
                                    {batch * beam}, 
                                    sequence_lengths)
                                },
                            }
                        );

                    std::cout << "decoder logits: \n";
                    print_to_screen(logits_buffer, 384);
                    std::cout << "about to enter sampler\n";

                    sampler.forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
                    std::cout << "step: " << idx << "\n";
                    std::cout << cudaDeviceSynchronize() << "\n";

                    std::cout << "beam search outputs: \n";
                    std::cout << "output_ids: \n";
                    print_to_screen(output_id_beams + idx * output_beams_lda, 5);
                    std::cout << "finished: \n";
                    print_to_screen(finished, 5);
                    std::cout << "cum_log_probs: \n";
                    print_to_screen(cumulative_log_probs, 5);
                    
                }
                else {
                    // Top_k currently not supported. 
                    assert(false);
                }
            }

            // std::cout << "dynamic decode outputs: \n";
            // std::cout << "output_id_beams: ";
            // printMatrix((int*) output_id_beams, out_seq, batch * beam, batch* beam, true);
            // std::cout << "cum_log_probs: ";
            // printMatrix(cumulative_log_probs, 1, batch * beam, batch * beam, true);
            // std::cout << "sequence length: \n";
            // print_to_screen(sequence_lengths, batch * beam);
            // std::cout << "cache_indir: ";
            // printMatrix(cache_indirs[tgt_idx].getPtr<int>(), batch * beam, out_seq, out_seq, true);
            // while(std::cin.get() != '\n');

            std::cout << "post-sampler: " << idx << ": ";
            print_to_screen(output_id_beams + idx * output_beams_lda, batch * beam);

        }
        std::cout << "output beams: \n";
        printMatrix((int*) output_id_beams, 10, batch * beam, batch*beam, true);
        std::cout << "cum log probs: \n";
        printMatrix(cumulative_log_probs, 1, batch * beam, batch * beam, true);
        invokeCopyTransposeMaxBy<uint32_t,float>(output_tensors.at("output_ids").getPtr<uint32_t>(), output_id_beams, cumulative_log_probs, out_seq, batch, beam, context_->stream_);
        if(is_free_buffer_after_forward_)
        {
            freeBuffer();
        }
    }
    template<typename T>
    void WhisperContextDecoder<T>::allocateBuffer(uint32_t batch, uint32_t beam, uint32_t seq, uint32_t out_seq)
    {
        decoder.allocateBuffer(batch * beam, seq);
        IAllocator *allocator = context_->iallocator;
        padding_count = (int*) allocator->malloc(sizeof(int) * batch * beam);
        parent_ids_buf = (uint32_t*) allocator->malloc(sizeof(uint32_t) * batch * beam * out_seq);
        decoder_input_buf = (T*) allocator->malloc(sizeof(T) * batch * beam * seq * config_.d_model);
        cumulative_log_probs = (T*) allocator->malloc(sizeof(float) * batch * beam);
        self_key_cache = (T*) allocator->malloc(sizeof(T) * batch * beam * out_seq * config_.d_model * config_.decoder_layers);
        self_value_cache = (T*) allocator->malloc(sizeof(T) * batch * beam * out_seq * config_.d_model * config_.decoder_layers);
        cross_key_cache = (T*) allocator->malloc(sizeof(T) * batch * beam * seq * config_.d_model * config_.decoder_layers);
        cross_value_cache = (T*) allocator->malloc(sizeof(T) * batch * beam * seq * config_.d_model * config_.decoder_layers);
        cache_indir1 = (uint32_t*) allocator->malloc(sizeof(uint32_t) * batch * beam * out_seq * 2, true);
        cache_indir2 = cache_indir1 + batch * beam * out_seq;
        logits_buffer = (T*) allocator->malloc(sizeof(T) * batch * beam * config_.vocab_size);
        sequence_lengths = (int*) allocator->malloc(sizeof(int) * batch * beam); 
        finished = (bool*) allocator->malloc(sizeof(bool) * batch * beam);
        output_id_beams = (uint32_t*) allocator->malloc(sizeof(uint32_t) * batch * beam * out_seq);
        is_buffers_allocated_ = true;
    }
    template<typename T> 
    void WhisperContextDecoder<T>::freeBuffer()
    {
        decoder.freeBuffer();
        IAllocator *allocator = context_->iallocator;
        allocator->free((void**) &padding_count);
        allocator->free((void**) &decoder_input_buf);
        allocator->free((void**) &cumulative_log_probs);
        allocator->free((void**) & self_key_cache);
        allocator->free((void**) &self_value_cache);
        allocator->free((void**) &cross_key_cache);
        allocator->free((void**) &cross_value_cache);
        allocator->free((void**) &cache_indir1);
        allocator->free((void**) &logits_buffer);
        allocator->free((void**) &sequence_lengths);
        allocator->free((void**) &finished);
        allocator->free((void**) &output_id_beams);
        allocator->free((void**) &parent_ids_buf);
        is_buffers_allocated_ = false;
    }
    template<typename T>
    WhisperContextDecoder<T>::WhisperContextDecoder(WhisperCudaContext *context, WhisperConfig config, bool is_free_buffer_after_forward):
            context_(context),
            config_(config),
            is_free_buffer_after_forward_(is_free_buffer_after_forward),
            is_buffers_allocated_(false),
            decoder(config, context, false),
            sampler(config.vocab_size, config.vocab_size, config.eos_token_id, context->stream_, context->cublas_wrapper, context->iallocator, false, &context->prop_)
            {}
    
    template<typename T>
    WhisperContextDecoder<T>::~WhisperContextDecoder()
    {
        if(!is_buffers_allocated_) freeBuffer();
    }
    template class WhisperContextDecoder<float>;
}