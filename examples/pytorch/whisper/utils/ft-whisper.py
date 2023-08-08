# %%
from transformers import WhisperForConditionalGeneration
import torch as th
import einops as e
from tokenizer import get_tokenizer
# %%
th.classes.load_library("/workspaces/FasterTransformer/build/lib/libth_transformer.so")


class FTWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def cuda(self, batch, beam):
        tokenizer = get_tokenizer(True)
        if self.config._name_or_path[-3:] == ",en":
            self.bot_sequence = [tokenizer.sot, tokenizer.no_timestamps]
        config = self.convert_config(batch, beam)
        weights = [weight.to("cuda") for weight in [ self.base_model.encoder.conv1.weight
                , self.base_model.encoder.conv1.bias
                , self.base_model.encoder.conv2.weight
                , self.base_model.encoder.conv2.bias]]
        for layer in self.base_model.encoder.layers:
            weights += \
            [   weight.to("cuda") for weight in 
            [   layer.self_attn.k_proj.weight.T.contiguous()  # * (384/6)**-0.25
            ,   layer.self_attn.v_proj.weight.T.contiguous()
            ,   layer.self_attn.v_proj.bias
            ,   layer.self_attn.q_proj.weight.T.contiguous()  # * (384/6)**-0.25
            ,   layer.self_attn.q_proj.bias  # * (384/6)**-0.25
            ,   layer.self_attn.out_proj.weight.T.contiguous()  
            ,   layer.self_attn.out_proj.bias
            ,   layer.self_attn_layer_norm.weight
            ,   layer.self_attn_layer_norm.bias
            ,   layer.fc1.weight.T.contiguous()
            ,   layer.fc1.bias
            ,   layer.fc2.weight.T.contiguous()
            ,   layer.fc2.bias
            ,   layer.final_layer_norm.weight
            ,   layer.final_layer_norm.bias
            ]]
        weights += [ weight.to("cuda") for weight in 
            [   self.base_model.encoder.layer_norm.weight
            ,   self.base_model.encoder.layer_norm.bias]]

        th.cuda.set_device(0)
        self.ft_encoder = th.classes.FasterTransformer.FTWhisperEncoder(config, weights)
        

        decoder_weights = [
            weight.to("cuda") for weight in
            [
                self.base_model.decoder.embed_tokens.weight,
                self.base_model.decoder.embed_positions.weight
            ]

        ]
        for layer in self.base_model.decoder.layers:
            decoder_weights += \
                [weight.to("cuda") for weight in 
                [
                    layer.self_attn_layer_norm.weight,
                    layer.self_attn_layer_norm.bias,
                    # layer.self_attn.k_proj.weight.T.contiguous(),
                    # th.zeros_like(layer.self_attn.q_proj.bias),
                    th.cat([
                        layer.self_attn.q_proj.weight,
                        layer.self_attn.k_proj.weight,
                        layer.self_attn.v_proj.weight
                        ],0).T.contiguous(),
                    th.cat( [ layer.self_attn.q_proj.bias
                            , th.zeros_like(layer.self_attn.q_proj.bias)
                            , layer.self_attn.v_proj.bias
                            ]
                        , 0),
                    # layer.self_attn.v_proj.weight.T.contiguous(),
                    # layer.self_attn.v_proj.bias,
                    layer.self_attn.out_proj.weight.T.contiguous(),
                    layer.self_attn.out_proj.bias,
                    layer.encoder_attn_layer_norm.weight,
                    layer.encoder_attn_layer_norm.bias,
                    layer.encoder_attn.k_proj.weight.T.contiguous(),
                    th.zeros_like(layer.encoder_attn.q_proj.bias),
                    layer.encoder_attn.q_proj.weight.T.contiguous(),
                    layer.encoder_attn.q_proj.bias,
                    layer.encoder_attn.v_proj.weight.T.contiguous(),
                    layer.encoder_attn.v_proj.bias,
                    layer.encoder_attn.out_proj.weight.T.contiguous(),
                    layer.encoder_attn.out_proj.bias,
                    layer.final_layer_norm.weight,
                    layer.final_layer_norm.bias,
                    layer.fc1.weight.T.contiguous(),
                    layer.fc1.bias,
                    layer.fc2.weight.T.contiguous(),
                    layer.fc2.bias
                ]]
        decoder_weights += [weight.to("cuda") for weight in [self.base_model.decoder.layer_norm.weight, self.base_model.decoder.layer_norm.bias]]
        self.ft_decoder = th.classes.FasterTransformer.FTWhisperDecoder(config, decoder_weights)
        
        self.built_model = True
    def generate(self, inputs, previous_ids = None, prefix = None, input_ids = None):
        batch = inputs.shape[0]
        if not input_ids: 
            input_ids,input_lengths = self.prepare_inputs(previous_ids, prefix, batch)
        encoder_hidden_states = self.encoder.forward(inputs)
        return self.decoder.forward(encoder_hidden_states, input_ids, input_lengths, 0.0)
    def prepare_inputs(self, previous_ids, prefix):
        # previous ids and prompt not supported.
        return (th.tensor([self.bot_sequence], dtype = th.int32, device = 'cuda'), th.tensor([2]))
    def convert_config(self, batch, beam):
        return th.classes.FasterTransformer.FTWhisperConfig(
            batch,
            self.config.vocab_size + 1,
            self.config.num_mel_bins,
            self.config.encoder_layers,
            self.config.encoder_attention_heads,
            self.config.decoder_layers,
            self.config.decoder_attention_heads,
            self.config.decoder_ffn_dim,
            self.config.encoder_ffn_dim,
            self.config.max_source_positions * 2,
            self.config.max_target_positions, 
            self.config.d_model,
            beam,
            self.config.eos_token_id
        )
        
