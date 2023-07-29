from transformers import WhisperForConditionalGeneration
import torch as th
import einops as e

th.classes.load_library("/workspaces/FasterTransformer/build/lib/libth_transformer.so")

class FTWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
    
    def cuda(self):
        weights = [weight.to("cuda") for weight in [ self.base_model.encoder.conv1.weight
                , self.base_model.encoder.conv1.bias
                , self.base_model.encoder.conv2.weight
                , self.base_model.encoder.conv2.bias]]
        # %%
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
        # %%
        weights += [ weight.to("cuda") for weight in 
            [   self.base_model.encoder.layer_norm.weight
            ,   self.base_model.encoder.layer_norm.bias]]

        # %%
        th.cuda.set_device(0)
        # %%
        self.ft_encoder = th.classes.FasterTransformer.FTWhisperEncoder(weights)
        

        decoder_weights = [
            weight.to("cuda") for weight in
            [
                self.base_model.decoder.embed_tokens.weight,
                self.base_model.decoder.embed_positions.weight
            ]

        ]
        # %%
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
        # %%
        self.ft_decoder = th.classes.FasterTransformer.FTWhisperDecoder(decoder_weights)
        
        self.built_model = True
    def generate(self, inputs, previous_ids = None, prefix = None, input_ids = None):
        if not input_ids: 
            input_ids = self.prepare_inputs(previous_ids, prefix)
        encoder_hidden_states = self.encoder.forward(inputs)
        return self.decoder.forward(encoder_hidden_states, input_ids, 0.0)
    def prepare_inputs(self, previous_ids, prefix):
        # previous ids and prompt not supported.
        forced_inputs = []
        forced_inputs.append(self.config.bos_token_id)
        forced_inputs.append()
        
