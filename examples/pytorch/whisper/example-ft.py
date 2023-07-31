# %%
# %%
import torch as th
import einops as e

th.classes.load_library("/workspaces/FasterTransformer/build/lib/libth_transformer.so")
# %%

from transformers import AutoProcessor
from utils import WhisperForConditionalGeneration, WhisperDecoder
from datasets import load_dataset
from utils.tokenizer import Tokenizer, get_encoding

processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
# %%
class PrintingDecoder(WhisperDecoder):
    def forward(self,*args, **kwargs):
        print(kwargs['input_ids'])
        return WhisperDecoder.forward(self, *args, **kwargs)
class PrintingWhisper(WhisperForConditionalGeneration):
    def beam_search(self, *args, **kwargs):
        # print('beam_search')
        return WhisperForConditionalGeneration.beam_search(self, *args, **kwargs)
    def forward(self, *args, **kwargs):
        self.base_model.decoder.__class__ = PrintingDecoder
        # print('use cache' if kwargs['past_key_values'] else "no cache")
        return WhisperForConditionalGeneration.forward(self, *args, **kwargs)
# model = PrintingWhisper.from_pretrained("openai/whisper-tiny.en")

# %%
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

input_features = inputs.input_features


generated_ids = model.generate(inputs=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Transcription:{transcription}")

# %%
weights = [weight.to("cuda") for weight in [ model.base_model.encoder.conv1.weight
          , model.base_model.encoder.conv1.bias
          , model.base_model.encoder.conv2.weight
          , model.base_model.encoder.conv2.bias]]
# %%
for layer in model.base_model.encoder.layers:
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
    [   model.base_model.encoder.layer_norm.weight
    ,   model.base_model.encoder.layer_norm.bias]]

# %%
th.cuda.set_device(0)
# %%
ft_whisper = th.classes.FasterTransformer.FTWhisperEncoder(weights)
# %%
input_features_r = e.rearrange(input_features, "b c s -> b s c").contiguous().to("cuda")

# %%
ret = ft_whisper.forward(input_features_r, th.tensor([3000]))
# %%
import torch.nn as nn

hfret1 = e.rearrange(nn.functional.gelu(
    model.base_model.encoder.conv2(
        nn.functional.gelu(
            model.base_model.encoder.conv1(input_features))))
    ,   "b c s -> b s c") + model.base_model.encoder.embed_positions.weight

hfret1 = model.base_model.encoder.layers[0].self_attn(model.base_model.encoder.layers[0].self_attn_layer_norm(hfret1))
diff1 = hfret1[0] - (ret.to('cpu') + model.base_model.encoder.layers[0].self_attn.out_proj.bias)
# %%

hfret = model.base_model.encoder(input_features).last_hidden_state
th.cuda.synchronize()

# %%
diff = hfret - ret.to('cpu')
# %%
decoder_weights = [
    weight.to("cuda") for weight in
    [
        model.base_model.decoder.embed_tokens.weight,
        model.base_model.decoder.embed_positions.weight
    ]

]
# %%
for layer in model.base_model.decoder.layers:
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
decoder_weights += [weight.to("cuda") for weight in [model.base_model.decoder.layer_norm.weight, model.base_model.decoder.layer_norm.bias]]
# %%
ft_whisper_decoder = th.classes.FasterTransformer.FTWhisperDecoder(decoder_weights)
# %%
# print(f"hf logits: {model.base_model.decoder.forward(encoder_hidden_states = hfret, input_ids = th.zeros((1,1), dtype = th.int32))[0]}")

# %%
th.inference_mode()
out_seq = 448
tokenizer = Tokenizer(get_encoding('gpt2'))
tokenizer.language = 'en'
init_seq = th.tensor([[tokenizer.sot, tokenizer.no_timestamps]], dtype = th.int32, device = 'cuda')
decoder_ret = ft_whisper_decoder.forward(ret, init_seq, th.tensor([2],dtype=th.int, device="cuda" ), 0.0)
# %%
# %%

