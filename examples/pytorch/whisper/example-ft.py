# %%
import torch as th
import einops as e

th.classes.load_library("/workspaces/FasterTransformer/build/lib/libth_transformer.so")
# %%

from transformers import AutoProcessor
from utils import WhisperForConditionalGeneration
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")


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