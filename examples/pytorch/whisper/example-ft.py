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
th.cuda.set_device(0)
ft_whisper = th.classes.FasterTransformer.FTWhisperEncoder(weights)
# %%
input_features_r = e.rearrange(input_features, "b c s -> b s c").to("cuda")

# %%
ret = ft_whisper.forward(input_features_r, th.tensor([3000]))
# %%
import torch.nn as nn
hfret = e.rearrange(
    nn.functional.gelu(
        model.base_model.encoder.conv2(nn.functional.gelu(
        model.base_model.encoder.conv1(input_features)
        )))
    ,   "b c s -> b s c"
    )
th.cuda.synchronize()

# %%
