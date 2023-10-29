

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

processor = AutoProcessor.from_pretrained("openai/whisper-large")

# %%
from utils import FTWhisperForConditionalGeneration
# %%

model = FTWhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
# %%
model.cuda(1,5)
# %%

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

input_features = inputs.input_features

# %%

input_features_r = e.rearrange(input_features, "b c s -> b s c").contiguous().to("cuda")

ret = model.ft_encoder.forward(input_features_r)

hfret = model.base_model.encoder(input_features).last_hidden_state
th.cuda.synchronize()

diff = hfret - ret.to('cpu')
print(f"encoder diff: {diff} \n")
# %%
generated_ids = model.generate(inputs=input_features)
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Transcription:{transcription}")
# %%
