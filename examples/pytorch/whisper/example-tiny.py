# %% 
import torch
import sys
from transformers import AutoProcessor
from utils import WhisperForConditionalGeneration
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")



ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

input_features = inputs.input_features

generated_ids = model.generate(inputs=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Transcription:{transcription}")
# %%
