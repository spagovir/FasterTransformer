# %% 
import torch
import sys
from transformers import AutoProcessor
from utils import WhisperForConditionalGeneration
from utils import FTWhisperForConditionalGeneration
from datasets import load_dataset
import numpy as np

processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
model = FTWhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.cuda(5,5)



ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
for i in range(0,10):
    input_features = torch.from_numpy(np.stack(processor([item["array"] for item in ds[5*i:5*i+5]["audio"]]).input_features))
    print(input_features.shape)
    generated_ids = model.generate(inputs=input_features)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(f"Transcription:{transcription}")
# %%
