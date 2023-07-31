# %%
print(None)
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
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
# %%
class PrintingDecoder(WhisperDecoder):
    def forward(self,*args, **kwargs):
        inputs = kwargs["input_ids"]
        print(f"inputs: {inputs};\n")
        past_key_values = kwargs["past_key_values"]
        print(f"past_key_values: {past_key_values};\n")
        tok_embeds = self.embed_tokens(inputs)
        pos_embeds = self.embed_positions(inputs, past_key_values_length = past_key_values[0][0].shape[2] if past_key_values else 0)
        print(f"tok_embeds: {tok_embeds};\n")
        print(f"pos_embeds: {pos_embeds};\n")
        print(f"embeds: {tok_embeds + pos_embeds};\n")
        rets = WhisperDecoder.forward(self, *args, **kwargs)
        print("\nhids: \n")
        print(rets[0])
        print("\nlogits: \n")
        print(rets[0] @ self.embed_tokens.weight.T)
        input("\n")
        return rets
class PrintingWhisper(WhisperForConditionalGeneration):
    def beam_search(self, *args, **kwargs):
        # print('beam_search')
        return WhisperForConditionalGeneration.beam_search(self, *args, **kwargs)
    def forward(self, *args, **kwargs):
        self.base_model.decoder.__class__ = PrintingDecoder
        # print('use cache' if kwargs['past_key_values'] else "no cache")
        return WhisperForConditionalGeneration.forward(self, *args, **kwargs)
model = PrintingWhisper.from_pretrained("openai/whisper-tiny.en")

# %%
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

input_features = inputs.input_features

th.inference_mode(True)
generated_ids = model.generate(inputs=input_features)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f"Transcription:{transcription}")

# %%