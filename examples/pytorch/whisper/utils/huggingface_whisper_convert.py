# %%

import torch
import sys
import os
from typing import Dict
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../..")
sys.path.append(dir_path)
sys.path.append(dir_path + "/..")
from utils import WhisperForConditionalGeneration

# %%
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
sd : Dict[str, torch.Tensor] = model.state_dict()
for k,v in sd.items():
    torch.save(v, k + ".pt")
    

# %%
