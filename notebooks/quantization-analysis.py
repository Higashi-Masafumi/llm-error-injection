# %%
import torch
from transformers import AutoModel, AutoTokenizer, HqqConfig

LLADA_PRETRAINED = "GSAI-ML/LLaDA-8B-Instruct"
hqq_config = HqqConfig(nbits=4)
llada = AutoModel.from_pretrained(LLADA_PRETRAINED, trust_remote_code=True, quantization_config=hqq_config, device_map="cuda:0")
print("LLaDA Model Structure:")
llada.model
# memory usage
print(f"Model memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Model max memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print(f"Model reserved memory: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
# %%
import torch
from transformers import AutoModel, AutoTokenizer, HqqConfig
DREAM_PRETRAINED = "Dream-org/Dream-v0-Instruct-7B"
hqq_config = HqqConfig(nbits=4)
dream = AutoModel.from_pretrained(DREAM_PRETRAINED, trust_remote_code=True, quantization_config=hqq_config, device_map="cuda:0")
print("Dream Model Structure:")
dream.model
# memory usage
print(f"Model memory usage: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
print(f"Model max memory: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
print(f"Model reserved memory: {torch.cuda.memory_reserved()/1024**3:.2f} GB")

# %%
