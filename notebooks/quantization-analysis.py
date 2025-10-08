# %%
from transformers import AutoModel, AutoTokenizer, HqqConfig

LLADA_PRETRAINED = "GSAI-ML/LLaDA-8B-Instruct"
hqq_config = HqqConfig(nbits=4)
llada = AutoModel.from_pretrained(LLADA_PRETRAINED, trust_remote_code=True, quantization_config=hqq_config, device_map="cuda:0")
llada_tokenizer = AutoTokenizer.from_pretrained(LLADA_PRETRAINED, trust_remote_code=True)

print("LLaDA Model Structure:")
llada.model

# %%
DREAM_PRETRAINED = "Dream-org/Dream-v0-Instruct-7B"
hqq_config = HqqConfig(nbits=4)
dream = AutoModel.from_pretrained(DREAM_PRETRAINED, trust_remote_code=True, quantization_config=hqq_config, device_map="cuda:0")
dream_tokenizer = AutoTokenizer.from_pretrained(DREAM_PRETRAINED, trust_remote_code=True)
print("Dream Model Structure:")
dream.model

# %%
