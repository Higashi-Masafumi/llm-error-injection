# %%
# モジュールを認識するためのパス追加
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# %%
# 必要なライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# modelの読み込み
from llada import LLaDAModelLM
from dream import DreamModel
LLADA_PRETRAINED_NAME = "GSAI-ML/LLaDA-8B-Instruct"
DREAM_PRETRAINED_NAME = "Dream-org/Dream-v0-Instruct-7B"

llada = LLaDAModelLM.from_pretrained(LLADA_PRETRAINED_NAME)
dream = DreamModel.from_pretrained(DREAM_PRETRAINED_NAME)

# %%
# モデルの構造確認
print("LLaDA Model Structure:")
llada.model

# %%
print("Dream Model Structure:")
dream.model

# %%
from transformers import AutoTokenizer
from torchview import draw_graph
llada_tokenizer = AutoTokenizer.from_pretrained(LLADA_PRETRAINED_NAME, trust_remote_code=True)
inputs = llada_tokenizer("Hello, my dog is cute", return_tensors="pt")
model_graph = draw_graph(llada.model, inputs)
model_graph.visual_graph

# %%
dream_tokenizer = AutoTokenizer.from_pretrained(DREAM_PRETRAINED_NAME, trust_remote_code=True)
inputs = dream_tokenizer("Hello, my dog is cute", return_tensors="pt")
dream.eval()
model_graph = draw_graph(dream.model, inputs, device="cpu")
model_graph.visual_graph

# %%
