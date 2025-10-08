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
