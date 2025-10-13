import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from block_diffusion import BD3LM


@hydra.main(config_path="../configs", config_name="blockdlm")
def main(cfg: DictConfig):
    model = BD3LM.from_pretrained(cfg.models.pretrained_name, trust_remote_code=True)
    print(OmegaConf.to_yaml(cfg))
    print("model structure:")
    print(model)

if __name__ == "__main__":
    main()
