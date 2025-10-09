import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import DeviceStatsMonitor, RichProgressBar, Callback
from pytorch_lightning.demos import Transformer, WikiText2
from pytorch_lightning.loggers import WandbLogger
from torch.optim import Optimizer
from torch.utils.data import DataLoader


# Define the model
class LanguageModel(pl.LightningModule):
    def __init__(self, vocab_size: int = 4096):
        super().__init__()
        self.model = Transformer(
            vocab_size=vocab_size,
            nlayers=64,
            nhid=4096,
            ninp=1024,
            nhead=64,
        )

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        input, target = batch
        output = self.model(input, target)
        # ここでは
        loss = F.nll_loss(input=output, target=target.view(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

pl.seed_everything(42)
dataset = WikiText2()
train_loader = DataLoader(dataset, batch_size=64)
wandb_logger = WandbLogger(project="language-model-training")

model = LanguageModel(vocab_size=dataset.vocab_size)

trainer = pl.Trainer(
    accelerator="cuda",
    devices=1,
    logger=[wandb_logger],
    callbacks=[RichProgressBar(), DeviceStatsMonitor()],
    max_epochs=3
)
trainer.fit(
    model=model,
    train_dataloaders=train_loader
)
trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

