import pytorch_lightning as pl
import torch

class ModelCheckpointCallback(pl.Callback):
    def __init__(self, weight_from:str):
        self.weight_from = weight_from 
    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        checkpoint = torch.load(self.weight_from)
        state_dict = checkpoint['state_dict']
        pl_module.load_state_dict(state_dict) 
        return super().on_train_start(trainer, pl_module)