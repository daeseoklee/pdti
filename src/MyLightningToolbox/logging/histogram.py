from abc import ABCMeta, abstractmethod

import pytorch_lightning as pl 
import torch



class HistogramWriter(pl.Callback, metaclass=ABCMeta):
    """
    """
    def __init__(self, write_diff=False, period=1):
        self.idx = 0
        self.count = 0 
        self.period = period
        self.write_diff = write_diff
        if write_diff:
            self.prev_data = {}

    def update_prev_data(self, nickname:str, data:torch.Tensor):
        self.prev_data[nickname] = data.clone()

    def get_diff(self, nickname, data: torch.Tensor):
        if self.idx == 0:
            return torch.zeros_like(data)
        return data - self.prev_data[nickname]

    @abstractmethod
    def get_nicknames(self):
        pass
    
    @abstractmethod
    def get_data(self, pl_module:"pl.LightningModule", nickname):
        pass

    def write_data(self, pl_module:"pl.LightningModule", nickname, data):
        pl_module.logger.experiment.add_histogram(nickname, data, self.idx)
        if self.write_diff:
            diff = self.get_diff(nickname, data)
            pl_module.logger.experiment.add_histogram(f'{nickname}%diff', diff, self.idx)
            self.update_prev_data(nickname, data)

    def write(self, pl_module):
        for nickname in self.get_nicknames():
            data = self.get_data(pl_module, nickname)
            self.write_data(pl_module, nickname, data)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.count % self.period != 0:
            self.count += 1
            return super().on_validation_epoch_end(trainer, pl_module)
        self.write(pl_module)
        self.count += 1
        self.idx += 1
        return super().on_validation_epoch_end(trainer, pl_module)

class ActivationHistogramWriter(HistogramWriter):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cached = [] 
    def get_activations_to_record(self):
        pass
    def get_nicknames(self):
        pass
    def get_reduction_methods(self):
        return {}
    def get_data(self, pl_module:"pl.LightningModule", nickname):
        l = self.cached[nickname]

        reduction_method = self.reduction_methods[nickname]
        if reduction_method == 'flatten-cat':
            data = torch.cat([data_piece.flatten() for data_piece in l])
        else:
            raise Exception(f'No reduction method "{reduction_method}"')
        self.cached[nickname] = []
        return data
    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.activations_to_record = self.get_activations_to_record()
        self.nicknames = self.get_nicknames()
        self.cached = {nickname: [] for nickname in self.nicknames}
        
        self.reduction_methods = self.get_reduction_methods()
        for nickname in self.nicknames:
            if not nickname in self.reduction_methods:
                self.reduction_methods[nickname] = 'flatten-cat' 

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs, batch, batch_idx: int, dataloader_idx: int) -> None:

        if not 'activations' in outputs:
            raise Exception('ActivationHistogramWriter expects "activations" key from LightningModule.validation_step() outputs')
        activations = outputs['activations']
        for nickname in self.nicknames:
            data_piece = activations[nickname]
            self.cached[nickname].append(data_piece)
        return super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)

class WeightHistogramWriter(HistogramWriter):
    """
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_nicknames(self):
        return self.nickname_to_name.keys()

    def get_data(self, pl_module:"pl.LightningModule", nickname):
        name = self.nickname_to_name[nickname]
        data = self.get_weight_reduction(nickname, pl_module.get_parameter(name).data)
        return data

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        name_to_nickname = self.get_weight_name_to_nickname(pl_module)
        self.nickname_to_name = {val: key for key, val in name_to_nickname.items()}
        return super().on_fit_start(trainer, pl_module)
    
    #change this in subclasses
    def get_weight_name_to_nickname(self, pl_module: "pl.LightningModule"):
        return {name: name for name, _ in pl_module.named_parameters()}
    
    #change this in subclasses
    def get_weight_reduction(self, nickname:str, t:torch.Tensor):
        return t 


    




