from paths import paths
from train.pdti_dataset import PdtiDataset
from model.pdti_models import PdtiModel
from metric import CI, AUPR 

from argparse import ArgumentParser
from train.search_space import iter_search_space
import train.search_space as search_space

from typing import List, Dict, Tuple, Union, Any, Optional 
import json
from copy import deepcopy
from pathlib import Path

from MyLightningToolbox.schedulers.transformer_scheduler import get_scheduler, SchedulerLateTotalstepsSetter
from MyLightningToolbox.logging.histogram import WeightHistogramWriter, ActivationHistogramWriter
from MyLightningToolbox.logging.mylogger import MyLogger, MyLoggerCallback
from MyLightningToolbox.checkpoint.model_checkpoint import ModelCheckpointCallback

import torch 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import MSELoss
import torch.nn as nn 

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin

from asym.data_collection import DataCollection

if False:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

if True:
    torch.multiprocessing.set_sharing_strategy('file_system')

def load_json(filename):
    with open(filename, 'r') as reader:
        return json.load(reader)
    
class AlphaBetaHistogramWriter(WeightHistogramWriter):
    def __init__(self, softplus=True, **kwargs):
        super().__init__(**kwargs) 
        if softplus:
            self.softplus = nn.Softplus()
    def get_weight_name_to_nickname(self, pl_module: "pl.LightningModule"):
        assert type(pl_module) == PdtiTrain
        assert pl_module.model_config['pocket_emb']['model'].startswith('rayatt')
        model_config = pl_module.model_config['pocket_emb']['model_config']
        num_layers = int(model_config['RA']['num_layers'])
        d = {} 
        for i in range(num_layers):
            d[f'model.pocket_embedder.RA.units.{i}.RA.alpha'] = f'{i+1}th alpha'
            d[f'model.pocket_embedder.RA.units.{i}.RA.beta'] = f'{i+1}th beta'
        return d
    
    def get_weight_reduction(self, nickname:str, t:torch.Tensor):
        if hasattr(self, 'softplus'):
            return self.softplus(t.flatten())
        return t.flatten()

class PdtiActivationHistogramWriter(ActivationHistogramWriter):
    def __init__(self, args, **kwargs):
        super().__init__(**kwargs)
        self.args = args
    def get_activations_to_record(self):
        activations_to_record = []
        if self.args.record_confidence:
            activations_to_record.append('pocket_confidence_scores')
        return activations_to_record
    def get_nicknames(self):
        nicknames = []
        if self.args.record_confidence:
            nicknames = nicknames + ['mean_confidence_score', 'min_confidence_score', 'max_confidence_score']
        return nicknames

class PdtiTrain(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.train_config = deepcopy(config['train_config'])
        self.dataset_config = deepcopy(config['dataset_config'])
        self.trn_dataset_config = self.dataset_config['trn']
        self.val_dataset_config = self.dataset_config['val']
        self.model_config = deepcopy(config['model_config'])
        
        self.model = PdtiModel(self.model_config)
        
        self.loss = MSELoss(reduction='mean')
        self.ci = CI()
        self.aupr = AUPR(self.train_config['aupr_threshold'])

        self.gpus = self.train_config['gpus']
        self.prefetch = self.train_config['prefetch']
    
    def train_dataloader(self):
        dataset = PdtiDataset(self.trn_dataset_config)
        
        batch_size = self.train_config['batch_size']

        if self.prefetch:
            kwargs = {'num_workers': 0,
                      'persistent_workers': False}
        else:
            num_workers = self.train_config['num_workers']
            kwargs = {'num_workers': num_workers, 
                     'persistent_workers': True}

        if self.gpus > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.gpus, rank=self.global_rank, shuffle=True)
            kwargs['sampler'] = sampler
        else:
            kwargs['shuffle'] = True
        
        return DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, **kwargs)
    
    def val_dataloader(self):
        dataset = PdtiDataset(self.val_dataset_config)
        
        batch_size = self.train_config['batch_size']
        
        if self.prefetch:
            kwargs = {'num_workers': 0,
                      'persistent_workers': False}
        else:
            num_workers = self.train_config['num_workers']
            kwargs = {'num_workers': num_workers, 
                     'persistent_workers': True}

        if self.gpus > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.gpus, rank=self.global_rank, shuffle=False)
            kwargs['sampler'] = sampler
        else:
            kwargs['shuffle'] = False

        return DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn, **kwargs)
    
    def on_train_start(self):
        if self.prefetch:
            self.train_dataloader().dataset.prefetch_pockets(self.device)
            self.val_dataloader().dataset.prefetch_pockets(self.device)

    def on_train_epoch_start(self):
        self.train_dataloader().sampler.set_epoch(self.current_epoch)
    
    def configure_optimizers(self):

        sched_config = deepcopy(self.train_config['sched_config'])
        optimizer = optim.AdamW(self.parameters(), lr=sched_config['max_lr'])
        sched:str = self.train_config['scheduler']
        
        scheduler = get_scheduler(optimizer, sched, **sched_config)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    def to_current_device(self, batch):
        dc, regrouper, p, m, affinity = batch
        if self.prefetch:
            dc, regrouper, p.to_device(self.device), m.to_device(self.device), affinity.to(device=self.device)
        return dc.to_device(self.device), regrouper, p.to_device(self.device), m.to_device(self.device), affinity.to(device=self.device)

    def training_step(self, train_batch, batch_idx):
        dc, regrouper, p, m, affinity = self.to_current_device(train_batch)
        
        prediction, _ = self.model(dc, regrouper, p, m, activations_to_record=[])
        loss = self.loss(prediction, affinity)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        dc, regrouper, p, m, affinity = self.to_current_device(val_batch)
        
        prediction, activations = self.model(dc, regrouper, p, m, activations_to_record=self.activations_to_record) #self.activations_to_record comes from the ActivationHistogramWriter callback. 
        loss = self.loss(prediction, affinity)
        
        self.ci(prediction, affinity)
        self.aupr(prediction, affinity)

        return {'loss': loss, 'activations': activations}
    
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()

        val_ci = self.ci.compute()
        self.ci.reset()
        val_aupr = self.aupr.compute()
        self.aupr.reset()
        
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True)
        self.log('val_ci', val_ci) #don't need sync_dist
        self.log('val_aupr', val_aupr) #don't need sync_dist
        
def build_model_config(args):
    if args.no_whole:
        pemb_config = None
    else:
        pemb_model_config = load_json(Path(paths.model_config) / 'protein_embedders' / 'cnn.json')
        pemb_config = {
            'model': 'cnn',
            'model_config': pemb_model_config
        }

    memb_config = load_json(Path(paths.model_config) / 'molecule_embedders' / 'bert.json')
    memb_pretrained = str(Path(paths.sample_pretrained_models) / 'memb' / 'last.ckpt')
    
    regs_config = load_json(Path(paths.model_config) / 'reg' / f'{args.reg_ver}.json')

    
    model_config = {
        'pocket_emb': {
            'model': args.model,
            'pretrained': args.pretrained,
            'model_config': load_json(args.model_config)
        },
        'protein_emb': pemb_config,
        'molecule_emb': {
            'model': 'bert',
            'model_config': memb_config,
            'pretrained': memb_pretrained
        }
    }
    
    model_config.update(regs_config)
    
    return model_config

def build_dataset_config(args):
    common = {
        'dataset': args.dataset,
        'mode': args.mode,
        'fold': args.fold,
        'unseen': args.unseen,
        'pocket_ver': args.pocket_ver,
        'num_thresholds': args.num_thresholds,
        'pemb_model': args.emb
    }
    
    trn_dataset_config = deepcopy(common)
    if args.trn_use_val:
        trn_dataset_config['split'] = 'val'
    else:
        trn_dataset_config['split'] = 'trn'
    val_dataset_config = deepcopy(common)
    val_dataset_config['split'] = 'val'
    
    return {
        'trn': trn_dataset_config,
        'val': val_dataset_config
    }

def build_train_config(args):
    aupr_threshold = 12.1 if args.dataset == 'kiba' else 7.0
    train_config = {
        'record_confidence': args.record_confidence,
        'prefetch': args.prefetch,
        'aupr_threshold': aupr_threshold,
        'weight_from': args.weight_from,
        'batch_size': args.batch,
        'epochs': args.epochs,
        'scheduler': args.scheduler,
        'num_workers': args.num_workers,
        'gpus': args.gpus,
        'accumulate': args.accumulate,
        'sched_config': {
            'max_lr': args.lr,
            'warmup_steps': args.warmup
        }
    } 
    return train_config

def get_subdir(args):
    if args.unseen:
        return f'unseen_{args.mode}{args.fold}'
    return f'seen_{args.mode}{args.fold}'

def get_logger_and_callbacks(args, model_config):
    metrics_to_log = ['val_loss', 'val_ci', 'val_aupr']
    logger = MyLogger(
        save_dir = paths.train_logs,
        name=args.experiment,
        version=args.version,
        sub_dir=get_subdir(args),
        metrics_to_log=metrics_to_log,
        train_info=['dataset_config', 'train_config'],
        model_info=['model_config'],
        srcfile_prefix='src/train'
    )
    mc = ModelCheckpoint(dirpath=logger.log_dir, save_last=True, monitor='val_loss', mode='min')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    logger_callback = MyLoggerCallback(logger)
    scheduler_callback = SchedulerLateTotalstepsSetter(length_from='sampler' if args.gpus > 1 else 'dataloader')
    callbacks = [mc, lr_monitor, logger_callback, scheduler_callback]

    softplus = model_config['pocket_emb']['model_config']['RA'].get('softplus', True)
    callbacks.append(AlphaBetaHistogramWriter(softplus=softplus, write_diff=False, period=1))
    callbacks.append(PdtiActivationHistogramWriter(args, write_diff=False, period=1))
    
    if args.weight_from is not None:
        callbacks.append(ModelCheckpointCallback(args.weight_from))

    return logger, callbacks

def train(args):
    model_config = build_model_config(args)
    dataset_config = build_dataset_config(args)
    train_config = build_train_config(args)
    config = {
        'model_config': model_config,
        'dataset_config': dataset_config,
        'train_config': train_config
    }
    
    pdti_train = PdtiTrain(config)
    
    logger, callbacks = get_logger_and_callbacks(args, model_config)
    
    additional_args = {'profiler': 'simple'}
    if args.gpus > 1:
        additional_args['accelerator'] = 'ddp'
        additional_args['plugins'] = DDPPlugin(find_unused_parameters=False)
    
    trainer = pl.Trainer(max_epochs=train_config['epochs'], logger=logger, callbacks=callbacks, precision=args.precision, gpus=args.gpus if args.gpu_skip is None else [(args.gpu_skip + i)%args.total_gpus for i in range(args.gpus)], accumulate_grad_batches=args.accumulate, replace_sampler_ddp=True, **additional_args) 

    trainer.fit(pdti_train)

def main(args):
    """
    Based on [args.search] and [args.fold], 
    generate instances of [Namespace] and feed them into train() 
    """
    if args.search is None:
        
        if args.version is None:
            raise Exception('Need to specify the version name')
        
        train(args)
        torch.cuda.empty_cache()
    else:
        search_fn = getattr(search_space, args.search)
        
        for i, new_args in enumerate(iter_search_space(args, search_fn)):
            if args.search_from is not None and i < args.search_from:
                continue
            assert new_args.search is None
            assert args.fold == new_args.fold
            main(new_args)
            
if __name__ == '__main__':
    parser = ArgumentParser()
    #dataset_config
    parser.add_argument('--dataset', type=str, required=True) #'kiba', 'davis' etc. 
    parser.add_argument('--mode', type=str, required=True) #cv, test
    parser.add_argument('--fold', type=int, required=True) #0~4
    parser.add_argument('--unseen', action='store_true')
    parser.add_argument('--emb', type=str, default='prot_bert')
    parser.add_argument('--pocket_ver', type=int, default=1)
    parser.add_argument('--num_thresholds', type=int, default=1)
    
    #model
    parser.add_argument('--model', type=str, help='The model name abbreviation', required=True)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--reg_ver', required=True)
    parser.add_argument('--no_whole', action='store_true')

    #workflow
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--search', type=str, help='Hyperparameter search function name in search_space.py')  
    parser.add_argument('--search_from', type=int, help='Skip this number of search cases')
    parser.add_argument('--weight_from', type=str)


    # Depends on the hardware spec
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--total_gpus', type=int, default=1, help='Number of gpus to use')
    parser.add_argument('--gpus', type=int, default=0, help='Number of gpus to use')
    parser.add_argument('--gpu_skip', type=int, help='Number of gpus already in use')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of CPU workers to use in Dataset and DataLoader')
    parser.add_argument('--precision', type=int, default=32)


    #learning
    parser.add_argument('--batch', type=int, help='Batch size per gpu', required=True)
    parser.add_argument('--lr', type=float, help='maximum lr', required=True)
    parser.add_argument('--accumulate', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--epochs', type=int, help='Number of epochs', required=True)

    parser.add_argument('--warmup', type=float, default=10000, help='warmup steps of the learning rate scheduler')
    parser.add_argument('--scheduler', type=str, default='cos', help='lin, cos or con')

    
    #debugging 
    parser.add_argument('--trn_use_val', action='store_true', help='Use the validation dataset for training (for debugging)')
    
    #visualization 
    parser.add_argument('--record_confidence', action='store_true')
    
    
    
    args=parser.parse_args()
    
    main(args)
