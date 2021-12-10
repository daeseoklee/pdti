

from typing import Any, Callable, List, Optional

from torch import Tensor
import torch

from torchmetrics.functional.classification.auc import _auc_compute
from torchmetrics.functional.classification.precision_recall_curve import _precision_recall_curve_compute, _precision_recall_curve_update
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat

"""
These were implemented using "torchmetrics.metric.Metric" as the superclass in order to seemlessly aggregate multi-gpu per-step computations 
"""



def _ci_compute(preds, target):
    """
    This may consume a lot of GPU memory (~ 4 * (val dataset length) ** 2 bytes)
    """
    
    n = len(target) 
    
    x = torch.broadcast_to(target[:, None], (n, n)) > torch.broadcast_to(target[None, :], (n, n))
    y = torch.broadcast_to(preds[:, None], (n, n)) > torch.broadcast_to(preds[None, :], (n, n))
    z = torch.broadcast_to(preds[:, None], (n, n)) == torch.broadcast_to(preds[None, :], (n, n))
    
    total = torch.sum(torch.logical_and(x, y)) + torch.sum(torch.logical_and(x, z)) * 0.5
    num = torch.sum(x)
    
    return total / num

class CI(Metric):
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.add_state('preds', default=[], dist_reduce_fx='cat')
        self.add_state('target', default=[], dist_reduce_fx='cat')

    def update(self, preds: Tensor, target: Tensor) -> None:  

        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        
        return _ci_compute(preds, target)

    @property
    def is_differentiable(self) -> bool:
        return False
    
class AUPR(Metric):
    threshold: float
    preds: List[Tensor]
    target: List[Tensor]

    def __init__(
        self,
        threshold: float,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.threshold = threshold
        self.add_state('preds', default=[], dist_reduce_fx='cat')
        self.add_state('target', default=[], dist_reduce_fx='cat')

    def update(self, preds: Tensor, target: Tensor) -> None:

        self.preds.append(preds)
        self.target.append(target)

    def compute(self) -> Tensor:
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        
        target = (target >= self.threshold).to(dtype=int)
        #   print(f'preds.shape: {preds.shape}')
        precision, recall, thresholds = _precision_recall_curve_compute(preds, target, 1, pos_label=1)
        
        return _auc_compute(recall, precision, reorder=False)

    @property
    def is_differentiable(self) -> bool:
        return False
    
    