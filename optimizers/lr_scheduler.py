# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import List, Any, Union
from paddle.optimizer.lr import LRScheduler


class LinearWarmupCosineAnnealingLR(LRScheduler):

    def __init__(
            self,
            optim_lr: float,
            warmup_epochs: int,
            max_epochs: int,
            warmup_start_lr: float = 0.0,
            eta_min: float = 0.0,
            last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optim_lr (float): Initial learning rate.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.optim_lr = optim_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optim_lr, last_epoch)

    def get_lr(self) -> Union[Union[float, List[float]], Any]:
        """
        Compute learning rate using chainable form of the scheduler
        """
        if self.last_epoch == 0:
            return self.warmup_start_lr * len([self.base_lr])
        elif self.last_epoch < self.warmup_epochs:
            return self.optim_lr + (self.base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lr
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return self.optim_lr + (self.base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2

        return (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) / \
               (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))) * \
               (self.optim_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self) -> List[float]:
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return self.warmup_start_lr + self.last_epoch * (self.base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)

        return self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
                1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
