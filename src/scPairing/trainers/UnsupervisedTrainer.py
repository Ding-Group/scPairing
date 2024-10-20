import logging
import os
import time
from typing import List, Mapping, Optional, Union

import anndata
import numpy as np
import torch
from torch import optim

from ..batch_sampler import CellSampler
from ..logging_utils import initialize_logger, log_arguments
from ..models import model
from .trainer_utils import _stats_recorder

_logger = logging.getLogger(__name__)


class UnsupervisedTrainer:
    """Unsupervised trainer for single-cell modeling.

    Sets up the random seed, dataset split, optimizer and logger, and executes
    training and evaluation loop.

    Parameters
    ----------
    model
        The model to be trained.
    adata1
        AnnData object corresponding to the first modality of a multimodal single-cell dataset.
    adata2
        AnnData object corresponding to the second modality of a multimodal single-cell dataset.
    adata3
        AnnData object corresponding to the third modality of a multimodal single-cell dataset, if it exists.
    counts_layer
        Key(s) in ``adata1.layers`` and ``adata2.layers`` corresponding to the raw counts for each modality.
        If a string is provided, the same key will be applied to both ``adata1.layers`` and ``adata2.layers``.
        If ``None`` is provided, raw counts will be taken from ``adata1.X`` and/or ``adata2.X``.
    transformed_obsm
        Key(s) in ``adata1.obsm`` and ``adata2.obsm`` corresponding to the low-dimension
        representations of each individual modality. If a string is provided, the same key will
        be applied to both ``adata1.obsm`` and ``adata2.obsm``.
    ckpt_dir
        Directory to store the logs and the checkpoints.
    init_lr
        Initial learning rate.
    lr_decay
        The negative log of the decay rate of the learning rate.
        After each training step, lr = lr * ``exp(-lr_decay)``.
    batch_size
        Minibatch size.
    train_instance_name
        Name for this train instance for checkpointing.
    """

    attr_fname: Mapping[str, str] = dict(
        model = 'model',
        optimizer = 'opt'
    )

    @log_arguments
    def __init__(self,
        model: model.Model,
        adata1: anndata.AnnData,
        adata2: anndata.AnnData,
        adata3: Optional[anndata.AnnData] = None,
        counts_layer: Optional[Union[str, List[str]]] = None,
        transformed_obsm: Optional[Union[str, List[str]]] = None,
        ckpt_dir: Union[str, None] = None,
        init_lr: float = 5e-3,
        lr_decay: float = 6e-5,
        weight_decay: float = 0.,
        batch_size: int = 2000,
        train_instance_name: str = "scPairing"
    ) -> None:
        self.model: model.Model = model

        self.adata1 = adata1
        self.adata2 = adata2
        self.adata3 = adata3

        self.counts_layer = counts_layer

        self.transformed_obsm = transformed_obsm

        no_decay = list()
        decay = list()
        discriminator_params = list()
        for name, m in model.named_parameters():
            if 'decoder' in name and 'discriminator' not in name:
                decay.append(m)
            elif 'discriminator' not in name:
                no_decay.append(m)
            else:
                discriminator_params.append(m)
        self.optimizer = optim.Adam([
                {'params': no_decay},
                {'params': decay, 'weight_decay': weight_decay}
            ],
            lr=init_lr,
        )
        self.discriminator_optimizer = optim.Adam(discriminator_params) if len(discriminator_params) > 0 else None

        self.lr = self.init_lr = init_lr
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.steps_per_epoch = max(self.adata1.n_obs / self.batch_size, 1)
        self.device = model.device
        self.step = self.epoch = 0

        self.train_instance_name = train_instance_name
        if ckpt_dir is not None:
            self.ckpt_dir = os.path.join(ckpt_dir, f"{self.train_instance_name}_{time.strftime('%m_%d-%H_%M_%S')}")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            initialize_logger(self.ckpt_dir)
            _logger.info(f'ckpt_dir: {self.ckpt_dir}')
        else:
            self.ckpt_dir = None

    @staticmethod
    def _calc_weight(
        epoch: int,
        n_epochs: int,
        cutoff_ratio: float = 0.,
        warmup_ratio: float = 1/3,
        min_weight: float = 0.,
        max_weight: float = 1e-7
    ) -> float:
        """Returns the current weight for a loss term.

        Parameters
        ----------
        epoch
            Current epoch.
        n_epochs
            The total number of epochs to train the model.
        cutoff_ratio
            Ratio of cutoff epochs (set weight to zero) and ``n_epochs``.
        warmup_ratio
            Ratio of warmup epochs and ``n_epochs``.
        min_weight
            Minimum weight.
        max_weight
            Maximum weight.
        """

        fully_warmup_epoch = n_epochs * warmup_ratio
        if cutoff_ratio > warmup_ratio:
            _logger.warning(f'Cutoff_ratio {cutoff_ratio} is bigger than warmup_ratio {warmup_ratio}. This may not be an expected behavior.')
        if epoch < n_epochs * cutoff_ratio:
            return 0.
        if warmup_ratio:
            return max(min(1., epoch / fully_warmup_epoch) * max_weight, min_weight)
        else:
            return max_weight

    def update_step(self, jump_to_step: Union[None, int] = None) -> None:
        """Aligns the current step, epoch and lr to the given step number.

        Parameters
        ----------
        jump_to_step
            The step number to jump to. If None, increment the step number by one.
        """

        if jump_to_step is None:
            self.step += 1
        else:
            self.step = jump_to_step
        self.epoch = self.step / self.steps_per_epoch
        if self.lr_decay:
            if jump_to_step is None:
                self.lr *= np.exp(-self.lr_decay)
            else:
                self.lr = self.init_lr * np.exp(-jump_to_step * self.lr_decay)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    @log_arguments
    def train(self,
        n_epochs: int = 300,
        eval_every: int = 300,
        need_reconstruction: bool = True,
        kl_warmup_ratio: float = 0.,
        min_kl_weight: float = 0.,
        max_kl_weight: float = 1e-3,
        batch_col: str = "batch_indices",
        save_model_ckpt: bool = True,
        ping_every: Optional[int] = None,
        record_log_path: Union[str, None] = None,
        writer = None,
        seed: Optional[int] = None
    ) -> None:
        """Trains the model, optionally evaluates performance and logs results.

        Parameters
        ----------
        n_epochs
            The total number of epochs to train the model.
        eval_every
            Evaluate the model every this many epochs.
        kl_warmup_ratio
            Ratio of KL warmup epochs and n_epochs.
        min_kl_weight
            Minimum weight of the KL term.
        max_kl_weight
            Maximum weight of the KL term.
        batch_col
            A key in every AnnData's `.obs` for the batch column.
        save_model_ckpt
            Whether to save the model checkpoints.
        record_log_path: file path to log the training records. If None, do
                not log.
        seed
            Random seed.
        """
        if ping_every is None:
            ping_every = n_epochs
        
        # set up sampler and dataloader
        sampler = CellSampler(
            self.adata1,
            self.adata2,
            self.adata3,
            batch_size=self.batch_size,
            require_counts=need_reconstruction,
            counts_layer=self.counts_layer,
            transformed_obsm=self.transformed_obsm,
            sample_batch_id=self.model.need_batch,
            n_epochs=n_epochs - self.epoch,
            batch_col=batch_col,
            rng=np.random.default_rng(seed)
        )
        dataloader = iter(sampler)
        
        recorder = _stats_recorder(record_log_path=record_log_path, writer=writer, metadata=self.adata1.obs)
        next_ckpt_epoch = min(int(np.ceil(self.epoch / eval_every) * eval_every), n_epochs)

        while self.epoch < n_epochs:
            # Train one epoch
            new_record, hyper_param_dict = self.do_train_step(dataloader,
                n_epochs = n_epochs,
                kl_warmup_ratio=kl_warmup_ratio,
                min_kl_weight=min_kl_weight,
                max_kl_weight=max_kl_weight,
            )
            recorder.update(new_record, self.epoch, n_epochs, next_ckpt_epoch)
            self.update_step()  # updates the learning rate

            # log and evaluate
            if self.epoch >= next_ckpt_epoch or self.epoch >= n_epochs:
                _logger.info('=' * 10 + f'Epoch {next_ckpt_epoch:.0f}' + '=' * 10)

                # log current lr and kl_weight
                if self.lr_decay:
                    _logger.info(f'{"lr":12s}: {self.lr:12.4g}')
                for k, v in hyper_param_dict.items():
                    _logger.info(f'{k:12s}: {v:12.4g}')

                # log statistics of tracked items
                recorder.log_and_clear_record()

                if next_ckpt_epoch and save_model_ckpt and self.ckpt_dir is not None:
                    # checkpointing
                    self.save_model_and_optimizer(next_ckpt_epoch)

                _logger.info('=' * 10 + 'End of evaluation' + '=' * 10)
                next_ckpt_epoch = min(eval_every + next_ckpt_epoch, n_epochs)

        # log_file.close()
        del recorder
        _logger.info("Optimization Finished: %s" % self.ckpt_dir)

    def save_model_and_optimizer(self, next_ckpt_epoch: int) -> None:
        """Saves the model and optimizer in the checkpoint directory.

        Parameters
        ----------
        next_ckpt_epoch
            The epoch which the model and optimizer is saved at.
        """

        for attr, fname in self.attr_fname.items():
            torch.save(getattr(self, attr).state_dict(), os.path.join(self.ckpt_dir, f'{fname}-{next_ckpt_epoch}'))

    def do_train_step(self, dataloader, **kwargs) -> Mapping[str, torch.Tensor]:
        """Train the model for one step

        Parameters
        ----------
        dataloader
            Batch sampler to get training data.
        **kwargs
            Keyword arguments for constructing hyperparameters.
        """

        # construct hyper_param_dict
        hyper_param_dict = {
            'kl_weight': self._calc_weight(self.epoch, kwargs['n_epochs'], 0, kwargs['kl_warmup_ratio'], kwargs['min_kl_weight'], kwargs['max_kl_weight'])
            # 'batch_weight': self._calc_weight(self.epoch, kwargs['n_epochs'], 0, 1, 1, 100)
        }

        # construct data_dict
        data_dict = {k: v.to(self.device) for k, v in next(dataloader).items()}

        # train for one step, record tracked items (e.g. loss)
        new_record = self.model.train_step([self.optimizer, self.discriminator_optimizer] , data_dict, hyper_param_dict)

        return new_record, hyper_param_dict
