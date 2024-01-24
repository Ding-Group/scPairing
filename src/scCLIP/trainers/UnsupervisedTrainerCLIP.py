import os
from pathlib import Path
from matplotlib.figure import Figure
import time
from typing import Mapping, Union, Tuple, Optional, List
import psutil
import logging

import numpy as np
import anndata
import torch
from torch import optim
# from torch.utils.tensorboard import SummaryWriter

from batch_sampler import CellSampler, TriCellSampler
from eval_utils import evaluate
from models import model
from logging_utils import initialize_logger, log_arguments
from .trainer_utils import train_test_split, train_test_split_cite, set_seed, _stats_recorder

_logger = logging.getLogger(__name__)


class UnsupervisedTrainer:
    """Unsupervised trainer for single-cell modeling.

    Sets up the random seed, dataset split, optimizer and logger, and executes
    training and evaluation loop.

    Attributes:
        attr_fname: a dict mapping attributes of the trainer (a model or an
            optimizer) to file name prefixes of checkpoints.
        model: the model to be trained.
        optimizer: the optimizer used to train the model.
        lr: the current learning rate.
        init_lr: the initial learning rate.
        lr_decay: the negative log of the decay rate of the learning rate.
            After each training step, lr = lr * exp(-lr_decay).
        batch_size: the training batch size.
        steps_per_epoch: #training steps to cover an epoch.
        device: device the model is on.
        step: current step.
        epoch: current epoch.
        seed: random seed.
        train_instance_name: name for this train instance for checkpointing.
        ckpt_dir: directory to store the logs, the checkpoints and the plots.
    
    New attributes:
        adata_1: first modality single-cell dataset
        adata_2: second modality single-cell dataset
        train_adata_mod1: the training data. Contains (1 - test_ratio) x 100% of
            adata_1.
        test_adata_mod2: the test data. Contains test_ratio x 100% of adata_2.
        counts_layer: layer in AnnData corresponding to raw counts. If None, use .X
    """

    attr_fname: Mapping[str, str] = dict(
        model = 'model',
        optimizer = 'opt'
    )

    @log_arguments
    def __init__(self,
        model: model.scCLIP,
        adata_1: anndata.AnnData,
        adata_2: anndata.AnnData,
        adata_3: Optional[anndata.AnnData] = None,
        counts_layer: Optional[Union[str, List[str]]] = None,
        transformed_obsm: Optional[Union[str, List[str]]] = None,
        ckpt_dir: Union[str, None] = None,
        test_ratio: float = 0.,
        data_split_seed: int = 1,
        init_lr: float = 5e-3,
        lr_decay: float = 6e-5,
        weight_decay: float = 0.,
        batch_size: int = 2000,
        train_instance_name: str = "scCLIP",
        restore_epoch: int = 0,
        seed: int = -1,
    ) -> None:
        """Initializes the UnsupervisedTrainer object.

        Args:
            model: the model to be trained.
            adata_1: the intact single-cell dataset (first modality).
            adata_2: the intact single-cell dataset (second modality).
            adata_3: the intact single-cell dataset (third modality), if there is one.
            counts_layer: layer corresponding to raw counts.
            transformed_obsm: key in obsm corresponding to transformed data.
            ckpt_dir: directory to store the logs, the checkpoints and the
                plots. If training from scratch (restore_epoch = 0), this would
                be the parent directory of the actual directory storing the
                checkpoints (self.ckpt_dir = ckpt_dir / train_instance_name);
                if restoring from checkpoints, this would be the directory
                holding the checkpoint files.
            test_ratio: ratio of the test data in adata.
            init_lr: the initial learning rate.
            lr_decay: the negative log of the decay rate of the learning rate.
                After each training step, lr = lr * exp(-lr_decay).
            batch_size: the training batch size.
            train_instance_name: name for this train instance for checkpointing.
            restore_epoch: the epoch to restore from ckpt_dir.
            seed: random seed.
        """

        if seed >= 0:
            set_seed(seed)

        self.model: model.scCLIP = model

        self.train_adata_1 = self.test_adata_1 = self.adata_1 = adata_1
        self.train_adata_2 = self.test_adata_2 = self.adata_2 = adata_2
        self.train_adata_3 = self.test_adata_3 = self.adata_3 = adata_3
        if test_ratio > 0:
            self.train_adata_1, self.test_adata_1, self.train_adata_2, self.test_adata_2 = \
                train_test_split_cite(adata_1, adata_2, test_ratio, seed=data_split_seed)
        
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
        self.steps_per_epoch = max(self.train_adata_1.n_obs / self.batch_size, 1)
        self.device = model.device
        self.step = self.epoch = 0
        self.seed = seed

        self.train_instance_name = train_instance_name
        if restore_epoch > 0 and type(self) == UnsupervisedTrainer:
            self.ckpt_dir = ckpt_dir
            self.load_ckpt(restore_epoch, self.ckpt_dir)
        elif ckpt_dir is not None and restore_epoch == 0:
            self.ckpt_dir = os.path.join(ckpt_dir, f"{self.train_instance_name}_{time.strftime('%m_%d-%H_%M_%S')}")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            initialize_logger(self.ckpt_dir)
            _logger.info(f'ckpt_dir: {self.ckpt_dir}')
        else:
            self.ckpt_dir = None

    @log_arguments
    def load_ckpt(self, restore_epoch: int, ckpt_dir: Union[str, None] = None) -> None:
        """Loads model checkpoints.

        After loading, self.step, self.epoch and self.lr are set to
        the corresponding values, and the loger will be re-initialized.

        Args:
            restore_epoch: the epoch to restore.
            ckpt_dir: the directory containing the model checkpoints. If None,
                set to self.ckpt_dir.
        """

        if ckpt_dir is None:
            ckpt_dir = self.ckpt_dir
        assert ckpt_dir is not None and os.path.exists(ckpt_dir), f"ckpt_dir {ckpt_dir} does not exist."
        for attr, fname in self.attr_fname.items():
            fpath = os.path.join(ckpt_dir, f'{fname}-{restore_epoch}')
            getattr(self, attr).load_state_dict(torch.load(fpath))
        _logger.info(f'Parameters and optimizers restored from {ckpt_dir}.')
        initialize_logger(self.ckpt_dir)
        _logger.info(f'ckpt_dir: {self.ckpt_dir}')
        self.update_step(restore_epoch * self.steps_per_epoch)

    @staticmethod
    def _calc_weight(
        epoch: int,
        n_epochs: int,
        cutoff_ratio: float = 0.,
        warmup_ratio: float = 1/3,
        min_weight: float = 0.,
        max_weight: float = 1e-7
    ) -> float:
        """Calculates weights.

        Args:
            epoch: current epoch.
            n_epochs: the total number of epochs to train the model.
            cutoff_ratio: ratio of cutoff epochs (set weight to zero) and
                n_epochs.
            warmup_ratio: ratio of warmup epochs and n_epochs.
            min_weight: minimum weight.
            max_weight: maximum weight.

        Returns:
            The current weight of the KL term.
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

        Args:
            jump_to_step: the step number to jump to. If None, increment the
                step number by one.
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
        n_epochs: int = 800,
        eval_every: int = 200,
        need_reconstruction: bool = True,
        kl_warmup_ratio: float = 0.,
        min_kl_weight: float = 0.,
        max_kl_weight: float = 1e-5,
        batch_col: str = "batch_indices",
        save_model_ckpt: bool = True,
        ping_every: Optional[int] = None,
        record_log_path: Union[str, None] = None,
        writer = None,
        **train_kwargs
    ) -> None:
        """Trains the model, optionally evaluates performance and logs results.

        Args:
            n_epochs: the total number of epochs to train the model.
            eval_every: evaluate the model every this many epochs.
            n_samplers: #samplers (#threads) to use to sample training
                minibatches.
            kl_warmup_ratio: ratio of KL warmup epochs and n_epochs.
            min_kl_weight: minimum weight of the KL term.
            max_kl_weight: maximum weight of the KL term.
            eval: whether to evaluate the model.
            batch_col: a key in adata.obs to the batch column.
            save_model_ckpt: whether to save the model checkpoints.
            record_log_path: file path to log the training records. If None, do
                not log.
            writer: an initialized SummaryWriter for tensorboard logging.
            eval_result_log_path: file path to log the evaluation results. If
                None, do not log.
            eval_kwargs: kwargs to pass to the evaluate function.
            train_kwargs: kwargs to pass to self.do_train_step().

        New parameters:
            reconstruct_warmup_ratio: ratio of contrastive warmup epochs to n_epochs
            reconstruct_cutoff_ratio: ratio of cutoff ratios to n_epochs
            min_reconstruct_weight: minimum weight of reconstruction term
            max_reconstruct_weight: maximum weight of reconstruction term
            flip_contrastive_reconstruct: proportion of epochs at which to switch from
                training using contrastive loss to using reconstructive loss
            flip_clip_dist: proporition of epochs at which to switch from training using
                clip loss to distance-based loss.
        """
        if ping_every is None:
            ping_every = n_epochs
        
        # set up sampler and dataloader
        # if n_samplers == 1 or self.batch_size >= self.train_adata_1.n_obs:
        if self.adata_3 is None:
            sampler = CellSampler(
                self.train_adata_1,
                self.train_adata_2,
                self.batch_size,
                require_counts=need_reconstruction,
                counts_layer=self.counts_layer,
                transformed_obsm=self.transformed_obsm,
                sample_batch_id=self.model.need_batch,
                n_epochs=n_epochs - self.epoch,
                batch_col=batch_col
            )
        else:
            sampler = TriCellSampler(
                self.train_adata_1,
                self.train_adata_2,
                self.train_adata_3,
                self.batch_size,
                require_counts=need_reconstruction,
                counts_layer=self.counts_layer,
                transformed_obsm=self.transformed_obsm,
                sample_batch_id=self.model.need_batch,
                n_epochs=n_epochs - self.epoch,
                batch_col=batch_col
            )
        dataloader = iter(sampler)
        
        # set up the stats recorder
        # if os.path.exists(os.path.join(self.ckpt_dir, 'stats.csv')):
        #     log_file = open(os.path.join(self.ckpt_dir, 'stats.csv'), 'a')
        # else:
        #     log_file = open(os.path.join(self.ckpt_dir, 'stats.csv'), 'w')
        recorder = _stats_recorder(record_log_path=record_log_path, writer=writer, metadata=self.adata_1.obs)
        next_ckpt_epoch = min(int(np.ceil(self.epoch / eval_every) * eval_every), n_epochs)
        next_ping_epoch = min(int(np.ceil(self.epoch / ping_every) * ping_every), n_epochs)

        # with open(os.path.join(self.ckpt_dir, 'device.txt'), 'w') as f:
        #     f.write(str(self.model.device))

        while self.epoch < n_epochs:
            # Train one epoch
            new_record, hyper_param_dict = self.do_train_step(dataloader,
                n_epochs = n_epochs,
                kl_warmup_ratio=kl_warmup_ratio,
                min_kl_weight=min_kl_weight,
                max_kl_weight=max_kl_weight,
                **train_kwargs
            )
            # if self.epoch == 0:
                # log_file.write(','.join(['epoch'] + list(new_record.keys())) + '\n')
            # log_file.write(','.join(map(str, [self.epoch] + list(new_record.values()))) + '\n')
            recorder.update(new_record, self.epoch, n_epochs, next_ckpt_epoch)
            self.update_step()  # updates the learning rate

            # log and evaluate
            if self.epoch >= next_ckpt_epoch or self.epoch >= n_epochs:
                _logger.info('=' * 10 + f'Epoch {next_ckpt_epoch:.0f}' + '=' * 10)

                # log memory cost
                # _logger.info(repr(psutil.Process().memory_info()))
                # log current lr and kl_weight
                if self.lr_decay:
                    _logger.info(f'{"lr":12s}: {self.lr:12.4g}')
                for k, v in hyper_param_dict.items():
                    _logger.info(f'{k:12s}: {v:12.4g}')

                # log statistics of tracked items
                recorder.log_and_clear_record()
                if self.test_adata_1 is not self.adata_1:
                    test_nll = self.model.get_cell_embeddings_and_nll(self.test_adata_1, self.test_adata_2, self.batch_size, batch_col=batch_col, emb_names=[])
                    if test_nll is not None:
                        _logger.info(f'test nll: {test_nll:7.4f}')
                else:
                    test_nll = None

                if next_ckpt_epoch and save_model_ckpt and self.ckpt_dir is not None:
                    # checkpointing
                    self.save_model_and_optimizer(next_ckpt_epoch)

                _logger.info('=' * 10 + f'End of evaluation' + '=' * 10)
                next_ckpt_epoch = min(eval_every + next_ckpt_epoch, n_epochs)

            if self.epoch >= next_ping_epoch:
                # with open(os.path.join(self.ckpt_dir, 'ping.txt'), 'w') as f:
                #     f.write(str(self.epoch))
                next_ping_epoch = ping_every + next_ping_epoch

        # log_file.close()
        del recorder
        _logger.info("Optimization Finished: %s" % self.ckpt_dir)

    def save_model_and_optimizer(self, next_ckpt_epoch: int) -> None:
        """Docstring (TODO)
        """

        for attr, fname in self.attr_fname.items():
            torch.save(getattr(self, attr).state_dict(), os.path.join(self.ckpt_dir, f'{fname}-{next_ckpt_epoch}'))

    def do_train_step(self, dataloader, **kwargs) -> Mapping[str, torch.Tensor]:
        """Docstring (TODO)
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
