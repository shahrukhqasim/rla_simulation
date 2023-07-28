import time
from pathlib import Path

import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from rlasim.lib.networks_conditional import MlpConditionalVAE, BaseVAE
import torch
import numpy as np
import pytorch_lightning as pl
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import argh

from rlasim.lib.organise_data import VaeDataset, ThreeBodyDecayDataset
from rlasim.lib.plotting import ThreeBodyDecayPlotter

Tensor = TypeVar('torch.tensor')

class ConditionalThreeBodyDecayVaeSimExperiment(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, plotter_params: dict) -> None:
        super(ConditionalThreeBodyDecayVaeSimExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.plotter_params = plotter_params
        self.curr_device = None

        if 'checkpoint_path' in params.keys():
            if not torch.cuda.is_available():
                self.load_state_dict(torch.load(params['checkpoint_path'], map_location=torch.device('cpu'))['state_dict'])
            else:
                self.load_state_dict(torch.load(params['checkpoint_path'])['state_dict'])

    def forward(self, input: Tensor, condition: Tensor, **kwargs) -> Tensor:
        x = self.model(input, condition, **kwargs)
        return x


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        self.curr_device = batch[0].device

        batch, condition = batch[0], batch[1]
        self.curr_device = batch.device
        preprocessor = self.trainer.datamodule.preprocessor
        batch_pp, condition_pp = preprocessor(batch, condition)
        output = self.forward(batch_pp, condition_pp)
        sampled = self.model.sample(len(condition_pp), self.curr_device, condition=condition_pp)

        reconstructed = output[0]
        print(reconstructed.shape, condition.shape)
        reconstructed_upp, _ = preprocessor(reconstructed, condition, direction=-1)
        sampled_upp, _ = preprocessor(sampled, condition, direction=-1)

        self.predict_results_dict['condition'] += [condition]
        self.predict_results_dict['true'] += [batch]
        self.predict_results_dict['reco'] += [reconstructed_upp]
        self.predict_results_dict['sampled'] += [sampled_upp]


    def on_predict_start(self) -> None:
        self.predict_results_dict = {
            'true':[],
            'reco':[],
            'sampled':[],
            'condition':[],
        }

    def on_predict_end(self) -> None:
        condition = torch.concatenate(self.predict_results_dict['condition'], dim=0)
        decays_true = torch.concatenate(self.predict_results_dict['true'], dim=0)
        decays_reco = torch.concatenate(self.predict_results_dict['reco'], dim=0)
        decays_sampled = torch.concatenate(self.predict_results_dict['sampled'], dim=0)

        self.produce_pdf(decays_true, condition, str='true', path=self.params['pdf_prefix']+'_true.pdf')
        self.produce_pdf(decays_reco, condition, str='reco', path=self.params['pdf_prefix']+'_reco.pdf')
        self.produce_pdf(decays_sampled, condition, str='sampled', path=self.params['pdf_prefix']+'_sampled.pdf')

        preprocessor = self.trainer.datamodule.preprocessor
        _, condition_pp = preprocessor(None, condition, direction=-1)

        filt = condition[:, 0, 2] <1000

        print(filt.shape, decays_true.shape, decays_reco.shape, condition.shape, decays_sampled.shape)
        print(self.params['pdf_prefix']+'_true_pm_1000.pdf')
        self.produce_pdf(decays_true[filt], condition[filt], str='true', path=self.params['pdf_prefix']+'_true_pm_1000.pdf')
        self.produce_pdf(decays_reco[filt], condition[filt], str='reco', path=self.params['pdf_prefix']+'_reco_pm_1000.pdf')
        self.produce_pdf(decays_sampled[filt], condition[filt], str='sampled', path=self.params['pdf_prefix']+'_sampled_pm_1000.pdf')


        # self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        batch, condition = batch[0], batch[1]
        self.curr_device = batch.device
        preprocessor = self.trainer.datamodule.preprocessor
        batch_pp, condition_pp = preprocessor(batch, condition)
        results = self.forward(batch_pp, condition_pp)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        # print(batch_idx, np.mean(batch_pp.cpu().numpy()), np.mean(condition_pp.cpu().numpy()), train_loss['loss_kld'], train_loss['loss_reco'])

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']


    def on_validation_start(self) -> None:
        if self.current_epoch % int(self.params['validate_after_epochs']) != 0:
            self.perform_validation=False
            return

        self.perform_validation=True
        self.validation_results_dict = {
            'true':[],
            'reco':[],
            'sampled':[],
            'condition':[],
        }


    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        if self.perform_validation:
            batch, condition = batch[0], batch[1]
            self.curr_device = batch.device
            preprocessor = self.trainer.datamodule.preprocessor
            batch_pp, condition_pp = preprocessor(batch, condition)
            output = self.forward(batch_pp, condition_pp)
            val_loss = self.model.loss_function(*output,
                                                M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                                optimizer_idx=optimizer_idx,
                                                batch_idx=batch_idx)

            self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

            sampled = self.model.sample(len(condition_pp), self.curr_device, condition=condition_pp)

            reconstructed = output[0]
            print(reconstructed.shape, condition.shape)

            reconstructed_upp, _ = preprocessor(reconstructed, condition, direction=-1)
            sampled_upp, _ = preprocessor(sampled, condition, direction=-1)

            self.validation_results_dict['condition'] += [condition]
            self.validation_results_dict['true'] += [batch]
            self.validation_results_dict['reco'] += [reconstructed_upp]
            self.validation_results_dict['sampled'] += [sampled_upp]


    def on_validation_end(self) -> None:
        if not self.perform_validation:
            return

        condition = torch.concatenate(self.validation_results_dict['condition'], dim=0)
        decays_true = torch.concatenate(self.validation_results_dict['true'], dim=0)
        decays_reco = torch.concatenate(self.validation_results_dict['reco'], dim=0)
        decays_sampled = torch.concatenate(self.validation_results_dict['sampled'], dim=0)

        self.produce_pdf(decays_true, condition, str='true')
        self.produce_pdf(decays_reco, condition, str='reco')
        self.produce_pdf(decays_sampled, condition, str='sampled')


        # if self.current_epoch % 10 == 0:
        #     self.sample_data(N=100)


    def produce_pdf(self, samples, samples_mother, log=False, str='samples', path = None, bins=50):
        try:
            samples = torch.reshape(samples*1, (-1, 3, 3))
            samples = samples.cpu().numpy()
            samples_mother = torch.reshape(samples_mother*1, (-1, 1, 3))
            samples_mother = samples_mother.cpu().numpy()

            if path is None:
                path = os.path.join(self.logger.log_dir,
                                       'samples',
                                       f"{self.logger.name}_{str}_Epoch_{self.current_epoch}.pdf")

            plotter = ThreeBodyDecayPlotter(**self.plotter_params)
            plotter.plot(samples, samples_mother, path)

        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.params['adam_lr'])
        return optimizer




def main(config_file='configs/vae_conditional.yaml', predict=False):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

    data = ThreeBodyDecayDataset(**config["data_params"])
    vae_network = MlpConditionalVAE(**config["network_params"])


    if predict:
        experiment = ConditionalThreeBodyDecayVaeSimExperiment(vae_network, config['generate_params'], config['plotter'])
        runner = Trainer()
        runner.predict(experiment, datamodule=data)

    else:
        experiment = ConditionalThreeBodyDecayVaeSimExperiment(vae_network, config['exp_params'], config['plotter'])
        tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                      name=config['model_params']['name'], )
        runner = Trainer(logger=tb_logger,
                         callbacks=[
                             LearningRateMonitor(),
                             ModelCheckpoint(save_top_k=3,
                                             dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                             monitor="val_loss",
                                             save_last=True),
                         ],
                         **config['trainer_params'])

        Path(f"{tb_logger.log_dir}/samples").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/gsamples").mkdir(exist_ok=True, parents=True)

        print(f"======= Training {config['model_params']['name']} =======")
        runner.fit(experiment, datamodule=data)

if __name__ == '__main__':
    argh.dispatch_command(main)