import gzip
import pickle
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

from rlasim.lib.organise_data import ThreeBodyDecayDataset, OnlineThreeBodyDecayMomentaPreprocessor, PreProcessor, \
    PostProcessor
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
        self.preprocessors = []

        if 'checkpoint_path' in params.keys():
            if not torch.cuda.is_available():
                self.load_state_dict(torch.load(params['checkpoint_path'], map_location=torch.device('cpu'))['state_dict'])
            else:
                self.load_state_dict(torch.load(params['checkpoint_path'])['state_dict'])

    def forward(self, input: dict, **kwargs) -> Tensor:
        x = self.model(input, **kwargs)
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        assert type(batch) is dict

        self.curr_device = list(batch.values())[0].device

        batch_2 = {}
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch))
        batch_2.update(batch)

        output = self.forward(batch_2)

        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        sampled = self.model.sample(len( list(batch.values())[0]), self.curr_device, condition=batch_2)

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PostProcessor):
                batch_2.update(preprocessor(sampled, direction=-1, on='sampled'))
                batch_2.update(preprocessor(output, direction=-1, on='reconstructed'))
        batch_2.update(batch)

        self.prediction_results += [batch_2]


    def on_predict_start(self) -> None:
        self.setup_preprocessors()
        self.prediction_results = []

    def on_predict_end(self) -> None:
        with gzip.open('dump.bin', 'wb') as f:
            pickle.dump(self.prediction_results, f)
            print("Dumped into dump.bin")

        self.produce_pdf(self.prediction_results, str='results', path=self.params['pdf_prefix']+'_results.pdf')

        # condition = torch.concatenate(self.predict_results_dict['condition'], dim=0)
        # decays_true = torch.concatenate(self.predict_results_dict['true'], dim=0)
        # decays_reco = torch.concatenate(self.predict_results_dict['reco'], dim=0)
        # decays_sampled = torch.concatenate(self.predict_results_dict['sampled'], dim=0)
        #
        # self.produce_pdf(decays_true, condition, str='true', path=self.params['pdf_prefix']+'_true.pdf')
        # self.produce_pdf(decays_reco, condition, str='reco', path=self.params['pdf_prefix']+'_reco.pdf')
        # self.produce_pdf(decays_sampled, condition, str='sampled', path=self.params['pdf_prefix']+'_sampled.pdf')
        #
        # preprocessor = self.trainer.datamodule.preprocessor
        # _, condition_pp = preprocessor(None, condition, direction=-1)
        #
        # filt = condition[:, 0, 2] <1000
        #
        # print(filt.shape, decays_true.shape, decays_reco.shape, condition.shape, decays_sampled.shape)
        # print(self.params['pdf_prefix']+'_true_pm_1000.pdf')
        # self.produce_pdf(decays_true[filt], condition[filt], str='true', path=self.params['pdf_prefix']+'_true_pm_1000.pdf')
        # self.produce_pdf(decays_reco[filt], condition[filt], str='reco', path=self.params['pdf_prefix']+'_reco_pm_1000.pdf')
        # self.produce_pdf(decays_sampled[filt], condition[filt], str='sampled', path=self.params['pdf_prefix']+'_sampled_pm_1000.pdf')


        # self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def setup_preprocessors(self):
        if len(self.preprocessors) > 0:
            return

        loader = self.trainer.datamodule.train_dataloader()
        all_data = []
        self.preprocessors = []
        for idx, batch in enumerate(loader):
            all_data.append(batch)
        self.preprocessors.append(OnlineThreeBodyDecayMomentaPreprocessor(all_data))

        print("X", len(self.preprocessors))


    def on_train_start(self) -> None:
        self.setup_preprocessors()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        assert type(batch) is dict
        self.curr_device = list(batch.values())[0].device

        batch_2 = {}
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch))
        batch_2.update(batch)

        results = self.forward(batch_2)
        train_loss = self.model.loss_function(results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        # print(batch_idx, np.mean(batch_pp.cpu().numpy()), np.mean(condition_pp.cpu().numpy()), train_loss['loss_kld'], train_loss['loss_reco'])

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']


    def on_validation_start(self) -> None:
        self.setup_preprocessors()
        self.validation_results = []

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        self.curr_device = list(batch.values())[0].device

        batch_2 = {}
        print(self.preprocessors)
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch))
        batch_2.update(batch)

        output = self.forward(batch_2)
        val_loss = self.model.loss_function(output,
                                            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        sampled = self.model.sample(len(list(batch.values())[0]), self.curr_device, condition=batch_2)

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PostProcessor):
                batch_2.update(preprocessor(sampled, direction=-1, on='sampled'))
                batch_2.update(preprocessor(output, direction=-1, on='reconstructed'))
        batch_2.update(batch)

        self.validation_results += [batch_2]


    def on_validation_end(self) -> None:
        self.produce_pdf(self.validation_results, str='results')

    def produce_pdf(self, samples, str='samples', path = None):
        try:
            if path is None:
                path = os.path.join(self.logger.log_dir,
                                       'samples',
                                       f"{self.logger.name}_{str}_Epoch_{self.current_epoch}.pdf")
            plotter = ThreeBodyDecayPlotter(**self.plotter_params)
            plotter.plot(samples, path)

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