import sys

import numpy as np
from torch.nn import ModuleList
from tqdm import tqdm

from rlasim.lib.data_core import RootBlockShuffledSubsetDataLoader
from rlasim.lib.networks_conditional_ship import VaeShipLoss, Permuter
from rlasim.lib.networks import BaseVAE

import torch
import pytorch_lightning as pl
from typing import List, Callable, Union, Any, TypeVar, Tuple
import os

from rlasim.lib.organise_data import ThreeBodyDecayDataset, OnlineThreeBodyDecayMomentaPreprocessor, OnlineThreeBodyDecayMomentaPreprocessor2, PreProcessor, \
    PostProcessor, MomentaCatPreprocessor
from rlasim.lib.plotting import ThreeBodyDecayPlotter, plot_latent_space
from rlasim.lib.plotting_ship import plot_summaries


#import csv
import pickle

from rlasim.lib.ship_data import RaggedToZeroPadded

Tensor = TypeVar('torch.tensor')


class ConditionalShipVaeSimExperiment(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, plotter_params: dict, data_params: dict, debug=False) -> None:
        super(ConditionalShipVaeSimExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.plotter_params = plotter_params
        self.curr_device = None
        # self.preprocessors = ModuleList([MomentaCatPreprocessor(), OnlineThreeBodyDecayMomentaPreprocessor()])
        self._debug=debug
        loss_params = self.params['loss_params']
        print(loss_params)
        self.loss_module = VaeShipLoss(**loss_params)
        self.permute = Permuter()

    def forward(self, input: dict, **kwargs) -> Tensor:
        x = self.model(input, **kwargs)
        return x

    def sample(self, num_samples, batch_condition):
        assert type(batch_condition) is dict
        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        batch_2 = dict()
        batch_2.update(batch_condition)

        sampled = self.model.sample(num_samples, self.curr_device, data_dict=batch_condition)
        batch_2.update(sampled)

        return batch_2

    def sample_and_reconstruct(self, batch):
        assert type(batch) is dict

        self.curr_device = list(batch.values())[0].device

        batch_2 = {}
        batch_2.update(batch)

        output = self.forward(batch_2)

        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        sampled = self.model.sample(len(list(batch.values())[0]), self.curr_device, data_dict=batch_2)
        batch_2.update(sampled)

        batch_2.update(batch)
        return batch_2

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        res_dict = self.sample_and_reconstruct(batch)
        self.prediction_results += [res_dict]


    def on_predict_start(self) -> None:
        self.prediction_results = []

    def on_predict_end(self) -> None:
        self.produce_pdf(self.prediction_results, str='results', path=self.params['pdf_prefix']+'_results_')



    def on_train_epoch_end(self) -> None:
        loader = self.trainer.datamodule.train_dataloader()
        if type(loader) is RootBlockShuffledSubsetDataLoader:
            prepare_next = True
            if 'reuse_prev_epoch_if_next_not_ready' in self.params:
                if self.params['reuse_prev_epoch_if_next_not_ready']:
                    print("Read progress", loader.get_read_progress())
                    if loader.get_read_progress() < 0.99:
                        prepare_next=False

            if prepare_next:
                loader.prepare_next_epoch()

    def training_step(self, batch_, batch_idx):
        assert type(batch_) is tuple
        mothers, daughters = batch_
        self.curr_device = list(mothers.values())[0].device
        batch = mothers
        batch.update(daughters)

        batch_2 = {}
        batch_2.update(batch)

        results = self.forward(batch_2)
        batch_2.update(results)
        train_loss = self.loss_module.forward(batch_2, batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']


    def on_validation_start(self) -> None:
        self.validation_results = []

    def validation_step(self, batch_, batch_idx, optimizer_idx=0):
        assert type(batch_) is tuple
        mothers, daughters = batch_
        self.curr_device = list(mothers.values())[0].device
        batch = mothers
        batch.update(daughters)

        batch_2 = {}
        batch_2.update(batch)

        results = self.forward(batch_2)
        batch_2.update(results)

        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        sampled = self.model.sample(len(list(batch_2.values())[0]), self.curr_device, data_dict=batch_2)
        batch_2.update(sampled)

        val_loss = self.loss_module.forward(batch_2)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        # print({k:v[0] for k,v in batch_2.items()} )

        self.validation_results += [batch_2]


    def on_validation_end(self) -> None:
        self.produce_pdf(self.validation_results, str='results')

    def produce_pdf(self, samples, str='samples', path = None):
        try:
            if path is None:
                path = os.path.join(self.logger.log_dir,
                                       'samples',
                                       f"{self.logger.name}_{str}_Epoch_{self.current_epoch}_")
            # try:
            plot_latent_space(samples, path=path+'latent_space_')
            plot_summaries(samples, path=path+'reco_', only_summary=False, t2='reconstructed')
            plot_summaries(samples, path=path+'sampled_', only_summary=False, t2='sampled')
            # reconstructed_results = plot_summaries(samples, path=path+'reco_', only_summary=False, t2='reconstructed')
            # sampled_results = plot_summaries(samples, path=path+'sampled_', only_summary=False, t2='sampled')

            #with open(f"Epoch_{self.current_epoch}_Data.csv", "w") as csvfile:
            #    writer = csv.DictWriter(csvfile, )

            """w = csv.writer(open(f"Epoch_{self.current_epoch}_results.csv", "w"))

            for key, val in sampled_results:
                w.writerow([key, val])
            """

            # with open(f'{path}reconstructed_data.pkl', 'wb') as handle:
            #     pickle.dump(reconstructed_results, handle)
            #
            # with open(f'{path}sampled_data.pkl', 'wb') as handle:
            #     pickle.dump(sampled_results, handle)

            # except Exception as e:
            #     print("Error occurred in plot summaries", e, file=sys.stderr)
            # plotter = ThreeBodyDecayPlotter(**self.plotter_params)
            # plotter.plot(samples, path)

        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.params['adam_lr'])
        return optimizer