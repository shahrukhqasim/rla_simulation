import sys

import numpy as np
from torch.nn import ModuleList
from tqdm import tqdm

from rlasim.lib.data_core import RootBlockShuffledSubsetDataLoader
from rlasim.lib.networks_conditional import MlpConditionalVAE, BaseVAE, ThreeBodyVaeLoss
import torch
import pytorch_lightning as pl
from typing import List, Callable, Union, Any, TypeVar, Tuple
import os

from rlasim.lib.networks_gan import ThreeBodyGanLoss
from rlasim.lib.organise_data import ThreeBodyDecayDataset, OnlineThreeBodyDecayMomentaPreprocessor, OnlineThreeBodyDecayMomentaPreprocessor2, PreProcessor, \
    PostProcessor, MomentaCatPreprocessor
from rlasim.lib.plotting import ThreeBodyDecayPlotter, plot_summaries,plot_latent_space

Tensor = TypeVar('torch.tensor')


class ThreeBodyDecayGanExperiment(pl.LightningModule):
    def __init__(self,
                 gan_network,
                 params: dict, plotter_params: dict, debug=False) -> None:
        super(ThreeBodyDecayGanExperiment, self).__init__()

        self.params = params
        self.plotter_params = plotter_params
        self.curr_device = None
        self.preprocessors = ModuleList([MomentaCatPreprocessor(), OnlineThreeBodyDecayMomentaPreprocessor()])
        self._debug=debug
        loss_params = self.params['loss_params']
        # print(loss_params)
        self.loss_module = ThreeBodyGanLoss(**loss_params)

        self.base_dist = torch.distributions.Normal(loc=0, scale=1)
        self.gan_network = gan_network



    def forward(self, input: dict, **kwargs) -> Tensor:
        return
        x = self.model(input, **kwargs)
        return x

    def sample(self, num_samples, batch_condition):
        return
        assert type(batch_condition) is dict
        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        batch_2 = dict()
        batch_2.update(batch_condition)

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch_2))

        sampled, sampled_dict = self.model.sample(num_samples, self.base_dist, data_dict=batch_condition)

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PostProcessor):
                batch_2.update(preprocessor(sampled_dict, direction=-1, on='sampled'))

        return batch_2

    def sample_and_reconstruct(self, batch):
        return
        assert type(batch) is dict
        batch = self.to_32b(batch)

        self.curr_device = list(batch.values())[0].device

        batch_2 = {}
        batch_2.update(batch)
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch_2))

        output = self.forward(batch_2)

        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        sampled = self.model.sample(len(list(batch.values())[0]), self.curr_device, data_dict=batch_2)

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PostProcessor):
                batch_2.update(preprocessor(sampled, direction=-1, on='sampled'))
                batch_2.update(preprocessor(output, direction=-1, on='reconstructed'))
        batch_2.update(batch)
        return batch_2

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return
        res_dict = self.sample_and_reconstruct(batch)
        self.prediction_results += [res_dict]


    def on_predict_start(self) -> None:
        self.prediction_results = []

    def on_predict_end(self) -> None:
        self.produce_pdf(self.prediction_results, str='results', path=self.params['pdf_prefix']+'_results_')

    def estimate_preprocessors(self, datamodule=None):
        # TODO: Can function be re-written better
        # if len(self.preprocessors) > 0:
        #     return
        if datamodule is None:
            loader = self.trainer.datamodule.train_dataloader()
        else:
            loader = datamodule.train_dataloader()

        all_data = []
        the_preprocessors = []
        momenta_cat_pp = self.preprocessors[0]
        for idx, batch in tqdm(enumerate(loader)):
            batch.update(momenta_cat_pp.forward(batch, direction=1))
            all_data.append(batch)

            # A million samples are enough to estimate max and min
            if idx * loader.batch_size >= 100000:
                break

        loader.reset()

        online_three_body_decay_pp = self.preprocessors[1]
        online_three_body_decay_pp.estimate(all_data)


    def on_train_start(self) -> None:
        self.estimate_preprocessors()

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

    def to_32b(self, tensor_dict):
        return tensor_dict

    def training_step(self, batch, batch_idx):
        assert type(batch) is dict
        batch = self.to_32b(batch)
        self.curr_device = list(batch.values())[0].device

        batch_2 = {}
        batch_2.update(batch)
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch_2))

        results = self.gan_network.forward(batch_2, self.base_dist, device=self.curr_device)

        print(results.keys())


        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PostProcessor):
                batch_2.update(preprocessor(results, direction=-1, on='sampled'))
        batch_2.update(results)

        train_loss = self.loss_module.forward(batch_2, batch_idx)
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']


    def on_validation_start(self) -> None:
        return
        self.validation_results = []

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        return
        self.curr_device = list(batch.values())[0].device
        batch = self.to_32b(batch)

        batch_2 = {}
        batch_2.update(batch)
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch_2))

        results = self.forward(batch_2)
        batch_2.update(results)

        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        sampled = self.model.sample(len(list(batch_2.values())[0]), self.curr_device, data_dict=batch_2)
        batch_2.update(sampled)
        print("\n\nCHECK", batch_2.keys())

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PostProcessor):
                batch_2.update(preprocessor(sampled, direction=-1, on='sampled'))
                batch_2.update(preprocessor(results, direction=-1, on='reconstructed'))

        val_loss = self.loss_module.forward(batch_2)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

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

            # except Exception as e:
            #     print("Error occurred in plot summaries", e, file=sys.stderr)
            # plotter = ThreeBodyDecayPlotter(**self.plotter_params)
            # plotter.plot(samples, path)

        except Warning:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gan_network.parameters(),
                                     lr=self.params['adam_lr'])
        return optimizer