from torch.nn import ModuleList
from tqdm import tqdm

from rlasim.lib.data_core import RootBlockShuffledSubsetDataLoader
from rlasim.lib.networks_conditional import MlpConditionalVAE, BaseVAE
import torch
import pytorch_lightning as pl
from typing import List, Callable, Union, Any, TypeVar, Tuple
import os

from rlasim.lib.organise_data import ThreeBodyDecayDataset, OnlineThreeBodyDecayMomentaPreprocessor, PreProcessor, \
    PostProcessor, MomentaCatPreprocessor
from rlasim.lib.plotting import ThreeBodyDecayPlotter

Tensor = TypeVar('torch.tensor')


class ConditionalThreeBodyDecayVaeSimExperiment(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict, plotter_params: dict, debug=False) -> None:
        super(ConditionalThreeBodyDecayVaeSimExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.plotter_params = plotter_params
        self.curr_device = None
        self.preprocessors = ModuleList([MomentaCatPreprocessor(), OnlineThreeBodyDecayMomentaPreprocessor()])
        self._debug=debug

    def forward(self, input: dict, **kwargs) -> Tensor:
        x = self.model(input, **kwargs)
        return x

    def sample(self, num_samples, batch_condition):
        assert type(batch_condition) is dict
        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        batch_2 = dict()
        batch_2.update(batch_condition)

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch_2))

        sampled = self.model.sample(num_samples, self.curr_device, data_dict=batch_condition)

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PostProcessor):
                batch_2.update(preprocessor(sampled, direction=-1, on='sampled'))

        return batch_2

    def sample_and_reconstruct(self, batch):
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
        res_dict = self.sample_and_reconstruct(batch)
        self.prediction_results += [res_dict]


    def on_predict_start(self) -> None:
        self.prediction_results = []

    def on_predict_end(self) -> None:
        self.produce_pdf(self.prediction_results, str='results', path=self.params['pdf_prefix']+'_results.pdf')

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
        # converted_dict = {}
        # for key, tensor in tensor_dict.items():
        #     if isinstance(tensor, np.ndarray):
        #         if tensor.dtype == np.float64:
        #             converted_dict[key] = tensor.astype(np.float32)
        #         elif tensor.dtype == np.int64:
        #             converted_dict[key] = tensor.astype(np.int32)
        #         else:
        #             converted_dict[key] = tensor
        #     elif isinstance(tensor, torch.Tensor):
        #         if tensor.dtype == torch.float64:
        #             converted_dict[key] = tensor.float()
        #         elif tensor.dtype == torch.int64:
        #             converted_dict[key] = tensor.int()
        #         else:
        #             converted_dict[key] = tensor
        #     else:
        #         raise ValueError("Unsupported tensor type")
        #
        # return converted_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        assert type(batch) is dict
        batch = self.to_32b(batch)
        self.curr_device = list(batch.values())[0].device

        batch_2 = {}
        batch_2.update(batch)
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch_2))
        # batch_2.update(batch)

        results = self.forward(batch_2)
        train_loss = self.model.loss_function(results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)
        # print(batch_idx, np.mean(batch_pp.cpu().numpy()), np.mean(condition_pp.cpu().numpy()), train_loss['loss_kld'], train_loss['loss_reco'])

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']


    def on_validation_start(self) -> None:
        self.validation_results = []

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        self.curr_device = list(batch.values())[0].device
        batch = self.to_32b(batch)

        batch_2 = {}
        batch_2.update(batch)
        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PreProcessor):
                batch_2.update(preprocessor(batch_2))

        output = self.forward(batch_2)
        val_loss = self.model.loss_function(output,
                                            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        # The code will find the momenta_mother_pp as the condition -- dicts are always easier to manage
        sampled = self.model.sample(len(list(batch.values())[0]), self.curr_device, data_dict=batch_2)

        for preprocessor in self.preprocessors:
            if isinstance(preprocessor, PostProcessor):
                batch_2.update(preprocessor(sampled, direction=-1, on='sampled'))
                batch_2.update(preprocessor(output, direction=-1, on='reconstructed'))
        batch_2.update(batch)

        # print({k:v[0] for k,v in batch_2.items()} )

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