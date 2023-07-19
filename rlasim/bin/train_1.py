import time
from pathlib import Path

import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from rlasim.lib.networks import VanillaVae, BaseVAE
import torch
import numpy as np
import pytorch_lightning as pl
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch.nn import functional as F
import os
import matplotlib.pyplot as plt
import argh

from rlasim.lib.organise_data import VaeDataset

Tensor = TypeVar('torch.tensor')


class VaeTrainingExperiment(pl.LightningModule):
    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VaeTrainingExperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None

        if 'checkpoint_path' in params.keys():
            if not torch.cuda.is_available():
                self.load_state_dict(torch.load(params['checkpoint_path'], map_location=torch.device('cpu'))['state_dict'])
            else:
                self.load_state_dict(torch.load(params['checkpoint_path'])['state_dict'])

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        x = self.model(input, **kwargs)
        return x


    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        self.curr_device = batch[0].device
        # batch = batch[0]
        #
        # if batch_idx < 10:
        #     self.curr_device = batch.device
        #     results = self.forward(batch)
        #     val_loss = self.model.loss_function(*results,
        #                                         M_N=self.params['kld_weight'])
        #
        #     print(val_loss)


    def on_predict_end(self) -> None:
        N = int(int(self.params['total_samples']) / int(self.params['batch_size']))

        self.sample_data(N=N, path = self.params['pdf_path'])



        # self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        batch = batch[0]
        self.curr_device = batch.device

        results = self.forward(batch)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        if batch_idx == 0:
            batch = batch[0]
            self.curr_device = batch.device
            results = self.forward(batch)
            val_loss = self.model.loss_function(*results,
                                                M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                                optimizer_idx=optimizer_idx,
                                                batch_idx=batch_idx)

            self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        if self.current_epoch % 10 == 0:
            self.sample_data(N=100)


    def produce_pdf(self, samples, log=False, str='samples', path = None, bins=50):
        try:
            samples = torch.reshape(samples, (-1, 3, 3))
            samples = samples.cpu().numpy()

            if path is None:
                path = os.path.join(self.logger.log_dir,
                                       str,
                                       f"{self.logger.name}_Epoch_{self.current_epoch}.pdf")
            indexing_dict = {0: 'x', 1: 'y', 2: 'z'}
            with PdfPages(path) as pdf:
                for particle in range(3):
                    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                    subplot_idx = 0
                    for i in range(3):
                        for j in range(i + 1, 3):
                            strx = 'particle_%d_p%s (minmax norm.)' % (particle, indexing_dict[i])
                            stry = 'particle_%d_p%s (minmax norm.)' % (particle, indexing_dict[j])

                            if log:
                                h = axes[subplot_idx].hist2d(samples[:, particle, i], samples[:, particle, j], bins=50,
                                                             norm=LogNorm(),
                                                             range=[[-1, 1], [-1, 1]])
                            else:
                                h = axes[subplot_idx].hist2d(samples[:, particle, i], samples[:, particle, j], bins=50,
                                                             range=[[-1, 1], [-1, 1]])

                            fig.colorbar(h[3], ax=axes[subplot_idx])
                            axes[subplot_idx].set_xlabel(strx)
                            axes[subplot_idx].set_ylabel(stry)
                            subplot_idx += 1

                    fig.tight_layout(pad=1.0)
                    pdf.savefig()
                    plt.close('all')

        except Warning:
            pass


    def sample_data(self, N=15, path=None):
        test_input = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input[0]
        test_input = test_input.to(self.curr_device)

        # samples = self.model.generate(test_input)
        # # samples = samples[:, 0:9]
        # # samples = test_input # Comment this out after checks
        # self.produce_pdf(samples, log=False, str='samples')

        batch_size = int(self.params['batch_size'])
        all_samples = []
        for i in tqdm(range(N)):
            samples = self.model.sample(batch_size, self.curr_device)
            all_samples += [samples]

        all_samples = torch.concatenate(all_samples, dim=0)

        self.produce_pdf(all_samples, log=True, str='gsamples', path=path)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.params['adam_lr'])
        return optimizer




def main(predict=False):

    vae_network = VanillaVae()
    vae_network(torch.Tensor(np.random.normal(0, 1, (100,9))))
    try:
        with open('configs/vae_1.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

    print(config)
    print(type(config))

    data = VaeDataset(**config["data_params"])

    if predict:
        experiment = VaeTrainingExperiment(vae_network, config['generate_params'])
        runner = Trainer()
        runner.predict(experiment, datamodule=data)

    else:
        experiment = VaeTrainingExperiment(vae_network, config['exp_params'])
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


    # for epoch in range(num_epochs):
    #     dataset.shuffle()
    #     dataset.permutate_particle_number()
    #     momenta_pp = dataset.get_data_i("momenta", mode="train", preprocessed=True).astype("float32")
    #     masses = dataset.get_data_i("masses", mode="train").astype("float32")
    #     training_loader = torch.utils.data.DataLoader(momenta_pp, batch_size=batch_size, shuffle=False)
    #     training_loader_masses = torch.utils.data.DataLoader(masses, batch_size=batch_size, shuffle=False)
    #
    #     print(f"EPOCH {epoch}")
    #     # For each batch in the dataloader
    #     for i, (momenta_minibatch, masses_minibatch) in enumerate(zip(training_loader, training_loader_masses), 0):
    #         momenta_minibatch = momenta_minibatch.reshape(-1, 9)
    #         masses_minibatch = masses_minibatch.reshape(-1, 3)
    #         training_minibatch = torch.cat((momenta_minibatch, masses_minibatch), dim=-1)
    #
    #         time.sleep(1)
    #         optimizer.zero_grad()
    #         rec, _, mu, log_var = vae_network.forward(training_minibatch)
    #         losses = vae_network.loss_function(rec, training_minibatch, mu, log_var, M_N=2)
    #
    #         losses['loss'].backward()
    #         optimizer.step()
    #         print("Epoch %05d, step %05d, loss %.3f, reco loss %.3f, kld loss %.3f"% (epoch, i, float(losses['loss']), float(losses['loss_reco']), float(losses['loss_kld'])))







if __name__ == '__main__':
    argh.dispatch_command(main)