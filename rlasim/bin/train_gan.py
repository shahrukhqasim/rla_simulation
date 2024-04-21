import time
from pathlib import Path

import pytorch_lightning
import torch
import yaml
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from rlasim.lib.experiments_gan import ThreeBodyDecayGanExperiment
from rlasim.lib.networks_gan import MlpConditionalWGAN
import os
import argh

from rlasim.lib.organise_data import ThreeBodyDecayDataset
from rlasim.lib.utils import load_checkpoint


def main(config_file='configs/gan_conditional_7.yaml', predict=False):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

    data = ThreeBodyDecayDataset(**config["data_params"])
    gan_network = MlpConditionalWGAN(**config["model_params"])

    if predict:
        raise NotImplementedError('Prediction functionality not implemented.')

    else:
        experiment = ThreeBodyDecayGanExperiment(gan_network, config['exp_params'], config['plotter'])
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

    data.exit()

if __name__ == '__main__':
    argh.dispatch_command(main)