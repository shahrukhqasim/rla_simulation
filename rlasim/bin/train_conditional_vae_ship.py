import time
from pathlib import Path

import pytorch_lightning
import torch
import yaml
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

from rlasim.lib.experiments_conditional_ship import ConditionalShipVaeSimExperiment
from rlasim.lib.networks_conditional_ship import MlpConditionalVAE, Permuter, VaeShipLoss
import os
import argh

import tensorflow as tf
from rlasim.lib.organise_data import ThreeBodyDecayDataset
from rlasim.lib.ship_data import ShipDisDataset, RaggedToZeroPadded, convert_tf_to_torch
from rlasim.lib.utils import load_checkpoint


def main(config_file='configs/vae_conditional_ship_1.yaml', predict=False):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

    data = ShipDisDataset(**config["data_params"])
    data.setup()
    train_loader = data.train_dataloader()
    pad_processor = RaggedToZeroPadded(max_length=config["data_params"]['max_num_daughters'])
    permute = Permuter()
    vae_network = MlpConditionalVAE(**config["model_params"])


    # loss = VaeShipLoss(kld_weight=config['exp_params']['loss_params']['kld_weight'])

    # for x in train_loader:
    #     mother, daughters = x
    #     print("Hello")
    #     mother, daughters = pad_processor.forward(mother, daughters)
    #     mother, daughters = convert_tf_to_torch(mother), convert_tf_to_torch(daughters)
    #
    #
    #     all_dict = mother
    #     all_dict.update(daughters)
    #
    #     all_dict_permuted = permute.forward(all_dict)
    #
    #     all_result = vae_network.forward(all_dict_permuted)
    #     all_dict_permuted.update(all_result)
    #     loss(all_dict_permuted)
    #
    #     for k, v in all_result.items():
    #         print("CHECK", k, v.shape)
    #     0/0
    #
    # 0/0



    if predict:
        raise NotImplementedError('Not implemented')
        # experiment = ConditionalShipVaeSimExperiment(vae_network, config['generate_params'], config['plotter'])
        # load_checkpoint(experiment, config['checkpoint']['path'])
        # runner = Trainer()
        # runner.predict(experiment, datamodule=data)

    else:
        experiment = ConditionalShipVaeSimExperiment(vae_network, config['exp_params'], config['plotter'], config['data_params'])
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
    # Allow GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(physical_devices), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    argh.dispatch_command(main)
