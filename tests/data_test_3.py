import argh
import numpy as np
from tqdm import tqdm

from rlasim.lib.organise_data import ThreeBodyDecayDataset, MomentaCatPreprocessor
import yaml
import torch
from rlasim.lib.plotting import ThreeBodyDecayPlotter


def main(config_file='configs/vae_conditional_6.yaml'):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

    plotter = ThreeBodyDecayPlotter(**config['plotter'])
    data = ThreeBodyDecayDataset(**config["data_params"])

    data.setup()
    # dt = data.dataset_train

    loader = data.val_dataloader()
    all_data = []
    preprocessor = MomentaCatPreprocessor()
    for idx, batch in tqdm(enumerate(loader)):
        x = preprocessor(batch)
        batch.update(x)
        all_data.append(batch)

        if idx * loader.batch_size >= 100000000:
            break

    params = config['plotter']
    plotter.plot(all_data, params['check_file_prefix'])


if __name__ == '__main__':
    argh.dispatch_command(main)