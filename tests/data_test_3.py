import argh
import numpy as np
from tqdm import tqdm

from rlasim.lib.organise_data import ThreeBodyDecayDataset
import yaml
import torch
from rlasim.lib.plotting import ThreeBodyDecayPlotter


def main(config_file='configs/vae_conditional.yaml'):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

    plotter = ThreeBodyDecayPlotter(**config['plotter'])
    data = ThreeBodyDecayDataset(**config["data_params"])
    data.setup()
    loader = data.train_dataloader()
    all_data = []
    for idx, batch in tqdm(enumerate(loader)):
        all_data.append(batch)

    params = config['plotter']
    plotter.plot(all_data, params['check_file_prefix'])



if __name__ == '__main__':
    argh.dispatch_command(main)