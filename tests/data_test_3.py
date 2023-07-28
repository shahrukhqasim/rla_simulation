import argh
import numpy as np
from tqdm import tqdm

from rlasim.lib.organise_data import ThreeBodyDecayDataset
import yaml
import torch
from rlasim.lib.plotting import ThreeBodyDecayPlotter


def main(config_file='configs/vae_conditional.yaml', predict=False):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

    data = ThreeBodyDecayDataset(**config["data_params"])
    data.setup()
    loader = data.train_dataloader()
    all_data = []
    all_data_mother = []
    for idx, batch in tqdm(enumerate(loader)):
        all_data.append(batch[0])
        all_data_mother.append(batch[1])

    samples = torch.cat(all_data, dim=0)
    samples_mother = torch.cat(all_data_mother, dim=0)

    print(len(samples), len(all_data[0]))

    plotter = ThreeBodyDecayPlotter(**config['plotter'])
    params = config['plotter']
    plotter.plot(samples, samples_mother, params['check_file'])



if __name__ == '__main__':
    argh.dispatch_command(main)