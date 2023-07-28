import argh
import matplotlib.pyplot as plt
import numpy as np
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
    X = []
    Y = []
    for idx, batch in enumerate(loader):
        all_data.append(batch[0])
        all_data_mother.append(batch[1])

        x = np.mean(batch[0].cpu().numpy())
        y = np.mean(batch[1].cpu().numpy())
        X += [x]
        Y += [y]

    fig, [ax1, ax2]= plt.subplots(2, 1)
    ax1.hist(X)
    ax2.hist(Y)
    plt.show()




if __name__ == '__main__':
    argh.dispatch_command(main)