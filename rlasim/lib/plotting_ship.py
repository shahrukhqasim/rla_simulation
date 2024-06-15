import matplotlib.pyplot as plt
import torch

from rlasim.lib.data_core import tensors_dict_join
import numpy as np

def plot_summaries(all_results, path=None, only_summary=False, t2='sampled', skip_derive_vars=False):
    if type(all_results) is list:
        all_results = tensors_dict_join(all_results)
    data_samples_2 = {}
    for k, v in all_results.items():
        if isinstance(v, torch.Tensor):
            data_samples_2[k] = v.cpu().numpy()
        else:
            data_samples_2[k] = v
    all_results = data_samples_2

    # Create figure and axes for subplots
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.5))

    x1 = np.sum(all_results['dau_mask'], axis=1)
    x2 = all_results['dau_mask_%s'%t2][:, :, 0]
    x2 = (x2 >= 0.5).astype(np.int32)
    x2 = np.sum(x2, axis=1)


    # Plotting the first histogram
    axs[0].hist(x1, bins=20, density=True, histtype='step', color='tab:red')
    axs[0].set_title(f'True')
    axs[0].set_xlabel('Multiplicity of Daughters')
    axs[0].set_ylabel('Frequency (a.u.)')
    # axs[0].grid(True)

    # Plotting the second histogram
    axs[1].hist(x2, bins=20, density=True, histtype='step', color='tab:orange')
    axs[1].set_title(f'%s'%('Sampled' if 'sampled' in t2 else 'Reconstructed'))
    axs[1].set_xlabel('Multiplicity of Daughters')
    axs[1].set_ylabel('Frequency (a.u.)')
    # axs[1].grid(True)

    # Adjust layout and save the plot as PDF
    plt.tight_layout()
    plt.savefig(path+'multiplicities.pdf')