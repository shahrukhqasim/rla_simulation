import yaml
from tqdm import tqdm

from rlasim.lib.data_core import RootBlockShuffledSubsetDataLoader, tensors_dict_join
from rlasim.lib.experiments_conditional import ConditionalThreeBodyDecayVaeSimExperiment
from rlasim.lib.networks_conditional import MlpConditionalVAE, BaseVAE
import argh

from rlasim.lib.utils import load_checkpoint


def main(config_file='configs/vae_conditional_7.yaml'):
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        exit()

    vae_network = MlpConditionalVAE(**config["model_params"])
    experiment = ConditionalThreeBodyDecayVaeSimExperiment(vae_network, config['generate_params'], config['plotter'])

    # Either change this in the yaml file or the path here where you saved the checkpoint file
    load_checkpoint(experiment, config['checkpoint']['path'])

    # The following loader will give you batches of data loaded from a root file.
    # A subset of the full dataset will be sampled (with total samples equal to num_blocks*block_size).
    # Leave num_blocks=-1 for the full dataset.
    # The first parameter is the path of the root file you can use your own path as well.
    loader = RootBlockShuffledSubsetDataLoader(config['data_params']['data_path']['validate']['path'], block_size=1000, num_blocks=100, batch_size=1024)

    # Don't have to call it, but it's nice to see the progress
    loader.wait_to_load()

    all_results = []
    for i, batch in tqdm(enumerate(loader)):
        # Result dict should already have all the original elements of the batch as well
        result_dict = experiment.sample_and_reconstruct(batch)
        all_results += [result_dict]

    all_results = tensors_dict_join(all_results)

    # upp = unpreprocessed
    # pp = preprocessed
    # For you, the variable of interest should be momenta_sampled_upp
    print(all_results.keys())

    # TODO: Now do your analysis on all_results

    # This loader starts some threads in the background so you should also exit it in the end.
    # This might take O(10s) but you can kill the program as well.
    print("All done just waiting to exit")
    loader.exit()

if __name__ == '__main__':
    argh.dispatch_command(main)