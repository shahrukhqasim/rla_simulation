import os
import time
import itertools
from pathlib import Path

import argh
import uproot
import numpy as np
import yaml
from particle import Particle
from tqdm import tqdm

from rlasim.lib.data_core import get_pdgid
import pandas as pd

# from particle import Particle
# print(Particle.from_pdgid(431))
# print(Particle.from_pdgid(-431))
# quit()

def run(config_file=None, section=None, dont_clean=False, N_events=1E6):
    """
    Run RapidSim.

    :param config_file: a yaml file specifying the config
    :param section: which section to run in the config file. Set both config_file and section together.
    :param dont_clean: do not  intermediary files after
    :return:
    """

    if config_file is None:
        particles = ["K", "mu", "p", "e"]

        # Set the desired length of each combination
        combination_length = 3

        # Generate all combinations with repetitions allowed
        all_combinations = list(itertools.product(particles, repeat=combination_length))

        # Remove duplicates by converting each combination to a set
        unique_combinations = {tuple(sorted(comb)) for comb in all_combinations}
        unique_combinations = [(x[0]+'+', x[1]+'+', x[2]+'-') for x in unique_combinations]

        # Convert back to a list of lists
        unique_combinations = [list(comb) for comb in unique_combinations]
        mother_particles = ["B+"]
        num_sims = [10000 for _ in mother_particles]

        decay_channels = {}

        N_channels_total = 0
        for mother_particle in mother_particles:
            decay_channels[mother_particle] = []
            for unique_combination in unique_combinations:
                decay_channels[mother_particle].append(unique_combination)
            N_channels_total += len(decay_channels[mother])


    elif config_file == "produce_all":
        
        particles = pd.read_csv('particles.dat', skiprows=1, names=['ID', 'part', 'anti', 'mass', 'width', 'charge', 'spin', 'shape', 'ctau'], sep='\s+')

        mother_particles = [
                                "B0", "B+", "Bs0", "Bc+", 
                                "D0", "D+", "Ds+"
                            ]
        daughter_particles = [
                                "gamma",
                                "K+", "pi+", "pi0",
                                "e-", "mu-",
                                "nue",
                            ]

        for index, row in particles.iterrows():
            if row['part'] in daughter_particles:
                if row['anti'] != '---':
                    daughter_particles.append(row['anti'])

        # Set the desired length of each combination
        combination_length = 3

        # Generate all combinations with repetitions allowed
        all_combinations = list(itertools.product(daughter_particles, repeat=combination_length))

        # Remove duplicates by converting each combination to a set
        unique_combinations = {tuple(sorted(comb)) for comb in all_combinations}
        unique_combinations = [(x[0], x[1], x[2]) for x in unique_combinations]

        # Convert back to a list of lists
        unique_combinations = [list(comb) for comb in unique_combinations]

        N_channels_total = 0

        decay_channels = {}

        for mother in mother_particles:

            print(f'\nComputing decay channels for {mother}...')

            mother_info = particles[particles['part'] == mother]

            decay_channels[mother] = []

            for unique_combination in unique_combinations:
                
                daughters_lists = {}
                daughters_lists['masses'] =[]
                daughters_lists['charges'] =[]

                for particle_i in unique_combination:
                    daughter_info = particles[particles['part'] == particle_i]
                    if daughter_info.shape[0] == 0:
                        daughter_info = particles[particles['anti'] == particle_i]
                    daughters_lists['masses'].append(daughter_info.mass.item())
                    daughters_lists['charges'].append(daughter_info.charge.item())

                if mother_info.mass.item() > sum(daughters_lists['masses']) and sum(daughters_lists['charges']) == mother_info.charge.item():
                    decay_channels[mother].append(unique_combination)

            print(f'{len(decay_channels[mother])} decay channels listed.')
            N_channels_total += len(decay_channels[mother])

        print(f'\nN_channels_total: {N_channels_total}\n\n')

    else:
        assert section is not None
        try:
            with open(config_file, 'r') as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            exit()

        decay_channels = {}

        decays = config[section]['decays']
        unique_combinations = [x['daughters'] for x in decays]
        mother_particles = [x['mother'] for x in decays]
        num_sims = [x['num_sims'] for x in decays]

        N_channels_total = 0
        for mother_particle in mother_particles:
            decay_channels[mother_particle] = []
            for unique_combination in unique_combinations:
                decay_channels[mother_particle].append(unique_combination)
            N_channels_total += len(decay_channels[mother])

    try:
        os.system('rm -r rs_output')
    except:
        pass

    os.mkdir('rs_output')
    os.chdir('rs_output')

    print(f"Running {N_channels_total} channels")
    rapid_sim_path = '~/RapidSim/RapidSim/build/src/RapidSim.exe'
    if os.environ.get('RAPID_SIM_EXE_PATH') is not None:
        rapid_sim_path = os.environ.get('RAPID_SIM_EXE_PATH')

    rs_idx = -1

    for particle_m in list(decay_channels.keys()):
        for particle_combination in decay_channels[particle_m]:

            particle_i = particle_combination[0]
            particle_j = particle_combination[1]
            particle_k = particle_combination[2]

            rs_idx += 1

            f = open('../BLANK.config', 'r')
            f_lines = f.readlines()
            with open(f'rs_{rs_idx}.config', 'a') as f_out:
                for idx, line in enumerate(f_lines):
                    if 'BLANK0' in line:
                        line = line.replace('BLANK0', particle_m)
                    if 'BLANK1' in line:
                        line = line.replace('BLANK1', particle_i)
                    if 'BLANK2' in line:
                        line = line.replace('BLANK2', particle_j)
                    if 'BLANK3' in line:
                        line = line.replace('BLANK3', particle_k)
                    f_out.write(line)

            f = open('../BLANK.decay', 'r')
            f_lines = f.readlines()
            with open(f'rs_{rs_idx}.decay', 'a') as f_out:
                for idx, line in enumerate(f_lines):
                    if 'BLANK0' in line:
                        line = line.replace('BLANK0', particle_m)
                    if 'BLANK1' in line:
                        line = line.replace('BLANK1', particle_i)
                    if 'BLANK2' in line:
                        line = line.replace('BLANK2', particle_j)
                    if 'BLANK3' in line:
                        line = line.replace('BLANK3', particle_k)
                    f_out.write(line)

    time_A_full = time.time()
    rs_idx = -1
    for particle_m in list(decay_channels.keys()):
        for particle_combination in tqdm(decay_channels[particle_m]):

            N = int(N_events/N_channels_total)

            particle_i = particle_combination[0]
            particle_j = particle_combination[1]
            particle_k = particle_combination[2]

            rs_idx += 1
            print(
                f"{rs_idx + 1}/{N_channels_total}, Running RapidSim for {particle_m} -> {particle_i} {particle_j} {particle_k}")
            time_A = time.time()
            os.system(f'{rapid_sim_path} rs_{rs_idx} {N} 1 > dump.txt  ')
            file = uproot.open(f'rs_{rs_idx}_tree.root')["DecayTree"]
            keys = file.keys()
            results = file.arrays(keys, library="np")
            file.close()

            results['mother_PID'] = np.zeros(len(results['mother_E']), dtype=np.int32) + get_pdgid(particle_m)
            results['particle_1_PID'] = np.zeros(len(results['mother_E']), dtype=np.int32) + get_pdgid(particle_i)
            results['particle_2_PID'] = np.zeros(len(results['mother_E']), dtype=np.int32) + get_pdgid(particle_j)
            results['particle_3_PID'] = np.zeros(len(results['mother_E']), dtype=np.int32) + get_pdgid(particle_k)

            file2 = uproot.recreate(f'rs_{rs_idx}_tree2.root')
            file2['DecayTree'] = results
            file2.close()

            time_B = time.time()

    time_B_full = time.time()

    # os.system('rm dump.txt')
    os.system('hadd -fk output.root *_tree2.root')
    if not dont_clean:
        os.system('rm *_tree.root')
        os.system('rm *_hists.root')
        os.system('rm *config')
        os.system('rm *decay')
    os.system('mv output.root ../.')
    os.chdir(Path(os.getcwd()).parents[0])
    if not dont_clean:
        os.system('rm -r rs_output')

    print(f"\n\n\ntime: {time_B_full - time_A_full:.4f}")
    os.system('ls -lh output.root')


if __name__ == '__main__':
    argh.dispatch_command(run)