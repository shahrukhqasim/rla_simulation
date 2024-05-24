import multiprocessing
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

from collections import Counter

# from particle import Particle
# print(Particle.from_pdgid(431))
# print(Particle.from_pdgid(-431))
# quit()

use_EVTGEN = True


particle_types = {}

particle_types["K+"] = "meson"
particle_types["K-"] = "meson"
particle_types["pi+"] = "meson"
particle_types["pi-"] = "meson"
particle_types["e-"] = "lepton"
particle_types["e+"] = "lepton"
particle_types["mu-"] = "lepton"
particle_types["mu+"] = "lepton"
particle_types["nue"] = "lepton_neutrino"
particle_types["anti-nue"] = "lepton_neutrino"
particle_types["gamma"] = "photon"
particle_types["K0b"] = "meson"



if os.getlogin() == 'am13743':
    rapid_sim_path = '$RAPIDSIM_ROOT/build/src/RapidSim.exe'
else:
    #rapid_sim_path = '~/RapidSim/RapidSim/build/src/RapidSim.exe'
    rapid_sim_path = 'RapidSim.exe'

if os.environ.get('RAPID_SIM_EXE_PATH') is not None:
    print(f"Settling rapid_sim_path to {os.environ.get('RAPID_SIM_EXE_PATH')}")
    rapid_sim_path = os.environ.get('RAPID_SIM_EXE_PATH')


def run_command(packed):
    rs_idx, particle_m, particle_combination, N = packed
    particle_i = particle_combination[0]
    particle_j = particle_combination[1]
    particle_k = particle_combination[2]

    time_A = time.time()
    try:
        os.system(f'{rapid_sim_path} rs_{rs_idx} {N} 1 > dump.txt  ')
    except:
        print("RAPID SIM FAILED")

    try:
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
    except ZeroDivisionError:
        print(f'rs_{rs_idx}_tree.root NOT PRESENT')

def run(config_file=None, section=None, dont_clean=False, N_events=1E6, output_name='output'):

    """
    Run RapidSim.

    :param config_file: a yaml file specifying the config
    :param section: which section to run in the config file. Set both config_file and section together.
    :param dont_clean: do not  intermediary files after
    :return:
    """

    if config_file is None:

        decay_channels = {}

        D_decay_channels = [
            {"decay": ["K0b", "e+", "nue"], "evtgen_model": "PHSP"},
            {"decay": ["K0b", "mu+", "numu"], "evtgen_model": "PHSP"},
            {"decay": ["pi-", "pi+", "pi+"], "evtgen_model": "PHSP"},
            {"decay": ["K-", "K+", "K+"], "evtgen_model": "PHSP"},
            {"decay": ["K+", "e+", "e-"], "evtgen_model": "PHSP"},
            {"decay": ["K+", "mu+", "mu-"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "e+", "e-"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "mu+", "mu-"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "e+", "mu-"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "e-", "mu+"], "evtgen_model": "PHSP"}
        ]

        B_decay_channels = [
            {"decay": ["K+", "pi-", "pi+"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "pi0", "pi0"], "evtgen_model": "PHSP"},
            {"decay": ["pi-", "pi+", "pi+"], "evtgen_model": "PHSP"},
            {"decay": ["K-", "K+", "K+"], "evtgen_model": "PHSP"},
            {"decay": ["K+", "e+", "e-"], "evtgen_model": "PHSP"},
            {"decay": ["K+", "mu+", "mu-"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "e+", "e-"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "mu+", "mu-"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "e+", "mu-"], "evtgen_model": "PHSP"},
            {"decay": ["pi+", "e-", "mu+"], "evtgen_model": "PHSP"}
        ]

        decay_channels["D+"] = [D_decay_channels[0]]
        decay_channels["B+"] = [B_decay_channels[0]]
        """decay_channels["D+"] = [D_decay_channels[5],
                                D_decay_channels[6],
                                D_decay_channels[7],
                                D_decay_channels[8],
                                D_decay_channels[9]
                                ]
        """
        """decay_channels["B+"] = [B_decay_channels[5],
                                B_decay_channels[6],
                                B_decay_channels[7],
                                B_decay_channels[8],
                                B_decay_channels[9]
                                ]
        """

        #decay_channels["B+"] = [B_decay_channels[7], B_decay_channels[8]]

        #decay_channels["D+"] = [{"decay": ["anti-K*(892)0", "e+", "nue"], "evtgen_model": "D_DALITZ"}]
        #decay_channels["D+"] = [{"decay":["K+", "pi+", "pi-"], "evtgen_model":"D_DALITZ"}]
        #decay_channels["B+"] = [{"decay":["K+", "e+", "e-"], "evtgen_model":"PHSP"}]

        """decay_channels["D+"] = [{"decay": ["K0b", "e+", "nue"], "evtgen_model": "PHSP"},
                                {"decay":["K0b", "mu+", "anti-numu"], "evtgen_model":"PHSP"},
                                {"decay": ["K+", "pi+", "pi-"], "evtgen_model": "PHSP"},
                                {"decay": ["K-", "K+", "K+"], "evtgen_model": "PHSP"},
                                {"decay": ["pi-", "K+", "K+"], "evtgen_model": "PHSP"}]
        """
        """decay_channels["D+"] = [{"decay": ["K0b", "e+", "nue"], "evtgen_model": "PHSP"},
                                {"decay": ["K0b", "mu+", "anti-numu"], "evtgen_model": "PHSP"}]
        """
        #decay_channels["D+"] = [{"decay": ["K0b", "e+", "nue"], "evtgen_model": "PHSP"}]

        N_channels_total = 1
        N_events = 1E5

    elif config_file == "produce_all":
        
        particles = pd.read_csv('particles.dat', skiprows=1, names=['ID', 'part', 'anti', 'mass', 'width', 'charge', 'spin', 'shape', 'ctau'], sep='\s+')

        mother_particles = [
                                "B0", "B+", "Bs0", "Bc+", 
                                "D0", "D+", "Ds+"
                            ]
        daughter_particles = [
                                "gamma",
                                "K+", "pi+",
                                "e-", "mu-",
                                "nue", "K0b"
                            ]

        # mother_particles = [
        #                         "D-",
        #                     ]
        # daughter_particles = [
        #                         "K+", "pi+",
        #                         "e-", "mu-",
        #                         "nue",
        #                     ]
        

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
            mother_charge_factor = 1
            if mother_info.shape[0] == 0:
                mother_info = particles[particles['anti'] == mother]
                mother_charge_factor = -1

            decay_channels[mother] = []

            for unique_combination in unique_combinations:
                
                daughters_lists = {}
                daughters_lists['masses'] =[]
                daughters_lists['charges'] =[]

                for particle_i in unique_combination:
                    daughter_info = particles[particles['part'] == particle_i]
                    charge_factor = 1
                    if daughter_info.shape[0] == 0:
                        daughter_info = particles[particles['anti'] == particle_i]
                        charge_factor = -1
                    daughters_lists['masses'].append(daughter_info.mass.item())
                    daughters_lists['charges'].append(daughter_info.charge.item()*charge_factor)
                
                if mother_info.mass.item() > sum(daughters_lists['masses']) and sum(daughters_lists['charges']) == mother_info.charge.item()*mother_charge_factor:

                    unique_combination_dict = {"decay":unique_combination, "evtgen_model":"PHSP"}

                    # D_DALITZ
                    if mother == "D+":
                        if Counter(unique_combination) == Counter(["K-", "pi+", "pi+"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["K-", "K+", "pi+"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["pi-", "K+", "pi+"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["pi-", "pi+", "pi+"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                    elif mother == "D-":
                        if Counter(unique_combination) == Counter(["K+", "pi-", "pi-"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["K+", "K-", "pi-"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["pi+", "K-", "pi-"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["pi+", "pi-", "pi-"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"        
                    elif mother == "Ds+":
                        if Counter(unique_combination) == Counter(["K-", "K+", "pi+"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["pi-", "pi+", "K+"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["pi-", "pi+", "pi+"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                    elif mother == "Ds-":
                        if Counter(unique_combination) == Counter(["K+", "K-", "pi-"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["pi+", "pi-", "K-"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                        elif Counter(unique_combination) == Counter(["pi+", "pi-", "pi-"]): unique_combination_dict["evtgen_model"] = "D_DALITZ"
                    
                    # ISGW2
                    particle_type = []
                    for particle in unique_combination:
                        particle_type.append(particle_types[particle])
                    if "meson" in particle_type and "lepton" in particle_type and "lepton_neutrino" in particle_type:
                        unique_combination_dict["evtgen_model"] = "ISGW2"

                    
                    decay_channels[mother].append(unique_combination_dict)
                    # print('\n')
                    # print(unique_combination, mother_info.charge.item(), sum(daughters_lists['charges']))
                    # for particle_i in unique_combination:
                    #     daughter_info = particles[particles['part'] == particle_i]
                    #     if daughter_info.shape[0] == 0:
                    #         daughter_info = particles[particles['anti'] == particle_i]
                    #     print(particle_i, "daughter_info:", daughter_info, "CHARGE", daughter_info.charge.item())
                    


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

    try: print(f"Running {N_channels_total} channels")
    except: pass

    rs_idx = -1

    for particle_m in list(decay_channels.keys()):
        for particle_combination in decay_channels[particle_m]:

            particle_i = particle_combination["decay"][0]
            particle_j = particle_combination["decay"][1]
            particle_k = particle_combination["decay"][2]

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
                    if 'useEvtGen' in line and not use_EVTGEN:
                        line = ''
                    if 'evtGenUsePHOTOS' in line and not use_EVTGEN:
                        line = ''
                    if 'evtGenModel' in line and not use_EVTGEN:
                        line = ''
                    elif 'evtGenModel' in line:
                        line = line.replace('pick_model', particle_combination["evtgen_model"])

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

        commands = []
        for particle_combination_idx, particle_combination in enumerate(tqdm(decay_channels[particle_m])):

            N = int(N_events/N_channels_total)

            particle_i = particle_combination["decay"][0]
            particle_j = particle_combination["decay"][1]
            particle_k = particle_combination["decay"][2]
            rs_idx += 1
            print(
                f"{rs_idx + 1}/{N_channels_total}, Running RapidSim for {particle_m} -> {particle_i} {particle_j} {particle_k} with model: {particle_combination['evtgen_model']}")

            # print(particle_combination_idx, particle_combination["decay"])
            run_command((rs_idx, particle_m, particle_combination["decay"], N))
            # commands += [(rs_idx, particle_m, particle_combination["decay"], N)]

        # num_processes = 20
        # pool = multiprocessing.Pool(processes=num_processes)

        # results = pool.map(run_command, commands)

        # pool.close()
        # pool.join()

        time_B = time.time()

    time_B_full = time.time()

    if ".root" in output_name:
        output_name = output_name[:-5]

    # os.system('rm dump.txt')
    os.system(f'hadd -fk {output_name}.root *_tree2.root')
    if not dont_clean:
        os.system('rm *_tree.root')
        os.system('rm *_hists.root')
        os.system('rm *config')
        os.system('rm *decay')
    os.system(f'mv {output_name}.root ../.')
    os.chdir(Path(os.getcwd()).parents[0])
    if not dont_clean:
        os.system('rm -r rs_output')

    print(f"\n\n\ntime: {time_B_full - time_A_full:.4f}")
    os.system(f'ls -lh {output_name}.root')


if __name__ == '__main__':
    argh.dispatch_command(run(output_name="training25"))