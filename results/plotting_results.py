import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def open_results_file(file):
    with open(file, 'rb') as f:
        x = pickle.load(f)
    return x

file = '/disk/users/adelri/first/rla_simulation/logs/training5_rot_test/version_14/samples/training5_rot_test_results_Epoch_0_sampled_data.pkl'

a = open_results_file(file)



