
import pidcalib_lib.chebyshev_functions as chebyshev_functions
import numpy as np
import json
import re

def query_PID(PT, ETA, particle, selection):

    selection_split = re.split('>|<', selection)

    PID_variable = selection_split[0]
    PID_variable_cut = selection_split[1]

    with open(f"{PID_variable}_for_{particle}/c_ijk.json", 'r') as openfile:
        c_ijk = json.load(openfile)

    probability = chebyshev_functions.query_chebyshev(c_ijk, PT, ETA, np.ones(np.shape(PT))*float(PID_variable_cut))

    if "<" in selection:
        probability = 1 - probability
    
    return probability

PT = np.asarray([60000, 65000])
ETA = np.asarray([3.0, 3.1])
particle = "Pi"
selection = "DLLe>-3"
probability = query_PID(PT, ETA, particle, selection)

print(probability)
