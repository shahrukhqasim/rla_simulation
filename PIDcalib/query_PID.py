
import pidcalib_lib.chebyshev_functions as chebyshev_functions
import numpy as np
import json
import re

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx



def query_PID(PT, ETA, particle, selection, polynomials=False):

    selection_split = re.split('>|<', selection)

    PID_variable = selection_split[0]
    PID_variable_cut = selection_split[1]

    if polynomials:
        try:
            with open(f"{PID_variable}_for_{particle}/c_ijk.json", 'r') as openfile:
                c_ijk = json.load(openfile)
        except:
            with open(f"/afs/cern.ch/work/m/marshall/rla_simulation/PIDcalib/{PID_variable}_for_{particle}/c_ijk.json", 'r') as openfile:
                c_ijk = json.load(openfile)
            

        probability = chebyshev_functions.query_chebyshev(c_ijk, PT, ETA, np.ones(np.shape(PT))*float(PID_variable_cut))
    else:

        try:
            pidcalib_response = np.load(f"{PID_variable}_for_{particle}/pidcalib_response.npy")
            pidcalib_response_grid = np.load(f"{PID_variable}_for_{particle}/pidcalib_response_grid.npy")
            PID_variable_values = np.load(f"{PID_variable}_for_{particle}/PID_variable_values.npy")
        except:
            pidcalib_response = np.load(f"/afs/cern.ch/work/m/marshall/rla_simulation/PIDcalib/{PID_variable}_for_{particle}/pidcalib_response.npy")
            pidcalib_response_grid = np.load(f"/afs/cern.ch/work/m/marshall/rla_simulation/PIDcalib/{PID_variable}_for_{particle}/pidcalib_response_grid.npy")
            PID_variable_values = np.load(f"/afs/cern.ch/work/m/marshall/rla_simulation/PIDcalib/{PID_variable}_for_{particle}/PID_variable_values.npy")

        PID_map_idx = find_nearest(PID_variable_values, float(PID_variable_cut))
        pidcalib_response = pidcalib_response[:,PID_map_idx]

        PT_bin_centers = np.unique(pidcalib_response_grid[0])
        ETA_bin_centers = np.unique(pidcalib_response_grid[1])
        
        pidcalib_response = pidcalib_response.reshape((len(PT_bin_centers), len(ETA_bin_centers)))

        PT_bin_centers = PT_bin_centers-((PT_bin_centers[1]-PT_bin_centers[0])/2.)
        PT_bin_edges = np.append(PT_bin_centers, PT_bin_centers[-1]+(PT_bin_centers[1]-PT_bin_centers[0]))

        ETA_bin_centers = ETA_bin_centers-((ETA_bin_centers[1]-ETA_bin_centers[0])/2.)
        ETA_bin_edges = np.append(ETA_bin_centers, ETA_bin_centers[-1]+(ETA_bin_centers[1]-ETA_bin_centers[0]))

        idx_PT = np.digitize(PT, PT_bin_edges, right=True)
        idx_ETA = np.digitize(ETA, ETA_bin_edges, right=True)

        idx_PT[np.where(idx_PT==0)] += 1
        probability = pidcalib_response[idx_PT-1,idx_ETA-1]


    if "<" in selection:
        probability = 1 - probability
    
    return np.asarray(probability)

# PT = np.asarray([60000, 65000])
# ETA = np.asarray([3.0, 3.1])
# particle = "Pi"
# selection = "DLLe>-3"
# probability = query_PID(PT, ETA, particle, selection)

# print(probability)
