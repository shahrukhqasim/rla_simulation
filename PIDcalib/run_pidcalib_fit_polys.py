
import pidcalib_lib.chebyshev_functions as chebyshev_functions
import pidcalib_lib.run_pidcalib as run_pidcalib
import numpy as np
import json

# for particle in ["Pi", "K", "Mu", "e_B_Jpsi"]:
for particle in ["e_B_Jpsi"]:
    for variable in ["DLLK", "DLLe", "DLLmu"]:
        
        print(f"Running {particle} {variable}")

        PID_variable_values = np.linspace(-5,5,10)

        run_pidcalib.run(particle, variable, PID_variable_values)

        pidcalib_response, pidcalib_response_grid, pidcalib_conf_histogram = run_pidcalib.collect_PIDcalib_hists(particle, variable, PID_variable_values)

        c_ijk = chebyshev_functions.fit_chebyshev(particle, variable, pidcalib_response, pidcalib_response_grid, pidcalib_conf_histogram, PID_variable_values, 10, 10, 10)

        with open(f"{variable}_for_{particle}/c_ijk.json", 'r') as openfile:
            c_ijk = json.load(openfile)

        chebyshev_functions.plot_chebyshev(c_ijk, particle, variable, PID_variable_values)


