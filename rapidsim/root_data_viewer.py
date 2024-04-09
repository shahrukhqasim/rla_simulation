import uproot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vector



file = uproot.open("/disk/users/adelri/first/rla_simulation/rapidsim/output.root")["DecayTree"]
keys = file.keys()
results = file.arrays(keys, library="np")
results = pd.DataFrame.from_dict(results)

pe_1 = np.sqrt(results.particle_1_M**2 + results.particle_1_PX**2 + results.particle_1_PY**2 + results.particle_1_PZ**2)
pe_2 = np.sqrt(results.particle_2_M**2 + results.particle_2_PX**2 + results.particle_2_PY**2 + results.particle_2_PZ**2)
pe_3 = np.sqrt(results.particle_3_M**2 + results.particle_3_PX**2 + results.particle_3_PY**2 + results.particle_3_PZ**2)




pe = pe_3 + pe_2
px = results.particle_3_PX + results.particle_2_PX
py = results.particle_3_PY + results.particle_2_PY
pz = results.particle_3_PZ + results.particle_2_PZ
p_32 = vector.obj(px=px, py=py, pz=pz, E=pe)

print(px)

"""
mass_32 = np.sqrt(p_32.E**2 - p_32.px**2 - p_32.py**2 - p_32.pz**2)

pe = pe_1 + pe_3
px = results.particle_1_PX + results.particle_3_PX
py = results.particle_1_PY + results.particle_3_PY
pz = results.particle_1_PZ + results.particle_3_PZ
p_13 = vector.obj(px=px, py=py, pz=pz, E=pe)

mass_13 = np.sqrt(p_13.E**2 - p_13.px**2 - p_13.py**2 - p_13.pz**2)

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("PHSP - default")
plt.hist2d(mass_32**2, mass_13**2, bins=50)
plt.xlabel("mass_{32}^2")
plt.ylabel("mass_{13}^2")

plt.savefig('test_plot')

"""



