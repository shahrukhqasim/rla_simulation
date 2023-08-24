import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import vector
import sys
sys.path.append('../')
from query_PID import query_PID

selection_string = "DLLe>0"


file = uproot.open("Kee.root")["DecayTree"]
keys = file.keys()

results_Kee = file.arrays(keys, library="np")
results_Kee = pd.DataFrame.from_dict(results_Kee)

for particle in ["1", "2", "3"]:
    PT = np.sqrt(results_Kee[f"particle_{particle}_PX"]**2 + results_Kee[f"particle_{particle}_PY"]**2)
    theta = np.abs(np.arctan(PT/results_Kee[f"particle_{particle}_PZ"]))
    ETA = -np.log(np.tan(theta/2.))

    results_Kee[f"particle_{particle}_PT"] = PT*1000
    results_Kee[f"particle_{particle}_ETA"] = ETA

    results_Kee = results_Kee.query(f"particle_{particle}_ETA>1.5 and particle_{particle}_ETA<4.5")

probability_2 = query_PID(results_Kee["particle_2_PT"], results_Kee["particle_2_ETA"], "e_B_Jpsi", selection_string)
probability_3 = query_PID(results_Kee["particle_3_PT"], results_Kee["particle_3_ETA"], "e_B_Jpsi", selection_string)
results_Kee["PID_prob"] = probability_2*probability_3

pe_1 = np.sqrt(results_Kee.particle_1_M**2 + results_Kee.particle_1_PX**2 + results_Kee.particle_1_PY**2 + results_Kee.particle_1_PZ**2)
pe_2 = np.sqrt(results_Kee.particle_2_M**2 + results_Kee.particle_2_PX**2 + results_Kee.particle_2_PY**2 + results_Kee.particle_2_PZ**2)
pe_3 = np.sqrt(results_Kee.particle_3_M**2 + results_Kee.particle_3_PX**2 + results_Kee.particle_3_PY**2 + results_Kee.particle_3_PZ**2)
pe = pe_1 + pe_2 + pe_3
px = results_Kee.particle_1_PX + results_Kee.particle_2_PX + results_Kee.particle_3_PX
py = results_Kee.particle_1_PY + results_Kee.particle_2_PY + results_Kee.particle_3_PY
pz = results_Kee.particle_1_PZ + results_Kee.particle_2_PZ + results_Kee.particle_3_PZ
B_Kee = vector.obj(px=px, py=py, pz=pz, E=pe)

Bmass_Kee = np.sqrt(B_Kee.E**2 - B_Kee.px**2 - B_Kee.py**2 - B_Kee.pz**2)





file = uproot.open("KKK.root")["DecayTree"]
keys = file.keys()

results_KKK = file.arrays(keys, library="np")
results_KKK = pd.DataFrame.from_dict(results_KKK)

for particle in ["1", "2", "3"]:
    PT = np.sqrt(results_KKK[f"particle_{particle}_PX"]**2 + results_KKK[f"particle_{particle}_PY"]**2)
    theta = np.abs(np.arctan(PT/results_KKK[f"particle_{particle}_PZ"]))
    ETA = -np.log(np.tan(theta/2.))

    results_KKK[f"particle_{particle}_PT"] = PT
    results_KKK[f"particle_{particle}_ETA"] = ETA

    results_KKK = results_KKK.query(f"particle_{particle}_ETA>1.5 and particle_{particle}_ETA<4.5")

probability_2 = query_PID(results_KKK["particle_2_PT"], results_KKK["particle_2_ETA"], "K", selection_string)
probability_3 = query_PID(results_KKK["particle_3_PT"], results_KKK["particle_3_ETA"], "K", selection_string)
results_KKK["PID_prob"] = probability_2*probability_3


results_KKK.particle_2_M = np.ones(np.shape(results_KKK.particle_2_M))*0.000511
results_KKK.particle_3_M = np.ones(np.shape(results_KKK.particle_3_M))*0.000511


pe_1 = np.sqrt(results_KKK.particle_1_M**2 + results_KKK.particle_1_PX**2 + results_KKK.particle_1_PY**2 + results_KKK.particle_1_PZ**2)
pe_2 = np.sqrt(results_KKK.particle_2_M**2 + results_KKK.particle_2_PX**2 + results_KKK.particle_2_PY**2 + results_KKK.particle_2_PZ**2)
pe_3 = np.sqrt(results_KKK.particle_3_M**2 + results_KKK.particle_3_PX**2 + results_KKK.particle_3_PY**2 + results_KKK.particle_3_PZ**2)
pe = pe_1 + pe_2 + pe_3
px = results_KKK.particle_1_PX + results_KKK.particle_2_PX + results_KKK.particle_3_PX
py = results_KKK.particle_1_PY + results_KKK.particle_2_PY + results_KKK.particle_3_PY
pz = results_KKK.particle_1_PZ + results_KKK.particle_2_PZ + results_KKK.particle_3_PZ
B_KKK = vector.obj(px=px, py=py, pz=pz, E=pe)

Bmass_KKK = np.sqrt(B_KKK.E**2 - B_KKK.px**2 - B_KKK.py**2 - B_KKK.pz**2)



PT = np.sqrt(results_Kee.mother_PX**2 + results_Kee.mother_PY**2)
theta = np.abs(np.arctan(PT/results_Kee.mother_PZ))
ETA = -np.log(np.tan(theta/2.))

print(np.sum(results_KKK.PID_prob))
print(np.sum(results_Kee.PID_prob))
print(np.shape(results_KKK.PID_prob))
print(np.shape(results_Kee.PID_prob))

plt.hist([Bmass_KKK, Bmass_Kee], bins=50, histtype='step', range=[4.880, 5.600], color=['tab:blue', 'tab:orange'])
plt.savefig('test.png')
plt.close('all')

plt.hist([Bmass_KKK, Bmass_Kee], weights=[results_KKK.PID_prob,results_Kee.PID_prob], bins=50, histtype='step', range=[4.880, 5.600], color=['tab:blue', 'tab:orange'], linestyle='--')
plt.savefig('test_PID.png')
plt.close('all')

plt.hist([Bmass_KKK, Bmass_Kee], weights=[results_KKK.PID_prob,results_Kee.PID_prob], bins=50, histtype='step', range=[4.880, 5.600], color=['tab:blue', 'tab:orange'], linestyle='--')
plt.yscale('log')
plt.savefig('test_PID_log.png')
plt.close('all')


