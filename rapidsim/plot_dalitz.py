import numpy as np
import uproot
import pandas as pd
import matplotlib.pyplot as plt
import vector

#file = uproot.open("test_PHSP.root")["DecayTree"]
file = uproot.open("D+_2modes.root")["DecayTree"]
keys = file.keys()

results = file.arrays(keys, library="np")
results = pd.DataFrame.from_dict(results)

px_1 = results.particle_1_PX
px_2 = results.particle_2_PX
px_3 = results.particle_3_PX

py_1 = results.particle_1_PY
py_2 = results.particle_2_PY
py_3 = results.particle_3_PY

pz_1 = results.particle_1_PZ
pz_2 = results.particle_2_PZ
pz_3 = results.particle_3_PZ

pe_1 = np.sqrt(results.particle_1_M**2 + px_1**2 + py_1**2 + pz_1**2)
pe_2 = np.sqrt(results.particle_2_M**2 + px_2**2 + py_2**2 + pz_2**2)
pe_3 = np.sqrt(results.particle_3_M**2 + px_3**2 + py_3**2 + pz_3**2)

pe_32 = pe_3 + pe_2
px_32 = px_3 + px_2
py_32 = py_3 + py_2
pz_32 = pz_3 + pz_2

pe2_32 = (pe_32*pe_32)
px2_32 = (px_32*px_32)
py2_32 = (py_32*py_32)
pz2_32 = (pz_32*pz_32)

mass_32 = np.sqrt(pe2_32 - px2_32 - py2_32 - pz2_32)
#print(mass_32)


#p_32 = vector.obj(px=px, py=py, pz=pz, E=pe)

#mass_32 = np.sqrt(p_32.E**2 - p_32.px**2 - p_32.py**2 - p_32.pz**2)


pe_13 = pe_1 + pe_3
px_13 = px_1 + px_3
py_13 = py_1 + py_3
pz_13 = pz_1 + pz_3

pe2_13 = (pe_13*pe_13)
px2_13 = (px_13*px_13)
py2_13 = (py_13*py_13)
pz2_13 = (pz_13*pz_13)

mass_13 = np.sqrt(pe2_13 - px2_13 - py2_13 - pz2_13)
#print(mass_13)



"""p_13 = vector.obj(px=px, py=py, pz=pz, E=pe)

mass_13 = np.sqrt(p_13.E**2 - p_13.px**2 - p_13.py**2 - p_13.pz**2)
"""

def dalitz_plotter(x, y, xlab, ylab, xrange, yrange,title, rows, cols, fig, num_bins=50):
    plt.subplot(rows, cols, fig)
    plt.hist2d(x, y, bins=num_bins, range=[xrange, yrange])
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

def freq_plotter(x, xlab, xmin, xmax, title, rows, cols, fig, yscale = "linear", num_bins=50):
    plt.subplot(rows, cols, fig)
    counts, bin_edges = np.histogram(x, bins=num_bins, range=[xmin, xmax])
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #plt.bar(bin_centers, counts, width=np.diff(bin_edges))
    plt.scatter(bin_centers, counts)
    tot_events = counts.sum()
    plt.text(0.95, 0.95,
             f"Events: {tot_events}", horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes)

    events_in_range = ((x >= xmin) & (x <= xmax)).sum()
    events_excluded = tot_events - events_in_range
    plt.text(0.95, 0.85,
             f"Events out of range: {events_excluded}", horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes)


    #plt.hist(x, bins=num_bins, range=(-4,4))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel("Frequency")
    plt.yscale(yscale)

num_figs = 9
plt.figure(figsize=(12,12*num_figs))

dalitz_plotter(mass_32**2, mass_13**2, "mass_{32}^2", "mass_{13}^2", (0, 2), (0, 2), "", num_figs, 1, 1)
dalitz_plotter(px_1, py_1, "px_1", "py_1", (-2, 2), (-2, 2), "", num_figs, 1, 2)
dalitz_plotter(px_3, pz_3, "px_3", "pz_3", (-2, 2), (0, 4), "", num_figs, 1, 3)
freq_plotter(px_1, "px_1", -4, 4, "", num_figs, 1, 4)
freq_plotter(px_1, "px_1", -4, 4, "", num_figs, 1, 5, "log")
freq_plotter(py_1, "py_1", -4, 4, "", num_figs, 1, 6)
freq_plotter(py_1, "py_1", -4, 4, "", num_figs, 1, 7, "log")
freq_plotter(pz_1, "pz_1", 0, 35, "", num_figs, 1, 8)
freq_plotter(pz_1, "pz_1", 0, 35, "", num_figs, 1, 9, "log")


plt.savefig("plot")
plt.close("all")

"""
plt.figure(figsize=(6,12))
plt.subplot(5,1,1)
plt.title("1")
plt.hist2d(mass_32**2, mass_13**2, bins=50)
plt.xlabel("mass_{32}^2")
plt.ylabel("mass_{13}^2")

plt.subplot(5, 1, 2)
plt.title("2")
plt.hist2d(px_1, py_1, bins=50)
plt.xlabel("px_1")
plt.ylabel()
"""





"""
#file = uproot.open("test_D_DALITZ.root")["DecayTree"]
file = uproot.open("output.root")["DecayTree"]
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

mass_32 = np.sqrt(p_32.E**2 - p_32.px**2 - p_32.py**2 - p_32.pz**2)

pe = pe_1 + pe_3
px = results.particle_1_PX + results.particle_3_PX
py = results.particle_1_PY + results.particle_3_PY
pz = results.particle_1_PZ + results.particle_3_PZ
p_13 = vector.obj(px=px, py=py, pz=pz, E=pe)

mass_13 = np.sqrt(p_13.E**2 - p_13.px**2 - p_13.py**2 - p_13.pz**2)

plt.subplot(1,2,2)
plt.title("D_DALITZ")
plt.hist2d(mass_32**2, mass_13**2, bins=50)
plt.xlabel("mass_{32}^2")
plt.ylabel("mass_{13}^2")
plt.savefig('dalitz')
plt.close('all')
"""


