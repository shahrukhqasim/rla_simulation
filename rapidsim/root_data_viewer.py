import uproot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vector



file = uproot.open("/disk/users/adelri/first/rla_simulation/rapidsim/D+_2modes.root")["DecayTree"]
keys = file.keys()
results = file.arrays(keys, library="np")
results = pd.DataFrame.from_dict(results)

unique_combinations = results[['particle_1_PID', 'particle_2_PID', 'particle_3_PID']].drop_duplicates()

results_dic = {}

for i, row in unique_combinations.iterrows():
    condition = ((results.particle_1_PID==row.particle_1_PID)
                 & (results.particle_2_PID==row.particle_2_PID)
                 & (results.particle_3_PID==row.particle_3_PID))

    filtered_df = results[condition]
    key = f"{row['particle_1_PID']}_{row['particle_2_PID']}_{row['particle_3_PID']}"
    results_dic[key] = filtered_df

keys_dic = list(results_dic.keys())


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


def plotter(dfs, name):
    px_1 = dfs.particle_1_PX
    px_2 = dfs.particle_2_PX
    px_3 = dfs.particle_3_PX

    py_1 = dfs.particle_1_PY
    py_2 = dfs.particle_2_PY
    py_3 = dfs.particle_3_PY

    pz_1 = dfs.particle_1_PZ
    pz_2 = dfs.particle_2_PZ
    pz_3 = dfs.particle_3_PZ

    pe_1 = np.sqrt(dfs.particle_1_M ** 2 + px_1 ** 2 + py_1 ** 2 + pz_1 ** 2)
    pe_2 = np.sqrt(dfs.particle_2_M ** 2 + px_2 ** 2 + py_2 ** 2 + pz_2 ** 2)
    pe_3 = np.sqrt(dfs.particle_3_M ** 2 + px_3 ** 2 + py_3 ** 2 + pz_3 ** 2)

    pe_32 = pe_3 + pe_2
    px_32 = px_3 + px_2
    py_32 = py_3 + py_2
    pz_32 = pz_3 + pz_2

    pe2_32 = (pe_32 * pe_32)
    px2_32 = (px_32 * px_32)
    py2_32 = (py_32 * py_32)
    pz2_32 = (pz_32 * pz_32)

    mass_32 = np.sqrt(pe2_32 - px2_32 - py2_32 - pz2_32)

    pe_13 = pe_1 + pe_3
    px_13 = px_1 + px_3
    py_13 = py_1 + py_3
    pz_13 = pz_1 + pz_3

    pe2_13 = (pe_13 * pe_13)
    px2_13 = (px_13 * px_13)
    py2_13 = (py_13 * py_13)
    pz2_13 = (pz_13 * pz_13)

    mass_13 = np.sqrt(pe2_13 - px2_13 - py2_13 - pz2_13)


    num_figs = 9
    plt.figure(figsize=(12, 12 * num_figs))

    dalitz_plotter(mass_32 ** 2, mass_13 ** 2, "mass_{32}^2", "mass_{13}^2", (0, 2), (0, 2), "", num_figs, 1, 1)
    dalitz_plotter(px_1, py_1, "px_1", "py_1", (-2, 2), (-2, 2), "", num_figs, 1, 2)
    dalitz_plotter(px_3, pz_3, "px_3", "pz_3", (-2, 2), (0, 4), "", num_figs, 1, 3)
    freq_plotter(px_1, "px_1", -4, 4, "", num_figs, 1, 4)
    freq_plotter(px_1, "px_1", -4, 4, "", num_figs, 1, 5, "log")
    freq_plotter(py_1, "py_1", -4, 4, "", num_figs, 1, 6)
    freq_plotter(py_1, "py_1", -4, 4, "", num_figs, 1, 7, "log")
    freq_plotter(pz_1, "pz_1", 0, 35, "", num_figs, 1, 8)
    freq_plotter(pz_1, "pz_1", 0, 35, "", num_figs, 1, 9, "log")

    plt.savefig("plot_"+name)

    plt.close("all")

for key in keys_dic:
    plotter(results_dic[key], key)





"""
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

