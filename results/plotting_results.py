import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from particle import Particle
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import wasserstein_distance
from scipy.stats import wasserstein_distance_nd
from scipy.special import rel_entr
from scipy.stats import ks_2samp


def open_results_file(file):
    with open(file, 'rb') as f:
        x = pickle.load(f)
    return x

def open_loss_file(file):
   data = pd.read_csv(file)
   return data

def log_scale_count(truth, sampled, bins_num):
    zmin = 1
    zmax = max(
        np.histogram2d(truth[0], truth[1], bins=bins_num)[0].max(),
        np.histogram2d(sampled[0], sampled[1], bins=bins_num)[0].max()
    )

    norm = LogNorm(vmin=zmin, vmax=zmax)

    return norm

def pool(truth, sample, plot_type):
    with np.errstate(divide='ignore', invalid='ignore'):
        pool_vals = (sample - truth) / np.sqrt(sample)

    pool_vals[np.isnan(pool_vals)] = 0  # Replace NaNs with 0
    pool_vals[np.isinf(pool_vals)] = 0  # Replace infinities with 0

    if plot_type == "dalitz":
        pool = sum(item for sublist in pool_vals for item in sublist)
    elif plot_type == "hist":
        pool = sum(pool_vals)
    else:
        pool = "NaN"

    return pool

def wasserstein_dalitz(truth, sampled):
    return wasserstein_distance_nd(truth, sampled)

def hypothesis(truth, sampled):
    ks_statistic, p_value = ks_2samp(sampled, truth)
    return ks_statistic, p_value


def symlog(array):
    return np.sign(array) * np.log(np.abs(array) + 1)

def dalitz_plot(ax, data, data_range, xlab, ylab, bin_num, barlab, decay, type, norm):
    [x, y] = data
    decay = (f"{Particle.from_pdgid(decay[3]).name} → "
             f"{Particle.from_pdgid(decay[0]).name} "
             f"{Particle.from_pdgid(decay[1]).name} "
             f"{Particle.from_pdgid(decay[2]).name}")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    counts, xedges, yedges, image = ax.hist2d(x, y, bins=bin_num, range=data_range, norm=norm)
    ax.set_title(f"{type}: {decay}")
    plt.colorbar(image, ax=ax, label=barlab)
    return counts

def dalitz_truth_sampled(ax, x, y, x_sampled, y_sampled, xlab, ylab, bin_num, coord_1="M", coord_2="M"):

    data_true = [x, y]
    data_sampled = [x_sampled, y_sampled]

    norm = log_scale_count(data_true, data_sampled, bin_num)

    data_range = [[-np.amax(data_true[0]), np.amax(data_true[0])],
                   [-np.amax(data_true[1]), np.amax(data_true[1])]]

    if coord_1 == "PZ" or coord_1 == "M":
        data_range[0] = [0, np.amax(data_true[0])]
    if coord_2 == "PZ" or coord_2 == "M":
        data_range[1] = [0, np.amax(data_true[1])]

    #ax_sub = plt.subplot(1, 2, 1)
    truth_hist = dalitz_plot(ax[0],
                data_true,
                data_range,
                xlab,
                ylab,
                bin_num,
                "Log-scaled count",
                unique_combination,
                "Truth",
                norm)

    #ax_sub = plt.subplot(1, 2, 2)
    sampled_hist = dalitz_plot(ax[1],
                data_sampled,
                data_range,
                xlab,
                ylab,
                bin_num,
                "Log-scaled count",
                unique_combination,
                "Sampled",
                norm)

    #plt.show()

    #pool_val = pool(truth_hist, sampled_hist, "dalitz")
    #return w_dist(truth_hist, sampled_hist)
    return wasserstein_dalitz(truth_hist, sampled_hist)

def dalitz_truth_sampled_momenta(axs, results_i, particle, coord_1, coord_2, bin_num):

    axs = axs.flatten()

    w_val = dalitz_truth_sampled(axs[0:2],
                         results_i[f"particle_{particle}_{coord_1}"],
                         results_i[f"particle_{particle}_{coord_2}"],
                         results_i[f"particle_{particle}_{coord_1}_SAMPLED"],
                         results_i[f"particle_{particle}_{coord_2}_SAMPLED"],
                         f"particle {particle} {coord_1}",
                         f"particle {particle} {coord_2}",
                         bin_num,
                         coord_1,
                         coord_2)
    #print(f"Wasserstein {particle} {coord_1} & {particle} {coord_2}: {wasserstein_dalitz(truth, sampled)}")
    w_val_symlog = dalitz_truth_sampled(axs[2:4],
                         symlog(results_i[f"particle_{particle}_{coord_1}"]),
                         symlog(results_i[f"particle_{particle}_{coord_2}"]),
                         symlog(results_i[f"particle_{particle}_{coord_1}_SAMPLED"]),
                         symlog(results_i[f"particle_{particle}_{coord_2}_SAMPLED"]),
                         f"particle {particle} SYMLOG({coord_1})",
                         f"particle {particle} SYMLOG({coord_2})",
                         bin_num,
                         coord_1,
                         coord_2)
    #print(f"Wasserstein Symlog {particle} {coord_1} & {particle} {coord_2}: {wasserstein_dalitz(truth, sampled)}")
    #return pool_val, pool_val_symlog
    return w_val, w_val_symlog

def hist_truth_sampled(ax, results_i, particle, coordinate, bin_num, unique_combination, scale="linear", symlog_check=False):

    if symlog_check == True:
        data = symlog(results_i[f'particle_{particle}_{coordinate}'])
        data_sampled = symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED'])
    else:
        data = results_i[f'particle_{particle}_{coordinate}']
        data_sampled = results_i[f'particle_{particle}_{coordinate}_SAMPLED']

    if particle == "mother":
        particle_name = str(Particle.from_pdgid(unique_combination[3]).name)

        if str(Particle.from_pdgid(unique_combination[3]).name) == "D+":
            lower_bound = 1.7
            upper_bound = 1.9
        elif str(Particle.from_pdgid(unique_combination[3]).name) == "B+":
            lower_bound = 4.9
            upper_bound = 5.8


        counts_truth, bin_edges_truth, patches_truth = ax.hist(data,
                                                               bins=bin_num,
                                                               histtype='step',
                                                               range=(lower_bound, upper_bound),
                                                               color='blue')

        counts_sampled, bin_edges_sampled, patches_sampled = ax.hist(data_sampled,
                                                                     bins=bin_num,
                                                                     histtype='step',
                                                                     range=(lower_bound, upper_bound),
                                                                     color='orange')

        #ax.set_title(f"Mass distribution of {particle_name} in coordinate {coordinate}")
        # outside_count = np.sum((results_i[f'particle_{particle}_{coordinate}_SAMPLED'] < lower_bound) | (results_i[f'particle_{particle}_{coordinate}_SAMPLED'] > upper_bound))
        #ax.text(0.25, 0.925, f"Out of bound: {outside_count}", transform=ax.transAxes, horizontalalignment='center', verticalalignment='center')


    else:
        counts_truth, bin_edges_truth, patches_truth = ax.hist(data,
                                                               bins=bin_num,
                                                               histtype='step',
                                                               color='blue')

        counts_sampled, bin_edges_sampled, patches_sampled = ax.hist(data_sampled,
                                                                     bins=bin_num,
                                                                     histtype='step',
                                                                     color='orange')

        #ax.set_title(f"Mass distribution of {particle} in coordinate {coordinate}")

    bin_centers_truth = 0.5 * (bin_edges_truth[1:] + bin_edges_truth[:-1])
    bin_centers_sampled = 0.5 * (bin_edges_sampled[1:] + bin_edges_sampled[:-1])

    errors_truth = poisson_asym_errors(counts_truth)
    errors_truth = np.where(errors_truth < 0, 0, errors_truth)
    ax.errorbar(bin_centers_truth, counts_truth, yerr=errors_truth, fmt='o', color='blue', label='Truth', capsize=3)

    errors_sampled = poisson_asym_errors(counts_sampled)
    errors_sampled = np.where(errors_sampled < 0, 0, errors_sampled)
    ax.errorbar(bin_centers_sampled, counts_sampled, yerr=errors_sampled, fmt='o', color='orange', label='Sampled', capsize=3)

    """if particle == 1:
        particle_name = str(Particle.from_pdgid(unique_combination[0]).name)
    elif particle == 2:
        particle_name = str(Particle.from_pdgid(unique_combination[1]).name)
    elif particle == 3:
        particle_name = str(Particle.from_pdgid(unique_combination[2]).name)
    elif particle == "mother":
        particle_name = str(Particle.from_pdgid(unique_combination[3]).name)
    else:
        particle_name = "NO NAME" """

    if symlog_check == True:
        if particle == "mother":
            ax.set_xlabel(f'Particle {particle_name} SYMLOG({coordinate})')
        else:
            ax.set_xlabel(f'Particle {particle} SYMLOG({coordinate})')
    else:
        if particle == "mother":
            ax.set_xlabel(f'Particle {particle_name} {coordinate}')
        else:
            ax.set_xlabel(f'Particle {particle} {coordinate}')
    ax.set_ylabel('Frequency')
    ax.set_yscale(scale)
    #plt.show()

    #pool_val = pool(counts_truth, counts_sampled, "hist")
    ax.legend()

    wasserstein = wasserstein_distance(data, data_sampled)
    ks_stat, p_val = hypothesis(data, data_sampled)

    return wasserstein, ks_stat, p_val

def poisson_asym_errors(y_points):
	# https://www.nikhef.nl/~ivov/Talks/2013_03_21_DESY_PoissonError.pdf option 4

	compute_up_to_N = 150
	poisson_asym_errors_lookup_table = pickle.load(open(f'/Users/adelrio/Desktop/training_files/poisson_asym_errors_lookup_table.pickle',"rb"))

	y_errors_asym = np.zeros((2,np.shape(y_points)[0]))

	for y_point_idx, y_point in enumerate(y_points):
		if y_point > compute_up_to_N:
			y_err = np.sqrt(y_point)
			error_low = y_err
			error_high = y_err
		else:
			error_low = poisson_asym_errors_lookup_table[int(y_point)][0]
			error_high = poisson_asym_errors_lookup_table[int(y_point)][1]

		y_errors_asym[0][y_point_idx] = error_low
		y_errors_asym[1][y_point_idx] = error_high

	return y_errors_asym



version = ""
training = "training20_rot"
epoch = "299"
type = "sampled"
desktop = False
plots = False
loss = False
dalitz_bins = 12
hist_bins = 12

# MODIFY LABELS TO REFLECT GIVEN TYPE

if desktop == False:
    if version == "":
        path = f"/Users/adelrio/Desktop/training_files/{training}/"
    else:
        path = f"/Users/adelrio/Desktop/training_files/{training}/{version}/"
else:
    path = f"/Users/adelrio/training_files/"


file = f"{path}{training}_results_Epoch_{epoch}_{type}_data.pkl"

results = open_results_file(file)
#print(results.keys())


unique_combinations = results[['particle_1_PID', 'particle_2_PID', 'particle_3_PID', 'mother']].drop_duplicates()
unique_combinations = unique_combinations.values.tolist()
for unique_combination in unique_combinations:
    print(unique_combination)

#print(unique_combinations)

if plots == True:
    pdf_path = f"{path}plots_{training}_Epoch_{epoch}_{type}.pdf"

    with PdfPages(pdf_path) as pdf:

        num_combinations = len(unique_combinations)
        wasserstein_df_list = []
        hypothesis_df_list = []

        for count, unique_combination in enumerate(unique_combinations):
            # unique_combination_str = f'{unique_combination[0]}_{unique_combination[1]}_{unique_combination[2]}_'


            wasserstein_values = []
            wasserstein_labels = []
            hypothesis_labels = []
            ks_statistic_values = []
            p_values = []
            print(f"Creating plots for decay mode {count+1} out of {num_combinations}...")
            results_i = results.query(
                f'particle_1_PID=={unique_combination[0]} and particle_2_PID=={unique_combination[1]} and particle_3_PID=={unique_combination[2]}')
            fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            w_val = dalitz_truth_sampled(axs1,
                                         results_i.mass_32 ** 2,
                                         results_i.mass_13 ** 2,
                                         results_i.mass_32_SAMPLED ** 2,
                                         results_i.mass_13_SAMPLED ** 2,
                                         "mass$_{32}^2$",
                                         "mass$_{13}^2$",
                                         dalitz_bins)
            wasserstein_values.append(w_val)
            wasserstein_labels.append("mass$_{32}^2$ mass$_{13}^2$")
            # plt.show()
            # plt.close()
            pdf.savefig(fig1)
            plt.close(fig1)

            fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            axs2 = axs2.flatten()
            w_val, ks_val, p_val = hist_truth_sampled(axs2[0],
                                       results_i,
                                       "mother",
                                       "M",
                                       hist_bins,
                                       unique_combination)
            wasserstein_labels.append("mother")
            wasserstein_values.append(w_val)
            hypothesis_labels.append("mother")
            ks_statistic_values.append(ks_val)
            p_values.append(p_val)

            w_val, ks_val, p_val = hist_truth_sampled(axs2[1],
                                       results_i,
                                       "mother",
                                       "M",
                                       hist_bins,
                                       unique_combination,
                                       "log")
            wasserstein_labels.append("mother log")
            wasserstein_values.append(w_val)
            #hypothesis_labels.append("mother")
            #ks_statistic_values.append(ks_val)
            #p_values.append(p_val)
            # plt.show()
            # plt.close()
            pdf.savefig(fig2)
            plt.close(fig2)

            particles = [1, 2, 3, "p_32", "p_13"]
            coordinates_dalitz = [["PX", "PY"],
                                  ["PX", "PZ"],
                                  ["PY", "PZ"]]

            for particle in particles:
                for coordinate in coordinates_dalitz:
                    # [coord_1, coord_2] = coordinate
                    coord_1 = coordinate[0]
                    coord_2 = coordinate[1]
                    fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
                    w_val, w_val_symlog = dalitz_truth_sampled_momenta(axs, results_i, particle, coord_1, coord_2, dalitz_bins)
                    wasserstein_values.append(w_val)
                    wasserstein_labels.append(f"{particle} {coord_1} {coord_2}")
                    wasserstein_values.append(w_val_symlog)
                    wasserstein_labels.append(f"{particle} {coord_1} {coord_2} Symlog")
                    # print(pool_val, pool_val_symlog)
                    # plt.show()
                    # plt.close()
                    pdf.savefig(fig)
                    plt.close(fig)

            coordinates_hist = ["PX", "PY", "PZ"]
            for particle in particles:
                for coordinate in coordinates_hist:
                    fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
                    axs = axs.flatten()

                    w_val, ks_val, p_val = hist_truth_sampled(axs[0],
                                               results_i,
                                               particle,
                                               coordinate,
                                               hist_bins,
                                               unique_combination,
                                               "linear")
                    wasserstein_labels.append(f"{particle} {coordinate}")
                    wasserstein_values.append(w_val)

                    hypothesis_labels.append(f"{particle} {coordinate}")
                    ks_statistic_values.append(ks_val)
                    p_values.append(p_val)

                    w_val, ks_val, p_val = hist_truth_sampled(axs[1],
                                       results_i,
                                       particle,
                                       coordinate,
                                       hist_bins,
                                       unique_combination,
                                       "linear",
                                       True)
                    wasserstein_labels.append(f"{particle} {coordinate} SYMLOG")
                    wasserstein_values.append(w_val)

                    #hypothesis_labels.append(f"{particle} {coordinate} SYMLOG")
                    #ks_statistic_values.append(ks_val)
                    #p_values.append(p_val)

                    w_val, ks_val, p_val = hist_truth_sampled(axs[2],
                                               results_i,
                                               particle,
                                               coordinate,
                                               hist_bins,
                                               unique_combination,
                                               "log")

                    wasserstein_labels.append(f"{particle} {coordinate} log")
                    wasserstein_values.append(w_val)

                    #hypothesis_labels.append(f"{particle} {coordinate} log")
                    #ks_statistic_values.append(ks_val)
                    #p_values.append(p_val)

                    w_val, ks_val, p_val = hist_truth_sampled(axs[3],
                                               results_i,
                                               particle,
                                               coordinate,
                                               hist_bins,
                                               unique_combination,
                                               "log",
                                               True)
                    wasserstein_labels.append(f"{particle} {coordinate} SYMLOG log")
                    wasserstein_values.append(w_val)

                    #hypothesis_labels.append(f"{particle} {coordinate} SYMLOG log")
                    #ks_statistic_values.append(ks_val)
                    #p_values.append(p_val)


                    pdf.savefig(fig)
                    plt.close(fig)

            coordinate = "M"
            particles = ["mother", "p_32", "p_13"]

            for particle in particles:
                fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
                axs = axs.flatten()

                w_val, ks_val, p_val = hist_truth_sampled(axs[0],
                                           results_i,
                                           particle,
                                           coordinate,
                                           hist_bins,
                                           unique_combination,
                                           "linear")
                wasserstein_labels.append(f"{particle} {coordinate}")
                wasserstein_values.append(w_val)

                hypothesis_labels.append(f"{particle} {coordinate}")
                ks_statistic_values.append(ks_val)
                p_values.append(p_val)

                w_val, ks_val, p_val = hist_truth_sampled(axs[1],
                                           results_i,
                                           particle,
                                           coordinate,
                                           hist_bins,
                                           unique_combination,
                                           "linear",
                                           True)
                wasserstein_labels.append(f"{particle} {coordinate} SYMLOG")
                wasserstein_values.append(w_val)
                #hypothesis_labels.append(f"{particle} {coordinate} SYMLOG")
                #ks_statistic_values.append(ks_val)
                #p_values.append(p_val)

                w_val, ks_val, p_val = hist_truth_sampled(axs[2],
                                           results_i,
                                           particle,
                                           coordinate,
                                           hist_bins,
                                           unique_combination,
                                           "log")
                wasserstein_labels.append(f"{particle} {coordinate} log")
                wasserstein_values.append(w_val)
                #hypothesis_labels.append(f"{particle} {coordinate} log")
                #ks_statistic_values.append(ks_val)
                #p_values.append(p_val)

                w_val, ks_val, p_val = hist_truth_sampled(axs[3],
                                           results_i,
                                           particle,
                                           coordinate,
                                           hist_bins,
                                           unique_combination,
                                           "log",
                                           True)
                wasserstein_labels.append(f"{particle} {coordinate} log SYMLOG")
                wasserstein_values.append(w_val)
                #hypothesis_labels.append(f"{particle} {coordinate} log SYMLOG")
                #ks_statistic_values.append(ks_val)
                #p_values.append(p_val)

                pdf.savefig(fig)
                plt.close(fig)
            print(f"Finished plots for decay mode {count+1}")

            """
            wasserstein_dict = dict(zip(wasserstein_labels, wasserstein_values))
            ks_dict = dict(zip(hypothesis_labels, ks_statistic_values))
            pval_dict = dict(zip(hypothesis_labels, p_values))

            
            for label in wasserstein_labels:
                key = f"wasserstein_{label}_{decay_mode}"
                if key not in wasserstein_data:
                    wasserstein_data[key] = []
                wasserstein_data[key].append(wasserstein_dict[label])

            for label in hypothesis_labels:
                #key1 = f"ks_{label}_{decay_mode}"
                key2 = f"pval_{label}_{decay_mode}"
                #if key1 not in hypothesis_data:
                #    hypothesis_data[key1] = []
                if key2 not in hypothesis_data:
                    hypothesis_data[key2] = []
                #hypothesis_data[key1].append(ks_dict[label])
                hypothesis_data[key2].append(pval_dict[label])
            """
            decay_mode = f"{Particle.from_pdgid(unique_combination[3])} --> {Particle.from_pdgid(unique_combination[0]).name} {Particle.from_pdgid(unique_combination[1])} {Particle.from_pdgid(unique_combination[2]).name}"

            df = pd.DataFrame({
                "Label": wasserstein_labels,
                f"Wasserstein Distance {decay_mode}": wasserstein_values,
            })

            wasserstein_df_list.append(df)

            df_hypothesis = pd.DataFrame({
                "Label": hypothesis_labels,
                f"KS Statistic {decay_mode}": ks_statistic_values,
                f"P-Value {decay_mode}": p_values
            })

            hypothesis_df_list.append(df_hypothesis)

            """
            df.to_csv(
                f"{path}wasserstein_{Particle.from_pdgid(unique_combination[3]).name}_{Particle.from_pdgid(unique_combination[0]).name}_{Particle.from_pdgid(unique_combination[1]).name}_{Particle.from_pdgid(unique_combination[2]).name}.csv",
                index=False)
            
            df_hypothesis = pd.DataFrame({
                "Hypothesis Label": hypothesis_labels,
                "KS Statistic": ks_statistic_values,
                "P-Value": p_values
            }
            )"""

            #print(f"Finished wasserstein errors for decay mode {count+1}")

        concatenated_wasserstein = pd.concat([df.set_index('Label') for df in wasserstein_df_list], axis=1)
        concatenated_wasserstein.to_csv(f"{path}wasserstein_values.csv")

        concatenated_hypothesis = pd.concat([df.set_index('Label') for df in hypothesis_df_list], axis=1)
        concatenated_hypothesis.to_csv(f"{path}hypothesis_values.csv")


if loss == True:

    pdf_path = f"{path}loss_plots_{training}_Epoch_{epoch}.pdf"
    with PdfPages(pdf_path) as pdf:
        loss_file = f"{path}{training}_loss.csv"
        loss_kld_file = f"{path}{training}_loss_kld.csv"
        loss_reco_file = f"{path}{training}_loss_reco.csv"

        df3 = pd.read_csv(loss_file)
        df2 = pd.read_csv(loss_kld_file)
        df1 = pd.read_csv(loss_reco_file)

        loss_max = 0.05
        loss_reco_max = 0.01
        loss_kld_min = -15
        loss_kld_max = -10
        xlim = 0

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        axs[0].plot(df1['Step'], df1['Value'])
        # axs[0].set_title('Plot 1')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Reconstruction Loss')
        axs[0].grid(True)
        axs[0].set_ylim([0, loss_reco_max])

        axs[1].plot(df2['Step'], df2['Value'])
        # axs[1].set_title('Plot 2')
        axs[1].set_xlabel('Step')
        axs[1].set_ylabel('KLD Loss')
        axs[1].grid(True)
        axs[1].set_ylim([loss_kld_min, loss_kld_max])

        axs[2].plot(df3['Step'], df3['Value'])
        # axs[2].set_title('Plot 3')
        axs[2].set_xlabel('Step')
        axs[2].set_ylabel('Total Loss')
        axs[2].grid(True)
        axs[2].set_ylim([0, loss_max])

        if xlim != 0:
            axs[0].set_xlim([0, xlim])
            axs[1].set_xlim([0, xlim])
            axs[2].set_xlim([0, xlim])

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)









