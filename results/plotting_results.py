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
import re


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

def dalitz_plot(ax, data, data_range, xlab, ylab, bin_num, barlab, decay, type, norm, title=""):
    [x, y] = data
    decay = (f"{Particle.from_pdgid(decay[3]).name} → "
             f"{Particle.from_pdgid(decay[0]).name} "
             f"{Particle.from_pdgid(decay[1]).name} "
             f"{Particle.from_pdgid(decay[2]).name}")
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    counts, xedges, yedges, image = ax.hist2d(x, y, bins=bin_num, range=data_range, norm=norm)
    if title=="":
        ax.set_title(f"{type}: {decay}")
    else:
        ax.set_title(title)
    plt.colorbar(image, ax=ax, label=barlab)
    return counts

def dalitz_truth_sampled(ax, x, y, x_sampled, y_sampled, xlab, ylab, bin_num, unique_combination, coord_1="M", coord_2="M"):

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

def dalitz_truth_sampled_momenta(axs, results_i, particle, coord_1, coord_2, bin_num, unique_combination):

    axs = axs.flatten()

    w_val = dalitz_truth_sampled(axs[0:2],
                         results_i[f"particle_{particle}_{coord_1}"],
                         results_i[f"particle_{particle}_{coord_2}"],
                         results_i[f"particle_{particle}_{coord_1}_SAMPLED"],
                         results_i[f"particle_{particle}_{coord_2}_SAMPLED"],
                         f"particle {particle} {coord_1}",
                         f"particle {particle} {coord_2}",
                         bin_num,
                         unique_combination,
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
                         unique_combination,
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
    ax.errorbar(bin_centers_truth, counts_truth, yerr=errors_truth, fmt='none', color='blue', label='Truth', capsize=3)

    errors_sampled = poisson_asym_errors(counts_sampled)
    errors_sampled = np.where(errors_sampled < 0, 0, errors_sampled)
    ax.errorbar(bin_centers_sampled, counts_sampled, yerr=errors_sampled, fmt='none', color='orange', label='Sampled', capsize=3)

    """decay = (f"{Particle.from_pdgid(unique_combination[3]).name} → "
             f"{Particle.from_pdgid(unique_combination[0]).name} "
             f"{Particle.from_pdgid(unique_combination[1]).name} "
             f"{Particle.from_pdgid(unique_combination[2]).name}")

    ax.set_title(f"Decay Mode: {decay}")"""

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
training = "training6_rot"
epoch = "1599"
type = "sampled"
desktop = False
plots = False
mass_plots = False
dalitz_mass_plots = True
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


def training_vals(training_thesis, training_experiment, epochs_experiment):
    return [training_thesis, training_experiment, epochs_experiment]

one_D = [
    training_vals(1, "6", 8799),
    training_vals(2, "22", 4699),
    training_vals(3, "7", 2799),
    training_vals(4, "23", 4599)
]

one_D_rot = [
    training_vals(1, "6_rot", 1599),
    training_vals(2, "22_rot", 4599),
    training_vals(3, "7_rot", 7599),
    training_vals(4, "23_rot", 4199)
]

two_D_rot = [
    training_vals(5, "8_rot", 6099),
    training_vals(6, "9_rot", 2199),
    training_vals(7, "24_rot", 2099)
]

five_D_rot = [
    training_vals(8, "17_rot", 2399)
]

ten_D_rot = [
    training_vals(9, "18_rot", 999)
]

one_B_D_rot =[
    training_vals(10, "25_rot", 2499),
    training_vals(11, "10_rot", 2599)
]

two_B_D_rot = [
    training_vals(12, "11_rot", 2599)
]

five_B_D_rot = [
    training_vals(13, "20_rot", 299),
    training_vals(14, "21_rot", 799)
]

def mass_plotter(trainings, label, num_decays, type,log):
    folder = "mass_plots"
    dest_path = f"/Users/adelrio/Desktop/training_files/{folder}"

    if ("_rot" in trainings[0][1]):
        frame = "_rot"
    else:
        frame = ""

    if log == True:
        log = "_log"
    else:
        log = ""

    all_results = []
    training_nums = []
    for training_val in trainings:
        [training_thesis, experiment, epochs] = training_val
        # path = f"{path}{training}_results_Epoch_{epoch}_{type}_data.pkl"
        file = f"/Users/adelrio/Desktop/training_files/training{experiment}/training{experiment}_results_Epoch_{epochs}_{type}_data.pkl"
        all_results.append(open_results_file(file))
        training_nums.append(training_thesis)
    #print(training_nums)
    #print(len(all_results))

    pdf_path = f"{dest_path}/{label}_{num_decays}mode{frame}{log}_mass_plots.pdf"
    with PdfPages(pdf_path) as pdf:

        if label == "D":
            if num_decays == 1:
                subplot_x = 2
                subplot_y = 2
                size = (8, 8)
            elif num_decays == 2:
                subplot_x = 3
                subplot_y = 2
                size = (8, 12)
            elif num_decays == 5:
                subplot_x = 5
                subplot_y = 1
                size = (4, 20)
            elif num_decays == 10:
                subplot_x = 5
                subplot_y = 2
                size = (8, 20)
        elif label == "B_D":
            if num_decays == 1:
                subplot_x = 2
                subplot_y = 2
                size = (8, 8)
            elif num_decays == 2:
                subplot_x = 2
                subplot_y = 2
                size = (8, 8)
            elif num_decays == 5:
                subplot_x = 5
                subplot_y = 4
                size = (16, 20)


        fig, axs = plt.subplots(subplot_x, subplot_y, figsize=size, constrained_layout=True)
        axs = axs.flatten()
        count = 0

        for i, results in enumerate(all_results):

            unique_combinations = results[
                ['particle_1_PID', 'particle_2_PID', 'particle_3_PID', 'mother']].drop_duplicates()
            unique_combinations = unique_combinations.values.tolist()

            # for unique_combination in unique_combinations:
            #    print(unique_combination)
            #print(unique_combinations)
            for unique_combination in unique_combinations:

                results_i = results.query(
                    f'particle_1_PID=={unique_combination[0]} and particle_2_PID=={unique_combination[1]} and particle_3_PID=={unique_combination[2]}')

                if log == "_log":
                    w_val, ks_val, p_val = hist_truth_sampled(axs[count],
                                                              results_i,
                                                              "mother",
                                                              "M",
                                                              hist_bins,
                                                              unique_combination,
                                                              "log")
                else:
                    w_val, ks_val, p_val = hist_truth_sampled(axs[count],
                                                              results_i,
                                                              "mother",
                                                              "M",
                                                              hist_bins,
                                                              unique_combination)


                decay = (f"{Particle.from_pdgid(unique_combination[3]).name} → "
                         f"{Particle.from_pdgid(unique_combination[0]).name} "
                         f"{Particle.from_pdgid(unique_combination[1]).name} "
                         f"{Particle.from_pdgid(unique_combination[2]).name}")

                #print(i)
                #print(training_nums)
                axs[count].set_title(f"Training {training_nums[i]}: {decay}")
                count += 1



        pdf.savefig(fig)
        plt.close(fig)

def mass_plotter_dalitz(trainings, label, num_decays, type, dalitz_bins):
    folder = "dalitz_mass_plots"
    dest_path = f"/Users/adelrio/Desktop/training_files/{folder}"

    if ("_rot" in trainings[0][1]):
        frame = "_rot"
    else:
        frame = ""

    all_results = []
    training_nums = []
    for training_val in trainings:
        [training_thesis, experiment, epochs] = training_val
        # path = f"{path}{training}_results_Epoch_{epoch}_{type}_data.pkl"
        file = f"/Users/adelrio/Desktop/training_files/training{experiment}/training{experiment}_results_Epoch_{epochs}_{type}_data.pkl"
        all_results.append(open_results_file(file))
        training_nums.append(training_thesis)
    # print(training_nums)
    # print(len(all_results))

    pdf_path = f"{dest_path}/{label}_{num_decays}mode{frame}_dalitz_mass_plots.pdf"
    with PdfPages(pdf_path) as pdf:
        if label == "D":
            if num_decays == 1:
                subplot_x = 2
                subplot_y = 4
                size = (16, 8)
            elif num_decays == 2:
                subplot_x = 3
                subplot_y = 4
                size = (16, 12)
            elif num_decays == 5:
                subplot_x = 2
                subplot_y = 5
                size = (20, 8)
            elif num_decays == 10:
                subplot_x = 5
                subplot_y = 4
                size = (16, 20)
        elif label == "B_D":
            if num_decays == 1:
                subplot_x = 2
                subplot_y = 4
                size = (16, 8)
            elif num_decays == 2:
                subplot_x = 2
                subplot_y = 4
                size = (16, 8)
            elif num_decays == 5:
                subplot_x = 5
                subplot_y = 8
                size = (40, 20)



        fig, axs = plt.subplots(subplot_x, subplot_y, figsize=size, constrained_layout=True)
        axs = axs.flatten()
        count = 0

        for i, results in enumerate(all_results):

            unique_combinations = results[
                ['particle_1_PID', 'particle_2_PID', 'particle_3_PID', 'mother']].drop_duplicates()
            unique_combinations = unique_combinations.values.tolist()

            for unique_combination in unique_combinations:
                results_i = results.query(
                    f'particle_1_PID=={unique_combination[0]} and particle_2_PID=={unique_combination[1]} and particle_3_PID=={unique_combination[2]}')

                data_true = [results_i.mass_32 ** 2, results_i.mass_13 ** 2]
                data_sampled = [results_i.mass_32_SAMPLED ** 2, results_i.mass_13_SAMPLED ** 2]
                norm = log_scale_count(data_true, data_sampled, dalitz_bins)

                data_range = [[-np.amax(data_true[0]), np.amax(data_true[0])],
                              [-np.amax(data_true[1]), np.amax(data_true[1])]]
                coord_1 = "M"
                coord_2 = "M"
                if coord_1 == "PZ" or coord_1 == "M":
                    data_range[0] = [0, np.amax(data_true[0])]
                if coord_2 == "PZ" or coord_2 == "M":
                    data_range[1] = [0, np.amax(data_true[1])]
                # ax_sub = plt.subplot(1, 2, 1)

                decay = (f"{Particle.from_pdgid(unique_combination[3]).name} → "
                         f"{Particle.from_pdgid(unique_combination[0]).name} "
                         f"{Particle.from_pdgid(unique_combination[1]).name} "
                         f"{Particle.from_pdgid(unique_combination[2]).name}")


                truth_hist = dalitz_plot(axs[2*count],
                                         data_true,
                                         data_range,
                                         "mass$_{32}^2$",
                                         "mass$_{13}^2$",
                                         dalitz_bins,
                                         "Log-scaled count",
                                         unique_combination,
                                         "Truth",
                                         norm,
                                         f"Truth: Training {training_nums[i]} {decay}")

                # ax_sub = plt.subplot(1, 2, 2)
                sampled_hist = dalitz_plot(axs[2*count+1],
                                           data_sampled,
                                           data_range,
                                           "mass$_{32}^2$",
                                           "mass$_{13}^2$",
                                           dalitz_bins,
                                           "Log-scaled count",
                                           unique_combination,
                                           "Sampled",
                                           norm,
                                           f"Sampled: Training {training_nums[i]} {decay}")




                # print(i)
                # print(training_nums)
                #axs[2*count].set_title(f"Training {training_nums[2*i]}: {decay}")
                #axs[2*count + 1].set_title(f"Training {training_nums[2*i + 1]}: {decay}")
                count += 1
        pdf.savefig(fig)
        plt.close(fig)


#print(unique_combinations)
if mass_plots == True:
    mass_plotter(one_D, "D", 1, "sampled", False)
    mass_plotter(one_D, "D", 1, "sampled", True)

    mass_plotter(one_D_rot, "D", 1, "sampled", False)
    mass_plotter(one_D_rot, "D", 1, "sampled", True)

    mass_plotter(two_D_rot, "D", 2, "sampled", False)
    mass_plotter(two_D_rot, "D", 2, "sampled", True)

    mass_plotter(five_D_rot, "D", 5, "sampled", False)
    mass_plotter(five_D_rot, "D", 5, "sampled", True)

    mass_plotter(ten_D_rot, "D", 10, "sampled", False)
    mass_plotter(ten_D_rot, "D", 10, "sampled", True)

    mass_plotter(one_B_D_rot, "B_D", 1, "sampled", False)
    mass_plotter(one_B_D_rot, "B_D", 1, "sampled", True)

    mass_plotter(two_B_D_rot, "B_D", 2, "sampled", False)
    mass_plotter(two_B_D_rot, "B_D", 2, "sampled", True)

    mass_plotter(five_B_D_rot, "B_D", 5, "sampled", False)
    mass_plotter(five_B_D_rot, "B_D", 5, "sampled", True)


if dalitz_mass_plots == True:
    mass_plotter_dalitz(one_D, "D", 1, "sampled", dalitz_bins)
    mass_plotter_dalitz(one_D, "D", 1, "sampled", dalitz_bins)

    mass_plotter_dalitz(one_D_rot, "D", 1, "sampled", dalitz_bins)
    mass_plotter_dalitz(one_D_rot, "D", 1, "sampled", dalitz_bins)

    mass_plotter_dalitz(two_D_rot, "D", 2, "sampled", dalitz_bins)
    mass_plotter_dalitz(two_D_rot, "D", 2, "sampled", dalitz_bins)

    mass_plotter_dalitz(five_D_rot, "D", 5, "sampled", dalitz_bins)
    mass_plotter_dalitz(five_D_rot, "D", 5, "sampled", dalitz_bins)

    mass_plotter_dalitz(ten_D_rot, "D", 10, "sampled", dalitz_bins)
    mass_plotter_dalitz(ten_D_rot, "D", 10, "sampled", dalitz_bins)

    mass_plotter_dalitz(one_B_D_rot, "B_D", 1, "sampled", dalitz_bins)
    mass_plotter_dalitz(one_B_D_rot, "B_D", 1, "sampled", dalitz_bins)

    mass_plotter_dalitz(two_B_D_rot, "B_D", 2, "sampled", dalitz_bins)
    mass_plotter_dalitz(two_B_D_rot, "B_D", 2, "sampled", dalitz_bins)

    mass_plotter_dalitz(five_B_D_rot, "B_D", 5, "sampled", dalitz_bins)
    mass_plotter_dalitz(five_B_D_rot, "B_D", 5, "sampled", dalitz_bins)

if plots == True:

    file = f"{path}{training}_results_Epoch_{epoch}_{type}_data.pkl"

    results = open_results_file(file)
    # print(results.keys())

    unique_combinations = results[['particle_1_PID', 'particle_2_PID', 'particle_3_PID', 'mother']].drop_duplicates()
    unique_combinations = unique_combinations.values.tolist()
    for unique_combination in unique_combinations:
        print(unique_combination)

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
                                         dalitz_bins,
                                         unique_combination)
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
                    w_val, w_val_symlog = dalitz_truth_sampled_momenta(axs, results_i, particle, coord_1, coord_2, dalitz_bins, unique_combination)
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

        loss_min = 0.012
        loss_max = 0.017
        loss_reco_min = 0.002
        loss_reco_max = 0.005
        loss_kld_min = -11
        loss_kld_max = -9
        xlim = 0

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

        axs[0].plot(df1['Step'], df1['Value'])
        # axs[0].set_title('Plot 1')
        axs[0].set_xlabel('Step')
        axs[0].set_ylabel('Reconstruction Loss')
        axs[0].grid(True)
        axs[0].set_ylim([loss_reco_min, loss_reco_max])

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
        axs[2].set_ylim([loss_min, loss_max])

        if xlim != 0:
            axs[0].set_xlim([0, xlim])
            axs[1].set_xlim([0, xlim])
            axs[2].set_xlim([0, xlim])

        plt.tight_layout()

        pdf.savefig(fig)
        plt.close(fig)









