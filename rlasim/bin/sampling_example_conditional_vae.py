import yaml
from tqdm import tqdm

from rlasim.lib.data_core import RootBlockShuffledSubsetDataLoader, tensors_dict_join
from rlasim.lib.experiments_conditional import ConditionalThreeBodyDecayVaeSimExperiment
from rlasim.lib.networks_conditional import MlpConditionalVAE, BaseVAE
import argh

from rlasim.lib.utils import load_checkpoint

from pytorch_lightning import Trainer
from rlasim.lib.organise_data import ThreeBodyDecayDataset

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import vector
import os
from matplotlib.backends.backend_pdf import PdfPages
import pickle 

def poisson_asym_errors(y_points):
	# https://www.nikhef.nl/~ivov/Talks/2013_03_21_DESY_PoissonError.pdf option 4

	compute_up_to_N = 150
	poisson_asym_errors_lookup_table = pickle.load(open(f'number_storage/poisson_asym_errors_lookup_table.pickle',"rb"))

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




def plot_x_y_yerr(ax, data, limits=None, bins=50, label=None, c='tab:blue', markersize=2, alpha=1, also_plot_hist=True):

	hist_i = np.histogram(data, range=limits, bins=bins)
	binw=hist_i[1][1]-hist_i[1][0]
	x_points = hist_i[1][:-1] + (hist_i[1][1]-hist_i[1][0])/2.
	y_points = hist_i[0]
	yerr_points = np.sqrt(np.histogram(data, bins=bins, range=[np.amin(hist_i[1]),np.amax(hist_i[1])])[0])
	xerr_points = np.diff(hist_i[1])/2.
	yerr_points = poisson_asym_errors(y_points)
	yerr_points = np.where(yerr_points<0, 0, yerr_points)

	ax.errorbar(x_points, y_points, yerr=yerr_points, xerr=xerr_points, 
				color=c,marker='o',fmt=' ',capsize=2,linewidth=1.75, 
				markersize=markersize,label=label,alpha=alpha,zorder=100)

	if also_plot_hist:
		plt.hist(data, bins=bins, range=[np.amin(hist_i[1]),np.amax(hist_i[1])], histtype='step', color=c, alpha=alpha)

	return x_points, y_points, yerr_points, [np.amin(hist_i[1]),np.amax(hist_i[1])]





def main(config_file='configs/vae_conditional_7.yaml', only_summary=False, data_file = None):
	try:
		with open(config_file, 'r') as file:
			config = yaml.safe_load(file)
	except yaml.YAMLError as exc:
		print(exc)
		exit()
	
	if data_file != None:
		config['data_params']['data_path']['validate']['path'] = data_file

	vae_network = MlpConditionalVAE(**config["model_params"])
	experiment = ConditionalThreeBodyDecayVaeSimExperiment(vae_network, config['generate_params'], config['plotter'])

	# Either change this in the yaml file or the path here where you saved the checkpoint file
	load_checkpoint(experiment, config['checkpoint']['path'])

	# The following loader will give you batches of data loaded from a root file.
	# A subset of the full dataset will be sampled (with total samples equal to num_blocks*block_size).
	# Leave num_blocks=-1 for the full dataset.
	# The first parameter is the path of the root file you can use your own path as well.
	loader = RootBlockShuffledSubsetDataLoader(config['data_params']['data_path']['validate']['path'], block_size=1000, num_blocks=100, batch_size=1024)

	# Don't have to call it, but it's nice to see the progress
	loader.wait_to_load()

	all_results = []
	for i, batch in tqdm(enumerate(loader)):
		# Result dict should already have all the original elements of the batch as well
		result_dict = experiment.sample_and_reconstruct(batch)
		all_results += [result_dict]

	all_results = tensors_dict_join(all_results)

	# upp = unpreprocessed
	# pp = preprocessed
	# For you, the variable of interest should be momenta_sampled_upp
	print(all_results.keys())

	array_PID = np.asarray([all_results['particle_1_PID'].detach().numpy(), all_results['particle_2_PID'].detach().numpy(), all_results['particle_3_PID'].detach().numpy()])
	unique_combinations = np.swapaxes(np.unique(np.abs(array_PID), axis=1),0,1)
	for unique_combination in unique_combinations:
		print(unique_combination)

	results = {}
	results['particle_1_PID'] = all_results['particle_1_PID'].detach().numpy()
	results['particle_2_PID'] = all_results['particle_2_PID'].detach().numpy()
	results['particle_3_PID'] = all_results['particle_3_PID'].detach().numpy()
	for particle_idx, particle in enumerate(['1','2','3']):
		results[f'particle_{particle}_M'] = all_results[f'particle_{particle}_M'].detach().numpy()
		for coordinate_idx, coordinate in enumerate(['PX','PY','PZ']):
			results[f'particle_{particle}_{coordinate}'] = all_results['momenta'].detach().numpy()[:,particle_idx,coordinate_idx]
			results[f'particle_{particle}_{coordinate}_SAMPLED'] = all_results['momenta_sampled_upp'].detach().numpy()[:,particle_idx,coordinate_idx]

	results = pd.DataFrame.from_dict(results)

	results = results.query('particle_1_PZ>0')
	results = results.query('particle_2_PZ>0')
	results = results.query('particle_3_PZ>0')

	################################################################################
	# B
	################################################################################
	for tag in ['', "_SAMPLED"]:
		pe_1 = np.sqrt(results.particle_1_M**2 + results[f'particle_1_PX{tag}']**2 + results[f'particle_1_PY{tag}']**2 + results[f'particle_1_PZ{tag}']**2)
		pe_2 = np.sqrt(results.particle_2_M**2 + results[f'particle_2_PX{tag}']**2 + results[f'particle_2_PY{tag}']**2 + results[f'particle_2_PZ{tag}']**2)
		pe_3 = np.sqrt(results.particle_3_M**2 + results[f'particle_3_PX{tag}']**2 + results[f'particle_3_PY{tag}']**2 + results[f'particle_3_PZ{tag}']**2)

		pe = pe_1 + pe_2 + pe_3
		px = results[f'particle_1_PX{tag}'] + results[f'particle_2_PX{tag}'] + results[f'particle_3_PX{tag}']
		py = results[f'particle_1_PY{tag}'] + results[f'particle_2_PY{tag}'] + results[f'particle_3_PY{tag}']
		pz = results[f'particle_1_PZ{tag}'] + results[f'particle_2_PZ{tag}'] + results[f'particle_3_PZ{tag}']
		p_B = vector.obj(px=px, py=py, pz=pz, E=pe)

		B_mass = np.sqrt(p_B.E**2 - p_B.px**2 - p_B.py**2 - p_B.pz**2)

		results[f'particle_B_M{tag}'] = B_mass
		results[f'particle_B_PX{tag}'] = p_B.px
		results[f'particle_B_PY{tag}'] = p_B.py
		results[f'particle_B_PZ{tag}'] = p_B.pz

	################################################################################
	# DALITZ
	################################################################################
	for tag in ['', "_SAMPLED"]:
		pe_1 = np.sqrt(results.particle_1_M**2 + results[f'particle_1_PX{tag}']**2 + results[f'particle_1_PY{tag}']**2 + results[f'particle_1_PZ{tag}']**2)
		pe_2 = np.sqrt(results.particle_2_M**2 + results[f'particle_2_PX{tag}']**2 + results[f'particle_2_PY{tag}']**2 + results[f'particle_2_PZ{tag}']**2)
		pe_3 = np.sqrt(results.particle_3_M**2 + results[f'particle_3_PX{tag}']**2 + results[f'particle_3_PY{tag}']**2 + results[f'particle_3_PZ{tag}']**2)

		pe = pe_3 + pe_2
		px = results[f'particle_3_PX{tag}'] + results[f'particle_2_PX{tag}']
		py = results[f'particle_3_PY{tag}'] + results[f'particle_2_PY{tag}']
		pz = results[f'particle_3_PZ{tag}'] + results[f'particle_2_PZ{tag}']
		p_32 = vector.obj(px=px, py=py, pz=pz, E=pe)

		mass_32 = np.sqrt(p_32.E**2 - p_32.px**2 - p_32.py**2 - p_32.pz**2)
		results[f'mass_32{tag}'] = np.asarray(mass_32)

		pe = pe_1 + pe_3
		px = results[f'particle_1_PX{tag}'] + results[f'particle_3_PX{tag}']
		py = results[f'particle_1_PY{tag}'] + results[f'particle_3_PY{tag}']
		pz = results[f'particle_1_PZ{tag}'] + results[f'particle_3_PZ{tag}']
		p_13 = vector.obj(px=px, py=py, pz=pz, E=pe)

		mass_13 = np.sqrt(p_13.E**2 - p_13.px**2 - p_13.py**2 - p_13.pz**2)
		results[f'mass_13{tag}'] = np.asarray(mass_13)

		results[f'particle_p_32_M{tag}'] = np.asarray(mass_32)
		results[f'particle_p_13_M{tag}'] = np.asarray(mass_13)
		results[f'particle_p_32_PX{tag}'] = p_32.px
		results[f'particle_p_32_PY{tag}'] = p_32.py
		results[f'particle_p_32_PZ{tag}'] = p_32.pz
		results[f'particle_p_13_PX{tag}'] = p_13.px
		results[f'particle_p_13_PY{tag}'] = p_13.py
		results[f'particle_p_13_PZ{tag}'] = p_13.pz

	results = results.dropna()

	def symlog(array):
		return np.sign(array)*np.log(np.abs(array)+1)
	
	################################################################################
	# Plotting
	################################################################################

	for unique_combination in unique_combinations:

		print(unique_combination)

		unique_combination_str = f'{unique_combination[0]}_{unique_combination[1]}_{unique_combination[2]}_'

		results_i = results.query(f'abs(particle_1_PID)=={unique_combination[0]} and abs(particle_2_PID)=={unique_combination[1]} and abs(particle_3_PID)=={unique_combination[2]}')
		print(results_i.shape)

		coordinates = ['PX', 'PY', 'PZ']

		with PdfPages(f'{unique_combination_str}summary_plots.pdf') as pdf:

			plt.figure(figsize=(15,15))

			ax = plt.subplot(3,3,1)
			plt.hist2d(results_i.mass_32**2, results_i.mass_13**2, bins=50, 
					range=[[0, np.amax(results_i.mass_32**2)], [0, np.amax(results_i.mass_13**2)]],
					norm=LogNorm())
			plt.xlabel(r"mass$_{32}^2$")
			plt.ylabel(r"mass$_{13}^2$")
			plt.title("Truth")
			ax = plt.subplot(3,3,2)
			plt.hist2d(results_i.mass_32_SAMPLED**2, results_i.mass_13_SAMPLED**2, bins=50, 
					range=[[0, np.amax(results_i.mass_32**2)], [0, np.amax(results_i.mass_13**2)]],
					norm=LogNorm())
			plt.xlabel(r"mass$_{32}^2$")
			plt.ylabel(r"mass$_{13}^2$")
			plt.title("Sampled")

			particle = "B"
			coordinate = 'M'

			range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate}']))*1.2
			range_vec = [-range_i, range_i]
			range_vec_log = [-np.log(range_i), np.log(range_i)]
			if coordinate == "PZ" or coordinate == "M":
				range_vec = [0, range_i]
				range_vec_log = [0, np.log(range_i)]

			ax = plt.subplot(3,3,3)
			plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
			plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
			plt.xlabel(f'particle {particle} {coordinate}')
			plt.legend(loc='upper right')

			ax = plt.subplot(3,3,6)
			plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
			plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
			plt.xlabel(f'particle {particle} {coordinate}')
			plt.yscale('log')


			particle = "1"
			coordinate_i_idx = 0
			coordinate_j_idx = 1

			coordinate_i = coordinates[coordinate_i_idx]
			coordinate_j = coordinates[coordinate_j_idx]

			range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate_i}']))*1.2
			range_vec_i = [-range_i, range_i]
			range_vec_i_log = [-np.log(range_i), np.log(range_i)]
			if coordinate_i == "PZ":
				range_vec_i = [0, range_i]
				range_vec_i_log = [0, np.log(range_i)]

			range_j = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate_j}']))*1.2
			range_vec_j = [-range_j, range_j]
			range_vec_j_log = [-np.log(range_j), np.log(range_j)]
			if coordinate_j == "PZ":
				range_vec_j = [0, range_j]
				range_vec_j_log = [0, np.log(range_j)]

			ax = plt.subplot(3,3,4)
			hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}'], results_i[f'particle_{particle}_{coordinate_j}'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
			plt.xlabel(f'particle {particle} {coordinate_i}')
			plt.ylabel(f'particle {particle} {coordinate_j}')
			plt.title("Truth")

			ax = plt.subplot(3,3,5)
			hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED'], results_i[f'particle_{particle}_{coordinate_j}_SAMPLED'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
			plt.xlabel(f'particle {particle} {coordinate_i}')
			plt.ylabel(f'particle {particle} {coordinate_j}')
			plt.title("Sampled")


			particle = "3"
			coordinate_i_idx = 0
			coordinate_j_idx = 2

			coordinate_i = coordinates[coordinate_i_idx]
			coordinate_j = coordinates[coordinate_j_idx]

			range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate_i}']))*1.2
			range_vec_i = [-range_i, range_i]
			range_vec_i_log = [-np.log(range_i), np.log(range_i)]
			if coordinate_i == "PZ":
				range_vec_i = [0, range_i]
				range_vec_i_log = [0, np.log(range_i)]

			range_j = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate_j}']))*1.2
			range_vec_j = [-range_j, range_j]
			range_vec_j_log = [-np.log(range_j), np.log(range_j)]
			if coordinate_j == "PZ":
				range_vec_j = [0, range_j]
				range_vec_j_log = [0, np.log(range_j)]

			ax = plt.subplot(3,3,7)
			hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}'], results_i[f'particle_{particle}_{coordinate_j}'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
			plt.xlabel(f'particle {particle} {coordinate_i}')
			plt.ylabel(f'particle {particle} {coordinate_j}')
			plt.title("Truth")

			ax = plt.subplot(3,3,8)
			hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED'], results_i[f'particle_{particle}_{coordinate_j}_SAMPLED'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
			plt.xlabel(f'particle {particle} {coordinate_i}')
			plt.ylabel(f'particle {particle} {coordinate_j}')
			plt.title("Sampled")


			particle = "1"
			coordinate = 'PZ'

			range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate}']))*1.2
			range_vec = [-range_i, range_i]
			range_vec_log = [-np.log(range_i), np.log(range_i)]
			if coordinate == "PZ" or coordinate == "M":
				range_vec = [0, range_i]
				range_vec_log = [0, np.log(range_i)]

			ax = plt.subplot(3,3,9)
			plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
			plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
			plt.xlabel(f'particle {particle} {coordinate}')
			plt.yscale('log')


			plt.tight_layout()
			pdf.savefig(bbox_inches='tight')
			plt.close()


			if not only_summary:
				
				plt.figure(figsize=(4,4))
				pdf.savefig(bbox_inches='tight')
				plt.close()

				for particle in ['1', '2', '3']:
					for coordinate in ['PX', 'PY', 'PZ']:

						print(f'{particle} {coordinate}')
					
						range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate}']))*1.2
						range_vec = [-range_i, range_i]
						range_vec_log = [-np.log(range_i), np.log(range_i)]
						if coordinate == "PZ":
							range_vec = [0, range_i]
							range_vec_log = [0, np.log(range_i)]

						plt.figure(figsize=(10,8))
						ax = plt.subplot(2,2,1)
						plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
						x_points, y_points, yerr_points, hist_range = plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
						plt.xlabel(f'particle {particle} {coordinate}')
						plt.text(0.95, 0.95, f'Events: {int(np.shape(results_i[f"particle_{particle}_{coordinate}_SAMPLED"])[0])}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
						plt.text(0.95, 0.85, f'Out of range: {int(np.shape(results_i[f"particle_{particle}_{coordinate}_SAMPLED"])[0]-np.sum(y_points))}', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)

						ax = plt.subplot(2,2,2)
						plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}']), limits=range_vec_log, bins=50, label='Truth', c='tab:blue')
						plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED']), limits=range_vec_log, bins=50, label='Sampled', c='tab:orange')
						plt.xlabel(f'particle {particle} SYMLOG({coordinate})')
						plt.legend(loc='upper right')

						ax = plt.subplot(2,2,3)
						plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
						plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
						plt.yscale('log')
						plt.xlabel(f'particle {particle} {coordinate}')

						ax = plt.subplot(2,2,4)
						plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}']), limits=range_vec_log, bins=50, label='Truth', c='tab:blue')
						plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED']), limits=range_vec_log, bins=50, label='Sampled', c='tab:orange')
						plt.yscale('log')
						plt.xlabel(f'particle {particle} SYMLOG({coordinate})')

						pdf.savefig(bbox_inches='tight')
						plt.close()


				plt.figure(figsize=(4,4))
				pdf.savefig(bbox_inches='tight')
				plt.close()

				coordinates = ['PX', 'PY', 'PZ']

				for particle in ['1', '2', '3']:

					for coordinate_i_idx in range(3):
						for coordinate_j_idx in range(coordinate_i_idx+1, 3):

							coordinate_i = coordinates[coordinate_i_idx]
							coordinate_j = coordinates[coordinate_j_idx]
							print(particle, coordinate_i, coordinate_j)

							range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate_i}']))*1.2
							range_vec_i = [-range_i, range_i]
							range_vec_i_log = [-np.log(range_i), np.log(range_i)]
							if coordinate_i == "PZ":
								range_vec_i = [0, range_i]
								range_vec_i_log = [0, np.log(range_i)]

							range_j = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate_j}']))*1.2
							range_vec_j = [-range_j, range_j]
							range_vec_j_log = [-np.log(range_j), np.log(range_j)]
							if coordinate_j == "PZ":
								range_vec_j = [0, range_j]
								range_vec_j_log = [0, np.log(range_j)]

							plt.figure(figsize=(10,8))
							ax = plt.subplot(2,2,1)
							hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}'], results_i[f'particle_{particle}_{coordinate_j}'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
							plt.xlabel(f'particle {particle} {coordinate_i}')
							plt.ylabel(f'particle {particle} {coordinate_j}')
							plt.title("Truth")

							ax = plt.subplot(2,2,2)
							hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED'], results_i[f'particle_{particle}_{coordinate_j}_SAMPLED'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
							plt.xlabel(f'particle {particle} {coordinate_i}')
							plt.ylabel(f'particle {particle} {coordinate_j}')
							plt.title("Sampled")


							ax = plt.subplot(2,2,3)
							hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}']), symlog(results_i[f'particle_{particle}_{coordinate_j}']), range=[range_vec_i_log,range_vec_j_log], bins=35, norm=LogNorm())
							plt.xlabel(f'particle {particle} SYMLOG({coordinate_i})')
							plt.ylabel(f'particle {particle} SYMLOG({coordinate_j})')
							plt.title("Truth")


							ax = plt.subplot(2,2,4)
							hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED']), symlog(results_i[f'particle_{particle}_{coordinate_j}_SAMPLED']), range=[range_vec_i_log,range_vec_j_log], bins=35, norm=LogNorm())
							plt.xlabel(f'particle {particle} SYMLOG({coordinate_i})')
							plt.ylabel(f'particle {particle} SYMLOG({coordinate_j})')
							plt.title("Sampled")

							plt.tight_layout()
							pdf.savefig(bbox_inches='tight')
							plt.close()

				plt.figure(figsize=(4,4))
				pdf.savefig(bbox_inches='tight')
				plt.close()

				particle = "B"
				for coordinate in ['M', 'PX', 'PY', 'PZ']:

					print(f'B {coordinate}')
				
					range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate}']))*1.2
					range_vec = [-range_i, range_i]
					range_vec_log = [-np.log(range_i), np.log(range_i)]
					if coordinate == "PZ" or coordinate == "M":
						range_vec = [0, range_i]
						range_vec_log = [0, np.log(range_i)]

					plt.figure(figsize=(10,8))
					ax = plt.subplot(2,2,1)
					plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
					plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
					
					ax = plt.subplot(2,2,2)
					plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}']), limits=range_vec_log, bins=50, label='Truth', c='tab:blue')
					plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED']), limits=range_vec_log, bins=50, label='Sampled', c='tab:orange')
					plt.legend(loc='upper right')
					plt.xlabel(f'particle {particle} SYMLOG({coordinate})')

					ax = plt.subplot(2,2,3)
					plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
					plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
					plt.xlabel(f'particle {particle} {coordinate}')
					plt.yscale('log')

					ax = plt.subplot(2,2,4)
					plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}']), limits=range_vec_log, bins=50, label='Truth', c='tab:blue')
					plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED']), limits=range_vec_log, bins=50, label='Sampled', c='tab:orange')
					plt.xlabel(f'particle {particle} SYMLOG({coordinate})')
					plt.yscale('log')
					plt.tight_layout()
					pdf.savefig(bbox_inches='tight')
					plt.close()


				plt.figure(figsize=(4,4))
				pdf.savefig(bbox_inches='tight')
				plt.close()

				for particle in ['p_32', 'p_13']:
					for coordinate in ['M', 'PX', 'PY', 'PZ']:

						print(f'{particle} {coordinate}')
					
						range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate}']))*1.2
						range_vec = [-range_i, range_i]
						range_vec_log = [-np.log(range_i), np.log(range_i)]
						if coordinate == "PZ" or coordinate == "M":
							range_vec = [0, range_i]
							range_vec_log = [0, np.log(range_i)]

						plt.figure(figsize=(10,8))
						ax = plt.subplot(2,2,1)
						plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
						plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
						plt.xlabel(f'particle {particle} {coordinate}')
						ax = plt.subplot(2,2,2)
						plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}']), limits=range_vec_log, bins=50, label='Truth', c='tab:blue')
						plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED']), limits=range_vec_log, bins=50, label='Sampled', c='tab:orange')
						plt.legend(loc='upper right')
						plt.xlabel(f'particle {particle} SYMLOG({coordinate})')
						ax = plt.subplot(2,2,3)
						plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
						plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
						plt.xlabel(f'particle {particle} {coordinate}')
						plt.yscale('log')
						ax = plt.subplot(2,2,4)
						plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}']), limits=range_vec_log, bins=50, label='Truth', c='tab:blue')
						plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED']), limits=range_vec_log, bins=50, label='Sampled', c='tab:orange')
						plt.xlabel(f'particle {particle} SYMLOG({coordinate})')
						plt.yscale('log')
						plt.tight_layout()
						pdf.savefig(bbox_inches='tight')
						plt.close()


				plt.figure(figsize=(4,4))
				pdf.savefig(bbox_inches='tight')
				plt.close()

				for particle in ['p_32', 'p_13']:

					for coordinate_i_idx in range(3):
						for coordinate_j_idx in range(coordinate_i_idx+1, 3):

							coordinate_i = coordinates[coordinate_i_idx]
							coordinate_j = coordinates[coordinate_j_idx]
							print(particle, coordinate_i, coordinate_j)

							range_i = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate_i}']))*1.2
							range_vec_i = [-range_i, range_i]
							range_vec_i_log = [-np.log(range_i), np.log(range_i)]
							if coordinate_i == "PZ":
								range_vec_i = [0, range_i]
								range_vec_i_log = [0, np.log(range_i)]

							range_j = np.amax(np.abs(results_i[f'particle_{particle}_{coordinate_j}']))*1.2
							range_vec_j = [-range_j, range_j]
							range_vec_j_log = [-np.log(range_j), np.log(range_j)]
							if coordinate_j == "PZ":
								range_vec_j = [0, range_j]
								range_vec_j_log = [0, np.log(range_j)]

							plt.figure(figsize=(10,8))
							ax = plt.subplot(2,2,1)
							hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}'], results_i[f'particle_{particle}_{coordinate_j}'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
							plt.xlabel(f'{particle} {coordinate_i}')
							plt.ylabel(f'{particle} {coordinate_j}')
							plt.title("Truth")

							ax = plt.subplot(2,2,2)
							hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED'], results_i[f'particle_{particle}_{coordinate_j}_SAMPLED'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
							plt.xlabel(f'{particle} {coordinate_i}')
							plt.ylabel(f'{particle} {coordinate_j}')
							plt.title("Sampled")


							ax = plt.subplot(2,2,3)
							hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}']), symlog(results_i[f'particle_{particle}_{coordinate_j}']), range=[range_vec_i_log,range_vec_j_log], bins=35, norm=LogNorm())
							plt.xlabel(f'{particle} SYMLOG({coordinate_i})')
							plt.ylabel(f'{particle} SYMLOG({coordinate_j})')
							plt.title("Truth")


							ax = plt.subplot(2,2,4)
							hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED']), symlog(results_i[f'particle_{particle}_{coordinate_j}_SAMPLED']), range=[range_vec_i_log,range_vec_j_log], bins=35, norm=LogNorm())
							plt.xlabel(f'{particle} SYMLOG({coordinate_i})')
							plt.ylabel(f'{particle} SYMLOG({coordinate_j})')
							plt.title("Sampled")

							plt.tight_layout()
							pdf.savefig(bbox_inches='tight')
							plt.close()

				plt.figure(figsize=(4,4))
				pdf.savefig(bbox_inches='tight')
				plt.close()

				plt.figure(figsize=(10,5))

				ax = plt.subplot(1,2,1)
				plt.hist2d(results_i.mass_32**2, results_i.mass_13**2, bins=50, 
						range=[[0, np.amax(results_i.mass_32**2)], [0, np.amax(results_i.mass_13**2)]],
						norm=LogNorm())
				plt.xlabel(r"mass$_{32}^2$")
				plt.ylabel(r"mass$_{13}^2$")
				plt.title("Truth")
				ax = plt.subplot(1,2,2)
				plt.hist2d(results_i.mass_32_SAMPLED**2, results_i.mass_13_SAMPLED**2, bins=50, 
						range=[[0, np.amax(results_i.mass_32**2)], [0, np.amax(results_i.mass_13**2)]],
						norm=LogNorm())
				plt.xlabel(r"mass$_{32}^2$")
				plt.ylabel(r"mass$_{13}^2$")
				plt.title("Sampled")

				plt.tight_layout()
				pdf.savefig(bbox_inches='tight')
				plt.close()


	
		print(f'{unique_combination_str} done')

	print("All done, exiting...")
	os._exit(0)


	# This loader starts some threads in the background so you should also exit it in the end.
	# This might take O(10s) but you can kill the program as well.
	print("All done just waiting to exit")
	loader.exit()





if __name__ == '__main__':
	argh.dispatch_command(main)