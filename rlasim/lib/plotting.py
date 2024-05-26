import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from matplotlib.colors import LogNorm

from rlasim.lib.data_core import tensors_dict_join
from rlasim.lib.plt_settings import set_sizing

import re

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import vector #0.8.0
import os
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from particle import Particle

small_fact = 1

class ThreeBodyDecayPlotter:
	def __init__(self, ranges=None, unit=None, conditions=None, **kwargs):
		# set_sizing()
		self.ranges = ranges

		self.range_min = {0:self.ranges['px_min'], 1:self.ranges['py_min'], 2:self.ranges['pz_min']}
		self.range_max = {0:self.ranges['px_max'], 1:self.ranges['py_max'], 2:self.ranges['pz_max']}
		self.unit = {0:'', 1:'', 2:''}
		if unit is not None:
			if type(unit) is str:
				if len(unit) > 0:
					self.unit = {0:' %s'%unit, 1:' %s'%unit, 2:' %s'%unit}
			else:
				NotImplementedError('Non str units not implemented')

		self.conditions = conditions
		if self.conditions is not None:
			assert type(self.conditions) is list

	def _make_histo_three_body_prods(self, samples, str='Decay products'):
		fig, axg = plt.subplots(3, 3, figsize=(9, 7))

		plt.subplots_adjust(left=0.12,
							bottom=0.13,
							right=0.95,
							top=0.9,
							wspace=0.3,
							hspace=0.1)
		fig.suptitle(str, fontsize=16)
		fig.tight_layout(pad=2.0)

		axl_dict = {0: 'x', 1: 'y', 2: 'z'}
		for p in range(3):
			for i in range(3):
				ax = axg[p][i]
				dat = samples[:, p, i]
				dat[dat < self.range_min[i]] = self.range_min[i]
				dat[dat > self.range_max[i]] = self.range_max[i]
				hist, bins = np.histogram(dat, bins=50, range=(self.range_min[i], self.range_max[i]))
				bin_centers = (bins[:-1] + bins[1:]) / 2
				ax.step(bin_centers, hist, where='mid', color='tab:red', linewidth=1)
				ax.set_yscale('log')
				ax.set_xlabel('${p_%d}_%s$%s' % (p + 1, axl_dict[i],self.unit[i]))
				ax.set_ylabel('Frequency')

		return fig, axg


	def _make_three_body_prods_hist_diff(self, samples, samples_2, str='Decay products', ylabel='true', ylabel_2='sampled'):
		fig, axg = plt.subplots(3, 3, figsize=(9, 7))

		plt.subplots_adjust(left=0.12,
							bottom=0.13,
							right=0.95,
							top=0.9,
							wspace=0.3,
							hspace=0.15)
		fig.suptitle(str, fontsize=16)
		fig.tight_layout(pad=2.5)

		axl_dict = {0: 'x', 1: 'y', 2: 'z'}
		for p in range(3):
			for i in range(3):
				ax = axg[p][i]

				dat = samples[:, p, i]
				dat[dat < self.range_min[i]] = self.range_min[i]
				dat[dat > self.range_max[i]] = self.range_max[i]

				dat_2 = samples_2[:, p, i]
				dat_2[dat_2 < self.range_min[i]] = self.range_min[i]
				dat_2[dat_2 > self.range_max[i]] = self.range_max[i]

				hist, bins = np.histogram(dat, bins=50, range=(self.range_min[i], self.range_max[i]))
				hist2, _ = np.histogram(dat_2, bins=50, range=(self.range_min[i], self.range_max[i]))

				bin_centers = (bins[:-1] + bins[1:]) / 2

				ax.step(bin_centers, np.log10(hist2/hist), where='mid', color='tab:red', linewidth=1)

				# ax.set_yscale('log')
				ax.set_xlabel('${p_%d}_%s$%s' % (p + 1, axl_dict[i],self.unit[i]))
				ax.set_ylabel('log10$(N_{\\mathrm{%s}}/N_{\\mathrm{%s}})$'%(ylabel_2, ylabel))

		return fig, axg

	def _make_2d_histo_three_body_prods(self, samples, str='Decay products'):
		fig, axg = plt.subplots(3, 3, figsize=(9, 7))

		plt.subplots_adjust(left=0.12,
							bottom=0.13,
							right=0.95,
							top=0.9,
							wspace=0.3,
							hspace=0.1)

		fig.suptitle(str, fontsize=16)

		combos = [(0, 1), (0, 2), (1, 2)]

		axl_dict = {0: 'x', 1: 'y', 2: 'z'}
		for p in range(3):
			for i in range(3):
				ax = axg[p][i]

				samples_x = samples[:, p, combos[i][0]]*1
				samples_y = samples[:, p, combos[i][1]]*1

				samples_x[samples_x < self.range_min[combos[i][0]]] = self.range_min[combos[i][0]]
				samples_x[samples_x > self.range_max[combos[i][0]]] = self.range_max[combos[i][0]]

				samples_y[samples_y < self.range_min[combos[i][1]]] = self.range_min[combos[i][1]]
				samples_y[samples_y > self.range_max[combos[i][1]]] = self.range_max[combos[i][1]]

				the_range = [
								  [self.range_min[combos[i][0]]*small_fact, self.range_max[combos[i][0]]*(1/small_fact)],
								  [self.range_min[combos[i][1]]*small_fact, self.range_max[combos[i][1]]*(1/small_fact)]
							  ]

				h = ax.hist2d(samples_x,
							  samples_y,
							  range=the_range,
							  norm=LogNorm(),
							  bins=30)
				if not len(samples_x) == 0:
					fig.colorbar(h[3], ax=ax)

				ax.set_xlabel('${p_%d}_%s$%s' % (p + 1, axl_dict[combos[i][0]], self.unit[combos[i][0]]))
				ax.set_ylabel('${p_%d}_%s$%s' % (p + 1, axl_dict[combos[i][1]], self.unit[combos[i][1]] ))
				# ax.set_ylabel('Frequency')
		fig.tight_layout(pad=2.0)
		return fig, axg

	def _make_2d_three_body_prods_hist_diff(self, samples, samples_2, str='Sampled vs true distr. diff.'):
		fig, axg = plt.subplots(3, 3, figsize=(9, 7))

		plt.subplots_adjust(left=0.12,
							bottom=0.13,
							right=0.95,
							top=0.9,
							wspace=0.3,
							hspace=0.1)

		fig.suptitle(str, fontsize=16)

		combos = [(0, 1), (0, 2), (1, 2)]

		axl_dict = {0: 'x', 1: 'y', 2: 'z'}
		for p in range(3):
			for i in range(3):
				ax = axg[p][i]

				samples_x = samples[:, p, combos[i][0]]*1
				samples_y = samples[:, p, combos[i][1]]*1

				samples_2_x = samples_2[:, p, combos[i][0]]*1
				samples_2_y = samples_2[:, p, combos[i][1]]*1

				samples_x[samples_x < self.range_min[combos[i][0]]] = self.range_min[combos[i][0]]
				samples_x[samples_x > self.range_max[combos[i][0]]] = self.range_max[combos[i][0]]

				samples_y[samples_y < self.range_min[combos[i][1]]] = self.range_min[combos[i][1]]
				samples_y[samples_y > self.range_max[combos[i][1]]] = self.range_max[combos[i][1]]

				samples_2_x[samples_2_x < self.range_min[combos[i][0]]] = self.range_min[combos[i][0]]
				samples_2_x[samples_2_x > self.range_max[combos[i][0]]] = self.range_max[combos[i][0]]

				samples_2_y[samples_2_y < self.range_min[combos[i][1]]] = self.range_min[combos[i][1]]
				samples_2_y[samples_2_y > self.range_max[combos[i][1]]] = self.range_max[combos[i][1]]

				H, x_edges, y_edges = np.histogram2d(samples_x,
							  samples_y,
							  range=[
								  [self.range_min[combos[i][0]], self.range_max[combos[i][0]]],
								  [self.range_min[combos[i][1]], self.range_max[combos[i][1]]]
							  ],
							  bins=30)
				H2, _, _ = np.histogram2d(samples_2_x,
							  samples_2_y,
							  range=[
								  [self.range_min[combos[i][0]], self.range_max[combos[i][0]]],
								  [self.range_min[combos[i][1]], self.range_max[combos[i][1]]]
							  ],
							  bins=30)

				ax.set_xlabel('${p_%d}_%s$%s' % (p + 1, axl_dict[combos[i][0]], self.unit[combos[i][0]]))
				ax.set_ylabel('${p_%d}_%s$%s' % (p + 1, axl_dict[combos[i][1]], self.unit[combos[i][1]] ))

				extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

				divergence = np.log10(H2/H)
				# divergence[np.logical_not(np.isreal(divergence))] = 0?

				# ax.imshow(divergence, extent=extent, origin='lower', aspect='auto', cmap='viridis')
				fig.colorbar(
					ax.imshow(divergence.T, extent=extent, origin='lower', aspect='auto', cmap='PiYG',
					vmin=-3, vmax=+3)
				)

				# ax.set_ylabel('Frequency')
		fig.tight_layout(pad=2.0)
		return fig, axg

	def _make_histo_mother(self, samples, str='Mother'):
		fig, axg = plt.subplots(1, 3, figsize=(9, 3))
		plt.subplots_adjust(left=0.12,
							bottom=0.13,
							right=0.95,
							top=0.9,
							wspace=0.3,
							hspace=0.1)
		fig.suptitle(str, fontsize=16)
		fig.tight_layout(pad=2.0)

		axl_dict = {0: 'x', 1: 'y', 2: 'z'}
		for i in range(3):
			ax = axg[i]
			dat = samples[:, 0, i]
			dat[dat < self.range_min[i]] = self.range_min[i]
			dat[dat > self.range_max[i]] = self.range_max[i]
			hist, bins = np.histogram(dat, bins=50, range=(self.range_min[i], self.range_max[i]))
			bin_centers = (bins[:-1] + bins[1:]) / 2
			ax.step(bin_centers, hist, where='mid', color='tab:red', linewidth=1)
			ax.set_yscale('log')
			ax.set_xlabel('${p_m}_%s$%s' % (axl_dict[i], self.unit[i]))
			ax.set_ylabel('Frequency')

		return fig, axg

	def _make_hist_pz_pz(self, samples, samples_mother, str='Mother vs Decay'):
		fig, axg = plt.subplots(1, 3, figsize=(9, 4))
		fig.suptitle(str, fontsize=16)
		plt.subplots_adjust(left=0.12,
							bottom=0.13,
							right=0.95,
							top=0.9,
							wspace=0.3,
							hspace=0.1)
		for i in range(3):
			ax = axg[i]
			samples_x = samples_mother[:, 0, 2]*1
			samples_y = samples[:, i, 2].reshape(-1)*1

			samples_x[samples_x < self.range_min[2]] = self.range_min[2]
			samples_x[samples_x > self.range_max[2]] = self.range_max[2]

			samples_y[samples_y < self.range_min[2]] = self.range_min[2]
			samples_y[samples_y > self.range_max[2]] = self.range_max[2]

			h = ax.hist2d(samples_x,
						  samples_y,
						  range=[
							  [self.range_min[2], self.range_max[2]],
							  [self.range_min[2], self.range_max[2]]
						  ],
						  norm=LogNorm(),
						  bins=30)
			if not len(samples_x) == 0:
				fig.colorbar(h[3], ax=ax)
			ax.set_xlabel('${p_m}_z$%s'%(self.unit[2]))
			ax.set_ylabel('${p_%d}_z$%s' % (i + 1, self.unit[2]))

		fig.tight_layout(pad=2.0)

		return fig, axg

	def _make_pz_pz_hist_diff(self, samples, samples_2, samples_mother, str='Mother vs Decay'):
		fig, axg = plt.subplots(1, 3, figsize=(9, 4))
		fig.suptitle(str, fontsize=16)
		plt.subplots_adjust(left=0.12,
							bottom=0.13,
							right=0.95,
							top=0.9,
							wspace=0.3,
							hspace=0.1)
		for i in range(3):
			ax = axg[i]
			samples_x = samples_mother[:, 0, 2]*1
			samples_y = samples[:, i, 2].reshape(-1)*1
			samples_y_2 = samples_2[:, i, 2].reshape(-1)*1

			samples_x[samples_x < self.range_min[2]] = self.range_min[2]
			samples_x[samples_x > self.range_max[2]] = self.range_max[2]

			samples_y[samples_y < self.range_min[2]] = self.range_min[2]
			samples_y[samples_y > self.range_max[2]] = self.range_max[2]

			samples_y_2[samples_y_2 < self.range_min[2]] = self.range_min[2]
			samples_y_2[samples_y_2 > self.range_max[2]] = self.range_max[2]

			H, x_edges, y_edges = np.histogram2d(samples_x,
												 samples_y,
												 range=[
													 [self.range_min[2], self.range_max[2]],
													 [self.range_min[2], self.range_max[2]]
												 ],
												 bins=30)

			H2, _, _ = np.histogram2d(samples_x,
												 samples_y_2,
												 range=[
													 [self.range_min[2], self.range_max[2]],
													 [self.range_min[2], self.range_max[2]]
												 ],
												 bins=30)

			extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

			divergence = np.log10(H2 / H)
			# divergence[np.logical_not(np.isreal(divergence))] = 0?

			# ax.imshow(divergence, extent=extent, origin='lower', aspect='auto', cmap='viridis')
			fig.colorbar(
				ax.imshow(divergence.T, extent=extent, origin='lower', aspect='auto', cmap='PiYG',
						  vmin=-3, vmax=+3)
			)

			ax.set_xlabel('${p_m}_z$%s'%(self.unit[2]))
			ax.set_ylabel('${p_%d}_z$%s' % (i + 1, self.unit[2]))

		fig.tight_layout(pad=2.0)

		return fig, axg

	def _do_plot(self, file, data_samples_, condition=None):
		if condition is not None:
			def replace_text_with_data(match):
				var_name = match.group(1)
				return f"data_samples_['{var_name}']"

			if type(condition) is list:
				assert len(condition) == 2
				condition_code_, condition_render_text = condition
			condition_code = re.sub(r"\{(\w+)\}", replace_text_with_data, condition_code_)

			condition_evaluated = eval(condition_code)
			print(np.unique(data_samples_['particle_1_PID']))
			print(np.unique(data_samples_['particle_2_PID']))
			print(np.unique(data_samples_['particle_3_PID']))
			print(condition_code, np.mean(condition_evaluated))
			condition_render_text += ' [%.2f%%]'%(np.mean(condition_evaluated)*100.0)
			data_samples = {}
			for k,v in data_samples_.items():
				data_samples[k] = v[condition_evaluated]
		else:
			condition_render_text = 'No condition'
			data_samples = data_samples_

		def add_condition_text(fig):
			fig.text(0.05, 0.92, condition_render_text, fontsize=8,
					 verticalalignment='bottom', horizontalalignment='left', color='black')

		with PdfPages(file) as pdf:
			# Create a new figure
			fig, ax = plt.subplots(figsize=(9, 4))  # A4 size in inches

			# Add text to the middle of the page
			ax.text(0.5, 0.5, condition_render_text, ha='center', va='center', fontsize=12)
			ax.set_axis_off()
			# Save the figure to the PDF
			pdf.savefig(fig)
			fig.clear()

			fig, ax = self._make_2d_histo_three_body_prods(data_samples['momenta'], 'True decay products')
			add_condition_text(fig)
			pdf.savefig(fig)
			fig.clear()

			if 'momenta_reconstructed_upp' in data_samples:
				fig, ax = self._make_2d_histo_three_body_prods(data_samples['momenta_reconstructed_upp'], 'Reconstructed decay products')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()


				fig, ax = self._make_2d_three_body_prods_hist_diff(data_samples['momenta'], data_samples['momenta_reconstructed_upp'],
						  'Reconstructed vs true dist. diff. $\\log_{10}(N_{\\mathrm{reco}} / N_{\\mathrm{true}})$')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

			if 'momenta_sampled_upp' in data_samples:
				fig, ax = self._make_2d_histo_three_body_prods(data_samples['momenta_sampled_upp'], 'Sampled decay products')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

				fig, ax = self._make_2d_three_body_prods_hist_diff(data_samples['momenta_sampled_upp'], data_samples['momenta_reconstructed_upp'],
						  'Sampled vs true dist. diff. $\\log_{10}(N_{\\mathrm{sampled}} / N_{\\mathrm{true}})$')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()


			fig, ax = self._make_histo_three_body_prods(data_samples['momenta'], str='True decay products')
			add_condition_text(fig)
			pdf.savefig(fig)
			fig.clear()

			if 'momenta_reconstructed_upp' in data_samples:
				fig, ax = self._make_histo_three_body_prods(data_samples['momenta_reconstructed_upp'], str='Reconstructed decay products')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

				fig, ax = self._make_three_body_prods_hist_diff(data_samples['momenta'], data_samples['momenta_reconstructed_upp'],
																str='Reconstructed vs true decay products', ylabel='true', ylabel_2='reco')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

			if 'momenta_sampled_upp' in data_samples:
				fig, ax = self._make_histo_three_body_prods(data_samples['momenta_sampled_upp'], str='Sampled decay products')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

				fig, ax = self._make_three_body_prods_hist_diff(data_samples['momenta'], data_samples['momenta_sampled_upp'],
																str='Sampled vs true decay products', ylabel='true', ylabel_2='sampled')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

			fig, ax = self._make_histo_mother(data_samples['momenta_mother'], str='Mother')
			add_condition_text(fig)
			pdf.savefig(fig)
			fig.clear()

			fig, ax = self._make_hist_pz_pz(data_samples['momenta'], data_samples['momenta_mother'], str='Mother vs true decay products ')
			add_condition_text(fig)
			pdf.savefig(fig)
			fig.clear()

			if 'momenta_reconstructed_upp' in data_samples:
				fig, ax = self._make_hist_pz_pz(data_samples['momenta_reconstructed_upp'], data_samples['momenta_mother'], str='Mother vs reconstructed decay products ')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

				fig, ax = self._make_pz_pz_hist_diff(data_samples['momenta'], data_samples['momenta_reconstructed_upp'], data_samples['momenta_mother'], str='Mother vs reconstructed decay products hist. diff. $\\log_{10}(N_{\\mathrm{reco}} / N_{\\mathrm{true}})$')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

			if 'momenta_sampled_upp' in data_samples:
				fig, ax = self._make_hist_pz_pz(data_samples['momenta_sampled_upp'], data_samples['momenta_mother'], str='Mother vs sampled decay products ')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

				fig, ax = self._make_pz_pz_hist_diff(data_samples['momenta'], data_samples['momenta_sampled_upp'], data_samples['momenta_mother'], str='Mother vs sampled decay products hist. diff. $\\log_{10}(N_{\\mathrm{sampled}} / N_{\\mathrm{true}})$')
				add_condition_text(fig)
				pdf.savefig(fig)
				fig.clear()

	def plot(self, data_samples, file):
		if type(data_samples) is list:
			data_samples = tensors_dict_join(data_samples)

		assert type(data_samples) is dict

		data_samples_2 = {}
		for k,v in data_samples.items():
			if isinstance(v, torch.Tensor):
				data_samples_2[k] = v.cpu().numpy()
			else:
				data_samples_2[k] = v

		assert len(data_samples_2['momenta'].shape) == 3
		assert len(data_samples_2['momenta_mother'].shape) == 3
		assert data_samples_2['momenta'].shape[1] == 3
		assert data_samples_2['momenta'].shape[2] == 3
		assert data_samples_2['momenta_mother'].shape[2] == 3
		assert data_samples_2['momenta_mother'].shape[1] == 1


		self._do_plot(file+'_%03d.pdf'%0, data_samples_2)
		if self.conditions is not None:
			for i,c in enumerate(self.conditions):
				self._do_plot(file + '_%03d.pdf' % (i+1), data_samples_2, c)






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

	# hist_i = np.histogram(data, range=limits, bins=bins)
	hist_i = np.histogram(data, bins=bins)
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


def plot_latent_space(samples, path='latent_space_'):
	if type(samples) is list:
		samples = tensors_dict_join(samples)
	data_samples_2 = {}
	for k, v in samples.items():
		if isinstance(v, torch.Tensor):
			data_samples_2[k] = v.cpu().numpy()
		else:
			data_samples_2[k] = v
	samples = data_samples_2

	mu = samples['mu']
	log_var = samples['log_var']

	num_dimensions = mu.shape[1]

	dim_max = 0

	pdf = PdfPages(path+'mu.pdf')
	pdf2 = PdfPages(path+'log_var.pdf')

	for i in range(math.ceil(num_dimensions / 16)):
		fig, axs = plt.subplots(4, 4, figsize=(10, 10))
		axs = axs.flatten()

		fig2, axs2 = plt.subplots(4, 4, figsize=(10, 10))
		axs2 = axs2.flatten()

		for i in range(16):
			axs[i].hist(mu[:, dim_max], bins=100)
			axs[i].set_xlabel('mu [dim %d]'%dim_max)

			axs2[i].hist(log_var[:, dim_max], bins=100)
			axs2[i].set_xlabel('log_var [dim %d]'%dim_max)

			dim_max += 1

			if dim_max >= num_dimensions:
				break

		pdf.savefig(fig)
		pdf2.savefig(fig2)

		if dim_max >= num_dimensions:
			break

	pdf.close()
	pdf2.close()

def plot_summaries(all_results, path=None, only_summary=False, t2='sampled'):
	if type(all_results) is list:
		all_results = tensors_dict_join(all_results)
	data_samples_2 = {}
	for k, v in all_results.items():
		if isinstance(v, torch.Tensor):
			data_samples_2[k] = v.cpu().numpy()
		else:
			data_samples_2[k] = v
	all_results = data_samples_2


	array_PID = np.asarray([all_results['particle_1_PID'], all_results['particle_2_PID'], all_results['particle_3_PID'], all_results['mother_PID']])
	unique_combinations = np.swapaxes(np.unique(np.abs(array_PID), axis=1),0,1)
	for unique_combination in unique_combinations:
		print("Check May 24: ", len(unique_combinations), unique_combination)


	results = {}
	results['particle_1_PID'] = all_results['particle_1_PID']
	results['particle_2_PID'] = all_results['particle_2_PID']
	results['particle_3_PID'] = all_results['particle_3_PID']
	results['mother'] = all_results['mother_PID']
	for particle_idx, particle in enumerate(['1','2','3']):
		results[f'particle_{particle}_M'] = all_results[f'particle_{particle}_M']
		for coordinate_idx, coordinate in enumerate(['PX','PY','PZ']):
			results[f'particle_{particle}_{coordinate}'] = all_results['momenta'][:,particle_idx,coordinate_idx]
			if f'momenta_{t2}' in all_results:
				results[f'particle_{particle}_{coordinate}_SAMPLED'] = all_results[f'momenta_{t2}'][:,particle_idx,coordinate_idx]
			else:
				results[f'particle_{particle}_{coordinate}_SAMPLED'] = all_results['momenta'][:,particle_idx,coordinate_idx]
			print('XXUU: ', f'momenta_{t2}', results[f'particle_{particle}_{coordinate}_SAMPLED'].shape)

	results = pd.DataFrame.from_dict(results)

	# results = results.query('particle_1_PZ>0')
	# results = results.query('particle_2_PZ>0')
	# results = results.query('particle_3_PZ>0')

	################################################################################
	# mother (formally B)
	################################################################################
	for tag in ['', "_SAMPLED"]:
		pe_1 = np.sqrt(results.particle_1_M**2 + results[f'particle_1_PX{tag}']**2 + results[f'particle_1_PY{tag}']**2 + results[f'particle_1_PZ{tag}']**2)
		pe_2 = np.sqrt(results.particle_2_M**2 + results[f'particle_2_PX{tag}']**2 + results[f'particle_2_PY{tag}']**2 + results[f'particle_2_PZ{tag}']**2)
		pe_3 = np.sqrt(results.particle_3_M**2 + results[f'particle_3_PX{tag}']**2 + results[f'particle_3_PY{tag}']**2 + results[f'particle_3_PZ{tag}']**2)

		pe = pe_1 + pe_2 + pe_3
		px = results[f'particle_1_PX{tag}'] + results[f'particle_2_PX{tag}'] + results[f'particle_3_PX{tag}']
		py = results[f'particle_1_PY{tag}'] + results[f'particle_2_PY{tag}'] + results[f'particle_3_PY{tag}']
		pz = results[f'particle_1_PZ{tag}'] + results[f'particle_2_PZ{tag}'] + results[f'particle_3_PZ{tag}']
		print("CCC", px.shape, py.shape, pz.shape, pe.shape, type(px), type(py), type(pz), type(pe))


		p_mother = vector.array({"px":px, "py":py, "pz":pz, "E":pe})

		mother_mass = np.sqrt(p_mother.E**2 - p_mother.px**2 - p_mother.py**2 - p_mother.pz**2)

		results[f'particle_mother_M{tag}'] = mother_mass
		results[f'particle_mother_PX{tag}'] = p_mother.px
		results[f'particle_mother_PY{tag}'] = p_mother.py
		results[f'particle_mother_PZ{tag}'] = p_mother.pz

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
		p_32 = vector.array({"px":px, "py":py, "pz":pz, "E":pe})

		mass_32 = np.sqrt(p_32.E**2 - p_32.px**2 - p_32.py**2 - p_32.pz**2)
		results[f'mass_32{tag}'] = np.asarray(mass_32)

		pe = pe_1 + pe_3
		px = results[f'particle_1_PX{tag}'] + results[f'particle_3_PX{tag}']
		py = results[f'particle_1_PY{tag}'] + results[f'particle_3_PY{tag}']
		pz = results[f'particle_1_PZ{tag}'] + results[f'particle_3_PZ{tag}']
		p_13 = vector.array({"px":px, "py":py, "pz":pz, "E":pe})

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
	# I need to export this results dictionary as a readable file!
	# B is supposed to be the mother particle (which can be D+ or B+)
	def symlog(array):
		return np.sign(array)*np.log(np.abs(array)+1)
	
	################################################################################
	# Plotting
	################################################################################

	for unique_combination in unique_combinations:

		print(unique_combination)

		unique_combination_str = f'{unique_combination[3]}_{unique_combination[0]}_{unique_combination[1]}_{unique_combination[2]}_'

		results_i = results.query(f'abs(particle_1_PID)=={unique_combination[0]} and abs(particle_2_PID)=={unique_combination[1]} and abs(particle_3_PID)=={unique_combination[2]}')
		print(results_i.shape, results.shape)

		coordinates = ['PX', 'PY', 'PZ']

		with PdfPages(f'{path}{unique_combination_str}summary_plots.pdf') as pdf:

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
			plt.title(t2)

			particle = "mother"
			mother_name = str(Particle.from_pdgid(unique_combination[3]).name)
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
			#plt.xlabel(f'particle {particle} {coordinate}')
			plt.xlabel(f'particle {mother_name} {coordinate}')
			plt.legend(loc='upper right')

			ax = plt.subplot(3,3,6)
			plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
			plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
			#plt.xlabel(f'particle {particle} {coordinate}')
			plt.xlabel(f'particle {mother_name} {coordinate}')
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
			plt.title(t2)


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
			plt.title(t2)


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
							# hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}'], results_i[f'particle_{particle}_{coordinate_j}'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
							hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}'], results_i[f'particle_{particle}_{coordinate_j}'], bins=35, norm=LogNorm())
							plt.xlabel(f'particle {particle} {coordinate_i}')
							plt.ylabel(f'particle {particle} {coordinate_j}')
							plt.title("Truth")

							ax = plt.subplot(2,2,2)
							# hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED'], results_i[f'particle_{particle}_{coordinate_j}_SAMPLED'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
							hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED'], results_i[f'particle_{particle}_{coordinate_j}_SAMPLED'], bins=35, norm=LogNorm())
							plt.xlabel(f'particle {particle} {coordinate_i}')
							plt.ylabel(f'particle {particle} {coordinate_j}')
							plt.title(t2)


							ax = plt.subplot(2,2,3)
							# hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}']), symlog(results_i[f'particle_{particle}_{coordinate_j}']), range=[range_vec_i_log,range_vec_j_log], bins=35, norm=LogNorm())
							hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}']), symlog(results_i[f'particle_{particle}_{coordinate_j}']), bins=35, norm=LogNorm())
							plt.xlabel(f'particle {particle} SYMLOG({coordinate_i})')
							plt.ylabel(f'particle {particle} SYMLOG({coordinate_j})')
							plt.title("Truth")


							ax = plt.subplot(2,2,4)
							# hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED']), symlog(results_i[f'particle_{particle}_{coordinate_j}_SAMPLED']), range=[range_vec_i_log,range_vec_j_log], bins=35, norm=LogNorm())
							hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED']), symlog(results_i[f'particle_{particle}_{coordinate_j}_SAMPLED']), bins=35, norm=LogNorm())
							plt.xlabel(f'particle {particle} SYMLOG({coordinate_i})')
							plt.ylabel(f'particle {particle} SYMLOG({coordinate_j})')
							plt.title(t2)

							plt.tight_layout()
							pdf.savefig(bbox_inches='tight')
							plt.close()

				plt.figure(figsize=(4,4))
				pdf.savefig(bbox_inches='tight')
				plt.close()

				particle = "mother"

				for coordinate in ['M', 'PX', 'PY', 'PZ']:

					print(f'mother {coordinate}')
				
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
					plt.xlabel(f'particle {mother_name} ...')

					ax = plt.subplot(2,2,2)
					plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}']), limits=range_vec_log, bins=50, label='Truth', c='tab:blue')
					plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED']), limits=range_vec_log, bins=50, label='Sampled', c='tab:orange')
					plt.legend(loc='upper right')
					#plt.xlabel(f'particle {particle} SYMLOG({coordinate})')
					plt.xlabel(f'particle {mother_name} SYMLOG({coordinate})')

					ax = plt.subplot(2,2,3)
					plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}'], limits=range_vec, bins=50, label='Truth', c='tab:blue')
					plot_x_y_yerr(ax, results_i[f'particle_{particle}_{coordinate}_SAMPLED'], limits=range_vec, bins=50, label='Sampled', c='tab:orange')
					#plt.xlabel(f'particle {particle} {coordinate}')
					plt.xlabel(f'particle {mother_name} {coordinate}')
					plt.yscale('log')

					ax = plt.subplot(2,2,4)
					plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}']), limits=range_vec_log, bins=50, label='Truth', c='tab:blue')
					plot_x_y_yerr(ax, symlog(results_i[f'particle_{particle}_{coordinate}_SAMPLED']), limits=range_vec_log, bins=50, label='Sampled', c='tab:orange')
					#plt.xlabel(f'particle {particle} SYMLOG({coordinate})')
					plt.xlabel(f'particle {mother_name} SYMLOG({coordinate})')
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
							# hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}'], results_i[f'particle_{particle}_{coordinate_j}'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
							hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}'], results_i[f'particle_{particle}_{coordinate_j}'], bins=35, norm=LogNorm())
							plt.xlabel(f'{particle} {coordinate_i}')
							plt.ylabel(f'{particle} {coordinate_j}')
							plt.title("Truth")

							ax = plt.subplot(2,2,2)
							# hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED'], results_i[f'particle_{particle}_{coordinate_j}_SAMPLED'], range=[range_vec_i,range_vec_j], bins=35, norm=LogNorm())
							hist = plt.hist2d(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED'], results_i[f'particle_{particle}_{coordinate_j}_SAMPLED'], bins=35, norm=LogNorm())
							plt.xlabel(f'{particle} {coordinate_i}')
							plt.ylabel(f'{particle} {coordinate_j}')
							plt.title(t2)

							ax = plt.subplot(2,2,3)
							# hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}']), symlog(results_i[f'particle_{particle}_{coordinate_j}']), range=[range_vec_i_log,range_vec_j_log], bins=35, norm=LogNorm())
							hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}']), symlog(results_i[f'particle_{particle}_{coordinate_j}']), bins=35, norm=LogNorm())
							plt.xlabel(f'{particle} SYMLOG({coordinate_i})')
							plt.ylabel(f'{particle} SYMLOG({coordinate_j})')
							plt.title("Truth")


							ax = plt.subplot(2,2,4)
							# hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED']), symlog(results_i[f'particle_{particle}_{coordinate_j}_SAMPLED']), range=[range_vec_i_log,range_vec_j_log], bins=35, norm=LogNorm())
							hist = plt.hist2d(symlog(results_i[f'particle_{particle}_{coordinate_i}_SAMPLED']), symlog(results_i[f'particle_{particle}_{coordinate_j}_SAMPLED']), bins=35, norm=LogNorm())
							plt.xlabel(f'{particle} SYMLOG({coordinate_i})')
							plt.ylabel(f'{particle} SYMLOG({coordinate_j})')
							plt.title(t2)

							plt.tight_layout()
							pdf.savefig(bbox_inches='tight')
							plt.close()

				plt.figure(figsize=(4,4))
				pdf.savefig(bbox_inches='tight')
				plt.close()

				plt.figure(figsize=(10,5))

				ax = plt.subplot(1,2,1)
				# plt.hist2d(results_i.mass_32**2, results_i.mass_13**2, bins=50,
				# 		range=[[0, np.amax(results_i.mass_32**2)], [0, np.amax(results_i.mass_13**2)]],
				# 		norm=LogNorm())
				plt.hist2d(results_i.mass_32**2, results_i.mass_13**2, bins=50,
						norm=LogNorm())
				plt.xlabel(r"mass$_{32}^2$")
				plt.ylabel(r"mass$_{13}^2$")
				plt.title("Truth")
				ax = plt.subplot(1,2,2)
				# plt.hist2d(results_i.mass_32_SAMPLED**2, results_i.mass_13_SAMPLED**2, bins=50,
				# 		range=[[0, np.amax(results_i.mass_32**2)], [0, np.amax(results_i.mass_13**2)]],
				# 		norm=LogNorm())
				plt.hist2d(results_i.mass_32_SAMPLED**2, results_i.mass_13_SAMPLED**2, bins=50,
						norm=LogNorm())
				plt.xlabel(r"mass$_{32}^2$")
				plt.ylabel(r"mass$_{13}^2$")
				plt.title(t2)

				plt.tight_layout()
				pdf.savefig(bbox_inches='tight')
				plt.close()


	
		print(f'{unique_combination_str} done')

	return results