# import jetnet
# from jetnet.datasets import JetNet
# from jetnet import evaluation
# from jetnet.datasets.normalisations import FeaturewiseLinearBounded, FeaturewiseLinear
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


import numpy as np

from os import remove
from os.path import exists

from tqdm import tqdm

import logging
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


from itertools import permutations



# loss = "og"
loss = "ls"
# loss = "w"


num_particles=3
latent_node_size=10+1 # 1 for mass
# G = model.MPNet_G(num_particles,latent_node_size, output_node_size=3, final_activation='tanh', mp_iters=2).to(device)
# latent_input_size = 3+1 # 1 for mass
# if loss == "w":
# 	D = model.MPNet_D(num_particles,latent_input_size, output_node_size=2, final_activation='linear', mp_iters=2, linear_args={'dropout_p':0.1}).to(device)
# else:
# 	D = model.MPNet_D(num_particles,latent_input_size, output_node_size=2, final_activation='sigmoid', mp_iters=2, linear_args={'dropout_p':0.1}).to(device)
# # D = model.MPDiscriminator(num_particles,3, output_node_size=1, final_activation='sigmoid', mp_iters=2, linear_args={'dropout_p':0.1}).to(device)



def permutate_particle_number(in_array):
	'''
	put this in data object
	'''
	a = np.array([0, 1, 2])
	perms = np.empty((0,np.shape(a)[0]))
	for perm in set(permutations(a)):
		perm = np.expand_dims(np.asarray(perm),0).astype(int)
		perms = np.append(perms, perm, axis=0)
	n_groups = np.shape(perms)[0]
	choice = np.random.choice(np.arange(n_groups), size=np.shape(in_array)[0], replace=True)

	for n_group in range(n_groups):
		perm = perms[n_group]
		where = np.where(choice==n_group)
		in_array_where = in_array[where]
		in_array_where_permed = in_array_where.copy()

		in_array_where_permed[:,0] = in_array_where[:,int(perm[0])]
		in_array_where_permed[:,1] = in_array_where[:,int(perm[1])]
		in_array_where_permed[:,2] = in_array_where[:,int(perm[2])]

		in_array[where] = in_array_where_permed

	return in_array

import rlasim.lib.organise_data as data
import vector

# dataset, preprocessors = data.get_data_simple('example_tree.root')
# dataset, preprocessors = data.get_data_simple('training_LARGE.root')
dataset, preprocessors = data.get_data_simple('data/example_tree.root')
# dataset.permutate_particle_number()


momenta = dataset.get_data_i("momenta", mode="train").astype("float32")
momenta_pp = dataset.get_data_i("momenta", mode="train", preprocessed=True).astype("float32")

# print(np.sum((momenta-momenta_pp)))
0/0

# print(momenta.shape)
# plt.hist(momenta_pp[:, :, 2])
# plt.savefig('check.pdf')
# plt.hist(momenta[:, :, 2])
# plt.savefig('check2.pdf')


momenta_pp_test = dataset.get_data_i("momenta", mode="test", preprocessed=True).astype("float32")
masses_test = dataset.get_data_i("masses", mode="test").astype("float32")

B_angles = dataset.get_data_i("B_angles", mode="train").astype("float32")
B_angles_pp = dataset.get_data_i("B_angles", mode="train", preprocessed=True).astype("float32")


masses = dataset.get_data_i("masses", mode="train").astype("float32")


indexing_dict = {0: 'x', 1:'y', 2:'z'}

with PdfPages(f'momenta_pp_test2.pdf') as pdf:
	for particle in range(3):
		fig, axes = plt.subplots(1, 3, figsize=(14, 4))
		subplot_idx = 0
		for i in range(3):
			for j in range(i+1, 3):
				strx = 'particle_%d_p%s (minmax norm.)'%(particle, indexing_dict[i])
				stry = 'particle_%d_p%s (minmax norm.)'%(particle, indexing_dict[j])

				h = axes[subplot_idx].hist2d(momenta_pp_test[:,particle,i], momenta_pp_test[:,particle,j], bins=50, norm=LogNorm(), range=[[-1,1],[-1,1]])
				fig.colorbar(h[3], ax=axes[subplot_idx])
				axes[subplot_idx].set_xlabel(strx)
				axes[subplot_idx].set_ylabel(stry)
				subplot_idx += 1
		fig.tight_layout(pad=1.0)
		pdf.savefig()
		plt.close('all')


# with PdfPages(f'momenta_pp_test.pdf') as pdf:
#
# 	for particle in range(3):
# 		plt.figure(figsize=(12,4))
# 		subplot_idx = 0
# 		for i in range(3):
# 			for j in range(i+1, 3):
# 				subplot_idx += 1
# 				plt.subplot(1,3,subplot_idx)
# 				plt.hist2d(momenta_pp_test[:,particle,i], momenta_pp_test[:,particle,j], bins=50, norm=LogNorm(), range=[[-1,1],[-1,1]])
# 		pdf.savefig(bbox_inches='tight')
# 		plt.close('all')
#
# 	momenta_pp_test = preprocessors["momenta"].postprocess(momenta_pp_test)
#
# 	P1_E = np.sqrt(masses_test[:,0,0]**2 + momenta_pp_test[:,0,0]**2 + momenta_pp_test[:,0,1]**2 + momenta_pp_test[:,0,2]**2)
# 	P2_E = np.sqrt(masses_test[:,1,0]**2 + momenta_pp_test[:,1,0]**2 + momenta_pp_test[:,1,1]**2 + momenta_pp_test[:,1,2]**2)
# 	P3_E = np.sqrt(masses_test[:,2,0]**2 + momenta_pp_test[:,2,0]**2 + momenta_pp_test[:,2,1]**2 + momenta_pp_test[:,2,2]**2)
# 	pe = P1_E + P2_E + P3_E
# 	px = momenta_pp_test[:,0,0] + momenta_pp_test[:,1,0] + momenta_pp_test[:,2,0]
# 	py = momenta_pp_test[:,0,1] + momenta_pp_test[:,1,1] + momenta_pp_test[:,2,1]
# 	pz = momenta_pp_test[:,0,2] + momenta_pp_test[:,1,2] + momenta_pp_test[:,2,2]
# 	B = vector.obj(px=px, py=py, pz=pz, E=pe)
# 	Bmass = np.sqrt(B.E**2 - B.px**2 - B.py**2 - B.pz**2)
#
# 	plt.figure(figsize=(6,4))
# 	plt.hist(Bmass, bins=50)
# 	pdf.savefig(bbox_inches='tight')
# 	plt.close('all')
#
# 	for particle in range(3):
# 		plt.figure(figsize=(12,4))
# 		subplot_idx = 0
# 		for i in range(3):
# 			for j in range(i+1, 3):
# 				subplot_idx += 1
# 				plt.subplot(1,3,subplot_idx)
# 				plt.hist2d(momenta_pp_test[:,particle,i], momenta_pp_test[:,particle,j], bins=50, norm=LogNorm())
# 		pdf.savefig(bbox_inches='tight')
# 		plt.close('all')
