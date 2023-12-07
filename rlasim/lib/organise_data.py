# Copied from
# https://gitlab.cern.ch/amarshal/dla-2-detector-response-mpgan/-/blob/main/tools/organise_data.py

import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vector
import math

from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import QuantileTransformer
from itertools import permutations
from typing import List, Optional, Sequence, Union, Any, Callable
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch import nn

from rlasim.lib.data_core import DictTensorDataset, tensors_dict_join, RootTensorDataset, RootTensorDataset2, \
    nbe_default_collate, RootBlockShuffledSubsetDataLoader
from rlasim.lib.progress_bar import AsyncProgressBar


def rotation_matrix_from_vectors_vectorised(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    norm_vec1 = np.swapaxes(
        np.asarray([np.linalg.norm(vec1, axis=1), np.linalg.norm(vec1, axis=1), np.linalg.norm(vec1, axis=1)]), 0, 1)
    norm_vec2 = np.swapaxes(
        np.asarray([np.linalg.norm(vec2, axis=1), np.linalg.norm(vec2, axis=1), np.linalg.norm(vec2, axis=1)]), 0, 1)
    shape = np.shape(vec1)[0]
    a, b = (vec1 / norm_vec1).reshape(shape, 3), (vec2 / norm_vec2).reshape(shape, 3)
    v = np.cross(a, b, axis=1)
    c = np.array([a[i, :].dot(b[i, :]) for i in range(shape)])
    s = np.linalg.norm(v, axis=1)
    kmat = np.array([[np.zeros(shape), -v[:, 2], v[:, 1]], [v[:, 2], np.zeros(shape), -v[:, 0]],
                     [-v[:, 1], v[:, 0], np.zeros(shape)]])
    rotation_matrix = np.array(
        [np.eye(3) + kmat[:, :, i] + kmat[:, :, i].dot(kmat[:, :, i]) * ((1 - c[i]) / (s[i] ** 2)) for i in
         range(shape)])
    return rotation_matrix


def mag(vec):
    sum_sqs = 0
    for component in vec:
        sum_sqs += component ** 2
    mag = np.sqrt(sum_sqs)
    return mag


def norm(vec):
    mag_vec = mag(vec)
    for component_idx in range(np.shape(vec)[0]):
        vec[component_idx] *= (1. / mag_vec)
    return vec


def dot(vec1, vec2):
    dot = 0
    for component_idx in range(np.shape(vec1)[0]):
        dot += vec1[component_idx] * vec2[component_idx]
    return dot


def rot_vectorised(vec, mat):
    rot_A = np.expand_dims(vec[0] * mat[:, 0, 0] + vec[1] * mat[:, 0, 1] + vec[2] * mat[:, 0, 2], 0)
    rot_B = np.expand_dims(vec[0] * mat[:, 1, 0] + vec[1] * mat[:, 1, 1] + vec[2] * mat[:, 1, 2], 0)
    rot_C = np.expand_dims(vec[0] * mat[:, 2, 0] + vec[1] * mat[:, 2, 1] + vec[2] * mat[:, 2, 2], 0)

    vec = np.concatenate((rot_A, rot_B, rot_C), axis=0)
    return vec

    reshaped = np.asarray([[training_parameters["B_phi"], training_parameters["B_theta"], training_parameters["B_P"]]])


class B_angles_preprocessor():

    def __init__(self, sample):

        self.limits = self.get_limits_from_samples(sample)

    def get_limits(self):
        return self.limits

    def get_limits_from_samples(self, sample):

        processing_limits = {}
        processing_limits['B_phi'] = {}
        processing_limits['B_theta'] = {}
        processing_limits['B_P'] = {}

        dataset_copy = sample.copy()

        p_idx = 0

        for j, var in enumerate(['B_phi', 'B_theta', 'B_P']):
            if 'pz' in var:
                dataset_copy[:, p_idx, j] = dataset_copy[:, p_idx, j] + 5
                dataset_copy[:, p_idx, j] = np.log(dataset_copy[:, p_idx, j])

        for j, var in enumerate(['B_phi', 'B_theta', 'B_P']):
            _min = np.amin(dataset_copy[:, :, j])
            _max = np.amax(dataset_copy[:, :, j])
            try:
                if processing_limits[var]['min'] > _min: processing_limits[var]['min'] = _min
                if processing_limits[var]['max'] < _max: processing_limits[var]['max'] = _max
            except:
                processing_limits[var]['min'] = _min
                processing_limits[var]['max'] = _max

        for j, var in enumerate(['B_phi', 'B_theta', 'B_P']):
            if processing_limits[var]['min'] > 0.:
                processing_limits[var]['min'] *= 0.9
            else:
                processing_limits[var]['min'] *= 1.1

            if processing_limits[var]['max'] < 0.:
                processing_limits[var]['max'] *= 0.9
            else:
                processing_limits[var]['max'] *= 1.1

        processing_limits['B_phi']['min'] = -math.pi
        processing_limits['B_phi']['max'] = math.pi

        return processing_limits

    def preprocess(self, sample):

        dataset_out = sample.copy()
        p_idx = 0

        for j, var in enumerate(['B_phi', 'B_theta', 'B_P']):
            if 'pz' in var:
                dataset_out[:, p_idx, j] = dataset_out[:, p_idx, j] + 5
                dataset_out[:, p_idx, j] = np.log(dataset_out[:, p_idx, j])

            # for j, var in enumerate([f'P{particle}_px',f'P{particle}_py',f'P{particle}_pz']):
            range_i = self.limits[var]['max'] - self.limits[var]['min']
            dataset_out[:, p_idx, j] = ((dataset_out[:, p_idx, j] - self.limits[var]['min']) / range_i) * 2. - 1.

        return dataset_out

    def postprocess(self, sample):

        dataset_out = sample.copy()
        p_idx = 0

        for j, var in enumerate(['B_phi', 'B_theta', 'B_P']):
            range_i = self.limits[var]['max'] - self.limits[var]['min']
            dataset_out[:, p_idx, j] = (((dataset_out[:, p_idx, j] + 1.) / 2.) * (range_i) + self.limits[var]['min'])

            # for j, var in enumerate([f'P{particle}_px',f'P{particle}_py',f'P{particle}_pz']):
            if 'pz' in var:
                dataset_out[:, p_idx, j] = np.exp(dataset_out[:, p_idx, j])
                dataset_out[:, p_idx, j] = dataset_out[:, p_idx, j] - 5.

        return dataset_out


class MomentaPreprocessor():

    def __init__(self, sample, sample_mother):

        self.limits = self.get_limits_from_samples(sample, sample_mother)

    def get_limits(self):
        return self.limits

    def get_limits_from_samples(self, sample, sample_mother):

        processing_limits = {}
        processing_limits['P1_px'] = {}
        processing_limits['P1_py'] = {}
        processing_limits['P1_pz'] = {}
        processing_limits['P2_px'] = {}
        processing_limits['P2_py'] = {}
        processing_limits['P2_pz'] = {}
        processing_limits['P3_px'] = {}
        processing_limits['P3_py'] = {}
        processing_limits['P3_pz'] = {}

        processing_limits['PM_px'] = {}
        processing_limits['PM_py'] = {}
        processing_limits['PM_pz'] = {}

        dataset_copy = sample.copy()
        dataset_mother_copy = sample.copy()

        for p_idx, particle in enumerate([1, 2, 3]):
            for j, var in enumerate([f'P{particle}_px', f'P{particle}_py', f'P{particle}_pz']):
                if 'pz' in var:
                    dataset_copy[:, p_idx, j] = dataset_copy[:, p_idx, j] + 5
                    dataset_copy[:, p_idx, j] = np.log(dataset_copy[:, p_idx, j])

        for j, var in enumerate([f'PM_px', f'PM_py', f'PM_pz']):
            if 'pz' in var:
                dataset_copy[:, 0, j] = dataset_copy[:, 0, j] + 5
                dataset_copy[:, 0, j] = np.log(dataset_copy[:, 0, j])

        for p_idx, particle in enumerate([1, 2, 3]):
            for j, var in enumerate([f'P{particle}_px', f'P{particle}_py', f'P{particle}_pz']):
                _min = np.amin(dataset_copy[:, :, j])
                _max = np.amax(dataset_copy[:, :, j])
                try:
                    if processing_limits[var]['min'] > _min: processing_limits[var]['min'] = _min
                    if processing_limits[var]['max'] < _max: processing_limits[var]['max'] = _max
                except:
                    processing_limits[var]['min'] = _min
                    processing_limits[var]['max'] = _max

            for j, var in enumerate([f'P{particle}_px', f'P{particle}_py', f'P{particle}_pz']):
                if processing_limits[var]['min'] > 0.:
                    processing_limits[var]['min'] *= 0.9
                else:
                    processing_limits[var]['min'] *= 1.1

                if processing_limits[var]['max'] < 0.:
                    processing_limits[var]['max'] *= 0.9
                else:
                    processing_limits[var]['max'] *= 1.1

        return processing_limits

    def preprocess(self, sample):

        dataset_out = sample.copy()

        if sample.shape[1] == 3:
            for p_idx, particle in enumerate([1, 2, 3]):

                for j, var in enumerate([f'P{particle}_px', f'P{particle}_py', f'P{particle}_pz']):
                    if 'pz' in var:
                        dataset_out[:, p_idx, j] = dataset_out[:, p_idx, j] + 5
                        dataset_out[:, p_idx, j] = np.log(dataset_out[:, p_idx, j])

                    # for j, var in enumerate([f'P{particle}_px',f'P{particle}_py',f'P{particle}_pz']):
                    range_i = self.limits[var]['max'] - self.limits[var]['min']
                    dataset_out[:, p_idx, j] = ((dataset_out[:, p_idx, j] - self.limits[var]['min']) / range_i) * 2. - 1.
        else:
            for j, var in enumerate([f'PM_px', f'PM_py', f'PM_pz']):
                if 'pz' in var:
                    dataset_out[:, 0, j] = dataset_out[:, 0, j] + 5
                    dataset_out[:, 0, j] = np.log(dataset_out[:, 0, j])

                # for j, var in enumerate([f'P{particle}_px',f'P{particle}_py',f'P{particle}_pz']):
                range_i = self.limits[var]['max'] - self.limits[var]['min']
                dataset_out[:, 0, j] = ((dataset_out[:, 0, j] - self.limits[var]['min']) / range_i) * 2. - 1.

        return dataset_out

    def postprocess(self, sample):

        dataset_out = sample.copy()

        for p_idx, particle in enumerate([1, 2, 3]):

            for j, var in enumerate([f'P{particle}_px', f'P{particle}_py', f'P{particle}_pz']):
                range_i = self.limits[var]['max'] - self.limits[var]['min']
                dataset_out[:, p_idx, j] = (
                            ((dataset_out[:, p_idx, j] + 1.) / 2.) * (range_i) + self.limits[var]['min'])

                # for j, var in enumerate([f'P{particle}_px',f'P{particle}_py',f'P{particle}_pz']):
                if 'pz' in var:
                    dataset_out[:, p_idx, j] = np.exp(dataset_out[:, p_idx, j])
                    dataset_out[:, p_idx, j] = dataset_out[:, p_idx, j] - 5.

        return dataset_out


class CoM_momenta_preprocessor():

    def __init__(self, sample):

        self.limits = self.get_limits_from_samples(sample)

    def get_limits(self):
        return self.limits

    def get_limits_from_samples(self, sample):

        processing_limits = {}
        processing_limits['P1_p'] = {}
        processing_limits['P2_px'] = {}
        processing_limits['P2_py'] = {}
        processing_limits['P2_pz'] = {}
        processing_limits['P3_px'] = {}
        processing_limits['P3_py'] = {}
        processing_limits['P3_pz'] = {}

        dataset_copy = sample.copy()

        for j, var in enumerate(['P1_p', 'P2_px', 'P2_py', 'P2_pz', 'P3_px', 'P3_py', 'P3_pz']):
            _min = np.amin(dataset_copy[:, j])
            _max = np.amax(dataset_copy[:, j])
            try:
                if processing_limits[var]['min'] > _min: processing_limits[var]['min'] = _min
                if processing_limits[var]['max'] < _max: processing_limits[var]['max'] = _max
            except:
                processing_limits[var]['min'] = _min
                processing_limits[var]['max'] = _max

        for j, var in enumerate(['P1_p', 'P2_px', 'P2_py', 'P2_pz', 'P3_px', 'P3_py', 'P3_pz']):
            if processing_limits[var]['min'] > 0.:
                processing_limits[var]['min'] *= 0.9
            else:
                processing_limits[var]['min'] *= 1.1

            if processing_limits[var]['min'] < 0.:
                processing_limits[var]['max'] *= 0.9
            else:
                processing_limits[var]['max'] *= 1.1

        processing_limits['P1_p']['min'] = 0.

        return processing_limits

    def preprocess(self, sample):

        dataset_out = sample.copy()

        for j, var in enumerate(['P1_p', 'P2_px', 'P2_py', 'P2_pz', 'P3_px', 'P3_py', 'P3_pz']):
            range_i = self.limits[var]['max'] - self.limits[var]['min']
            dataset_out[:, j] = ((dataset_out[:, j] - self.limits[var]['min']) / range_i) * 2. - 1.

        return dataset_out

    def postprocess(self, sample):

        dataset_out = sample.copy()

        for j, var in enumerate(['P1_p', 'P2_px', 'P2_py', 'P2_pz', 'P3_px', 'P3_py', 'P3_pz']):
            range_i = self.limits[var]['max'] - self.limits[var]['min']
            dataset_out[:, j] = (((dataset_out[:, j] + 1.) / 2.) * (range_i) + self.limits[var]['min'])

        return dataset_out


class CoM_angles_preprocessor():

    def __init__(self, sample):

        self.limits = self.get_limits_from_samples(sample)

    def get_limits(self):
        return self.limits

    def get_limits_from_samples(self, sample):

        processing_limits = {}
        processing_limits['phi_P2'] = {}
        processing_limits['phi_P3'] = {}
        processing_limits['theta_P2'] = {}
        processing_limits['theta_P3'] = {}

        for j, var in enumerate(['theta_P2', 'theta_P3']):
            processing_limits[var]['min'] = 0
            processing_limits[var]['max'] = math.pi / 2.

        for j, var in enumerate(['phi_P2', 'phi_P3']):
            processing_limits[var]['min'] = 0.
            processing_limits[var]['max'] = math.pi

        return processing_limits

    def preprocess(self, sample):

        dataset_out = sample.copy()

        # dataset_out[np.where(dataset_out[:,0]<0)[0],2] += -math.pi
        # dataset_out[np.where(dataset_out[:,0]<0)[0],0] += math.pi

        where = np.where(dataset_out[:, 0] < 0)
        dataset_out[where, 0] = dataset_out[where, 0] * -1.

        where = np.where(dataset_out[:, 2] < 0)
        dataset_out[where, 2] = dataset_out[where, 2] * -1.

        where = np.where(dataset_out[:, 1] > math.pi / 2.)
        dataset_out[where, 1] = ((dataset_out[where, 1] - math.pi / 2.) * -1.) + math.pi / 2.
        where = np.where(dataset_out[:, 1] < -math.pi / 2.)
        dataset_out[where, 1] = ((dataset_out[where, 1] + math.pi / 2.) * -1.) - math.pi / 2.

        where = np.where(dataset_out[:, 3] > math.pi / 2.)
        dataset_out[where, 3] = ((dataset_out[where, 3] - math.pi / 2.) * -1.) + math.pi / 2.
        where = np.where(dataset_out[:, 3] < -math.pi / 2.)
        dataset_out[where, 3] = ((dataset_out[where, 3] + math.pi / 2.) * -1.) - math.pi / 2.

        where = np.where(dataset_out[:, 1] < 0)
        dataset_out[where, 1] = dataset_out[where, 1] * -1.
        where = np.where(dataset_out[:, 3] < 0)
        dataset_out[where, 3] = dataset_out[where, 3] * -1.

        for j, var in enumerate(['phi_P2', 'theta_P2', 'phi_P3', 'theta_P3']):
            range_i = self.limits[var]['max'] - self.limits[var]['min']
            dataset_out[:, j] = ((dataset_out[:, j] - self.limits[var]['min']) / range_i) * 2. - 1.

        return dataset_out

    def postprocess(self, sample):

        dataset_out = sample.copy()

        for j, var in enumerate(['phi_P2', 'theta_P2', 'phi_P3', 'theta_P3']):
            range_i = self.limits[var]['max'] - self.limits[var]['min']
            dataset_out[:, j] = (((dataset_out[:, j] + 1.) / 2.) * (range_i) + self.limits[var]['min'])

        # barrier = dataset_out[:,0] - math.pi
        # where_upper = np.greater(dataset_out[:,2],barrier)
        # where_upper = np.where(where_upper)
        # dataset_out[where_upper[0],2] += math.pi
        # dataset_out[where_upper[0],0] += -math.pi

        random_sign = np.random.choice([-1., 1.], size=np.shape(dataset_out)[0])
        dataset_out[:, 3] = dataset_out[:, 3] * random_sign * -1.
        dataset_out[:, 1] = dataset_out[:, 1] * random_sign

        random_sign = np.random.choice([-1., 1.], size=np.shape(dataset_out)[0])
        where = np.where((random_sign == -1.) & (dataset_out[:, 3] > 0))
        dataset_out[where[0], 3] = ((dataset_out[where[0], 3] - math.pi / 2.) * -1.) + math.pi / 2.

        random_sign2 = random_sign * -1
        where = np.where((random_sign2 == -1.) & (dataset_out[:, 3] < 0))
        dataset_out[where[0], 3] = ((dataset_out[where[0], 3] + math.pi / 2.) * -1.) - math.pi / 2.

        where = np.where((random_sign == -1.) & (dataset_out[:, 1] > 0))
        dataset_out[where[0], 1] = ((dataset_out[where[0], 1] - math.pi / 2.) * -1.) + math.pi / 2.

        where = np.where((random_sign2 == -1.) & (dataset_out[:, 1] < 0))
        dataset_out[where[0], 1] = ((dataset_out[where[0], 1] + math.pi / 2.) * -1.) - math.pi / 2.

        where = np.where((dataset_out[:, 1] > math.pi / 2.) | (dataset_out[:, 1] < -math.pi / 2.))
        dataset_out[where[0], 0] = dataset_out[where[0], 0] * -1.

        where = np.where((dataset_out[:, 3] > math.pi / 2.) | (dataset_out[:, 3] < -math.pi / 2.))
        dataset_out[where[0], 2] = dataset_out[where[0], 2] * -1.

        return dataset_out


class B_properties_preprocessor():

    def __init__(self, samples):

        self.limits = self.get_limits_from_samples(samples)

        self.trans = QuantileTransformer(n_quantiles=500, output_distribution='normal')
        X = np.random.uniform(low=-math.pi, high=math.pi, size=1000000)
        X = np.expand_dims(X, 1)
        self.trans.fit(X)

    def get_limits(self):
        return self.limits

    def get_limits_from_samples(self, samples):

        processing_limits = {}
        processing_limits['B_pt'] = {}
        processing_limits['B_phi'] = {}
        # processing_limits['B_px'] = {}
        # processing_limits['B_py'] = {}
        processing_limits['B_pz'] = {}
        # processing_limits['B_P'] = {}

        for sample in samples:

            dataset_copy = sample.copy()

            dataset_copy[:, 2] = dataset_copy[:, 2] + 5
            dataset_copy[:, 2] = np.log(dataset_copy[:, 2])

            # dataset_copy[:,5] = dataset_copy[:,5] + 5
            # dataset_copy[:,5] = np.log(dataset_copy[:,5])

            # for j, var in enumerate(['B_pt','B_phi','B_px','B_py','B_pz','B_P']):
            for j, var in enumerate(['B_pt', 'B_phi', 'B_pz']):
                _min = np.amin(dataset_copy[:, j])
                _max = np.amax(dataset_copy[:, j])
                if var in ['B_pt']:
                    _min = 0.
                try:
                    if processing_limits[var]['min'] > _min: processing_limits[var]['min'] = _min
                    if processing_limits[var]['max'] < _max: processing_limits[var]['max'] = _max
                except:
                    processing_limits[var]['min'] = _min
                    processing_limits[var]['max'] = _max

        # for j, var in enumerate(['B_pt','B_phi','B_px','B_py','B_pz','B_P']):
        for j, var in enumerate(['B_pt', 'B_phi', 'B_pz']):
            if processing_limits[var]['min'] > 0.:
                processing_limits[var]['min'] *= 0.9
            else:
                processing_limits[var]['min'] *= 1.1

            if processing_limits[var]['min'] < 0.:
                processing_limits[var]['max'] *= 0.9
            else:
                processing_limits[var]['max'] *= 1.1

        # processing_limits['B_phi']['min'] = -math.pi*1.5
        # processing_limits['B_phi']['max'] = math.pi*1.5

        processing_limits['B_phi']['min'] = -math.pi
        processing_limits['B_phi']['max'] = math.pi

        return processing_limits

    def preprocess(self, sample):

        dataset_out = sample.copy()

        dataset_out[:, 2] = dataset_out[:, 2] + 5
        dataset_out[:, 2] = np.log(dataset_out[:, 2])

        # dataset_out[:,5] = dataset_out[:,5] + 5
        # dataset_out[:,5] = np.log(dataset_out[:,5])

        # for j, var in enumerate(['B_pt','B_phi','B_px','B_py','B_pz','B_P']):
        for j, var in enumerate(['B_pt', 'B_phi', 'B_pz']):
            # if var == 'B_phi':
            # 	dataset_out[:,j] = (self.trans.transform(np.expand_dims(dataset_out[:,j],1))[:,0])/7.
            # else:
            range_i = self.limits[var]['max'] - self.limits[var]['min']
            dataset_out[:, j] = ((dataset_out[:, j] - self.limits[var]['min']) / range_i) * 2. - 1.

        return dataset_out

    def postprocess(self, sample):

        dataset_out = sample.copy()

        # for j, var in enumerate(['B_pt','B_phi','B_px','B_py','B_pz','B_P']):
        for j, var in enumerate(['B_pt', 'B_phi', 'B_pz']):
            # if var == 'B_phi':
            # 	dataset_out[:,j] = self.trans.inverse_transform(np.expand_dims((dataset_out[:,j]*7.),1))[:,0]
            # else:
            range_i = self.limits[var]['max'] - self.limits[var]['min']
            dataset_out[:, j] = (((dataset_out[:, j] + 1.) / 2.) * (range_i) + self.limits[var]['min'])

        dataset_out[:, 2] = np.exp(dataset_out[:, 2])
        dataset_out[:, 2] = dataset_out[:, 2] - 5.

        # dataset_out[:,5] = np.exp(dataset_out[:,5])
        # dataset_out[:,5] = dataset_out[:,5] - 5.

        return dataset_out


class auxilary_preprocessor():

    def __init__(self, samples):

        self.dims = np.shape(samples)[1]
        self.limits = self.get_limits_from_samples(samples)

    def get_limits(self):
        return self.limits

    def get_limits_from_samples(self, samples):

        processing_limits = {}

        for index in range(self.dims):
            sample_i = samples[:, index]

            min_i = 0.
            max_i = np.amax(sample_i) * 1.2

            processing_limits[index] = {}
            processing_limits[index]['min'] = min_i
            processing_limits[index]['max'] = max_i

        return processing_limits

    def preprocess(self, sample):

        dataset_out = sample.copy()

        for index in range(self.dims):
            range_i = self.limits[index]['max'] - self.limits[index]['min']
            dataset_out[:, index] = ((dataset_out[:, index] - self.limits[index]['min']) / range_i) * 2. - 1.

        return dataset_out

    def postprocess(self, sample):

        dataset_out = sample.copy()

        for index in range(self.dims):
            range_i = self.limits[index]['max'] - self.limits[index]['min']
            dataset_out[:, index] = (((dataset_out[:, index] + 1.) / 2.) * (range_i) + self.limits[index]['min'])

        return dataset_out


class momentum_preprocessor():

    def __init__(self, samples):

        self.limits = self.get_limits_from_samples(samples)

    def get_limits(self):
        return self.limits

    def get_limits_from_samples(self, samples):

        processing_limits = {}
        processing_limits['px'] = {}
        processing_limits['py'] = {}
        processing_limits['pz'] = {}

        for sample in samples:

            dataset_copy = sample.copy()
            dataset_copy[:, :, 2] = dataset_copy[:, :, 2] + 5
            dataset_copy[:, :, 2] = np.log(dataset_copy[:, :, 2])

            for i, particle_i in enumerate(np.arange(1, np.shape(dataset_copy)[1] + 1)):
                for j, mom in enumerate(['px', 'py', 'pz']):
                    mom_min = np.amin(dataset_copy[:, i, j])
                    mom_max = np.amax(dataset_copy[:, i, j])
                    if mom in ['px', 'py']:
                        if np.abs(mom_min) > np.abs(mom_max):
                            mom_max = np.abs(mom_min)
                        else:
                            mom_min = -1. * mom_max
                    try:
                        if processing_limits[mom]['min'] > mom_min: processing_limits[mom]['min'] = mom_min
                        if processing_limits[mom]['max'] < mom_max: processing_limits[mom]['max'] = mom_max
                    except:
                        processing_limits[mom]['min'] = mom_min
                        processing_limits[mom]['max'] = mom_max

        for j, mom in enumerate(['px', 'py', 'pz']):
            processing_limits[mom]['min'] *= 1.1
            processing_limits[mom]['max'] *= 1.1

        return processing_limits

    def preprocess(self, sample):

        dataset_out = sample.copy()

        dataset_out[:, :, 2] = dataset_out[:, :, 2] + 5.
        dataset_out[:, :, 2] = np.log(dataset_out[:, :, 2])

        # for idx in [0,1]:
        # 	dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]<0.)] = -1.*np.sqrt(np.abs(dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]<0.)]))
        # 	dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]>0.)] = np.sqrt(dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]>0.)])

        for j, mom in enumerate(['px', 'py', 'pz']):
            range_i = self.limits[mom]['max'] - self.limits[mom]['min']
            dataset_out[:, :, j] = ((dataset_out[:, :, j] - self.limits[mom]['min']) / range_i) * 2. - 1.
        # dataset_out[:,:,j] = ((dataset_out[:,:,j] - processing_limits[mom]['min'])/range_i)

        return dataset_out

    def postprocess(self, sample):

        dataset_out = sample.copy()

        for j, mom in enumerate(['px', 'py', 'pz']):
            range_i = self.limits[mom]['max'] - self.limits[mom]['min']
            dataset_out[:, :, j] = (((dataset_out[:, :, j] + 1.) / 2.) * (range_i) + self.limits[mom]['min'])
        # dataset_out[:,:,j] = (((dataset_out[:,:,j]))*(range_i)+ processing_limits[mom]['min'])

        dataset_out[:, :, 2] = np.exp(dataset_out[:, :, 2])
        dataset_out[:, :, 2] = dataset_out[:, :, 2] - 5.

        # for idx in [0,1]:
        # 	dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]<0.)] = -1.*dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]<0.)]**2
        # 	dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]>0.)] = dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]>0.)]**2

        return dataset_out


def preprocess(dataset_i, processing_limits):
    # get_proccessing_limits(dataset, plotting=True)

    dataset_out = dataset_i.copy()

    dataset_out[:, :, 2] = dataset_out[:, :, 2] + 5.
    dataset_out[:, :, 2] = np.log(dataset_out[:, :, 2])

    # processing_limits = get_proccessing_limits(dataset_out)

    # for idx in [0,1]:
    # 	dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]<0.)] = -1.*np.sqrt(np.abs(dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]<0.)]))
    # 	dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]>0.)] = np.sqrt(dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]>0.)])

    for j, mom in enumerate(['px', 'py', 'pz']):
        range_i = processing_limits[mom]['max'] - processing_limits[mom]['min']
        dataset_out[:, :, j] = ((dataset_out[:, :, j] - processing_limits[mom]['min']) / range_i) * 2. - 1.
    # dataset_out[:,:,j] = ((dataset_out[:,:,j] - processing_limits[mom]['min'])/range_i)

    return dataset_out


def postprocess(dataset_i, processing_limits):
    dataset_out = dataset_i.copy()

    # processing_limits = get_proccessing_limits()

    for j, mom in enumerate(['px', 'py', 'pz']):
        range_i = processing_limits[mom]['max'] - processing_limits[mom]['min']
        dataset_out[:, :, j] = (((dataset_out[:, :, j] + 1.) / 2.) * (range_i) + processing_limits[mom]['min'])
    # dataset_out[:,:,j] = (((dataset_out[:,:,j]))*(range_i)+ processing_limits[mom]['min'])

    dataset_out[:, :, 2] = np.exp(dataset_out[:, :, 2])
    dataset_out[:, :, 2] = dataset_out[:, :, 2] - 5.

    # for idx in [0,1]:
    # 	dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]<0.)] = -1.*dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]<0.)]**2
    # 	dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]>0.)] = dataset_out[:,:,idx][np.where(dataset_out[:,:,idx]>0.)]**2

    return dataset_out

class PreProcessor():
    pass

class PostProcessor():
    pass

class MomentaCatPreprocessor(nn.Module, PreProcessor):
    def __init__(self):
        super().__init__()

    def forward(self, sample, direction=1, on=None):
        if direction != 1:
            raise ValueError('MomentaCatPreprocessor is only a preprocessor.')
        sample_2 = {}

        if 'particle_1_PX' in sample:
            momenta = torch.stack([sample['particle_1_PX'], sample['particle_1_PY'],
                                   sample['particle_1_PZ'], sample['particle_2_PX'],
                                   sample['particle_2_PY'], sample['particle_2_PZ'],
                                   sample['particle_3_PX'], sample['particle_3_PY'],
                                   sample['particle_3_PZ']], dim=1)

            momenta = momenta.reshape((len(sample['particle_1_PX']), 3, 3))

            sample_2['momenta'] = momenta

        if 'mother_PX_TRUE' in sample:
            momenta_mother = torch.stack([sample['mother_PX_TRUE'],
                                 sample['mother_PY_TRUE'], sample['mother_PZ_TRUE']], dim=1)
            momenta_mother = momenta_mother.reshape((len(sample['particle_1_PX']), 1, 3))

            sample_2['momenta_mother'] = momenta_mother

        return sample_2


def compute_dalitz_masses_2(vec4, sample, nan_to_num=True, squared=False, _engine='torch'):
    # This is recomputing it twice at least. This is fine as these things are fast but could be cleaned up later on.
    sample_2 = {}
    # sample_2.update(sample)
    for particle_idx, particle in enumerate(['1', '2', '3']):
        for coordinate_idx, coordinate in enumerate(['PX', 'PY', 'PZ']):
            sample_2[f'particle_{particle}_{coordinate}'] = vec4[
                                                                   :, particle_idx, coordinate_idx]

                # # Check
                # if coordinate_idx == 2:
                #     sample_2[f'particle_{particle}_{coordinate}_DECODED']=sample_2['momenta_reconstructed_upp'][
                #                                                        :, particle_idx, coordinate_idx]*0.0

    if _engine == 'np':
        engine = np
    elif _engine == 'torch':
        engine = torch
    else:
        raise RuntimeError("Unknown engine")

    pe_1 = engine.sqrt(
        sample['particle_1_M'] ** 2 + sample_2[f'particle_1_PX'] ** 2 + sample_2[f'particle_1_PY'] ** 2 + sample_2[
            f'particle_1_PZ'] ** 2)
    pe_2 = engine.sqrt(
        sample['particle_2_M'] ** 2 + sample_2[f'particle_2_PX'] ** 2 + sample_2[f'particle_2_PY'] ** 2 + sample_2[
            f'particle_2_PZ'] ** 2)
    pe_3 = engine.sqrt(
        sample['particle_3_M'] ** 2 + sample_2[f'particle_3_PX'] ** 2 + sample_2[f'particle_3_PY'] ** 2 + sample_2[
            f'particle_3_PZ'] ** 2)

    pe = pe_3 + pe_2
    px = sample_2[f'particle_3_PX'] + sample_2[f'particle_2_PX']
    py = sample_2[f'particle_3_PY'] + sample_2[f'particle_2_PY']
    pz = sample_2[f'particle_3_PZ'] + sample_2[f'particle_2_PZ']
    # p_32 = vector.obj(px=px, py=py, pz=pz, E=pe)

    mass_32 = pe ** 2 - px ** 2 - py ** 2 - pz ** 2
    if not squared:
        mass_32 = engine.sqrt(mass_32)

    pe = pe_1 + pe_3
    px = sample_2[f'particle_1_PX'] + sample_2[f'particle_3_PX']
    py = sample_2[f'particle_1_PY'] + sample_2[f'particle_3_PY']
    pz = sample_2[f'particle_1_PZ'] + sample_2[f'particle_3_PZ']
    # p_13 = vector.obj(px=px, py=py, pz=pz, E=pe)

    mass_13 = pe ** 2 - px ** 2 - py ** 2 - pz ** 2
    if not squared:
        mass_13 = engine.sqrt(mass_13)

    if nan_to_num:
        mass_13 = engine.nan_to_num(mass_13)
        mass_32 = engine.nan_to_num(mass_32)

    return mass_13, mass_32

def compute_parent_mass(sample, tag, nan_to_num=False, _engine='torch', reco_var='momenta_reconstructed_upp'):

    if _engine == 'np':
        engine = np
    elif _engine == 'torch':
        engine = torch
    else:
        raise RuntimeError("Unknown engine")

    sample_2 = {}
    sample_2.update(sample)

    for particle_idx, particle in enumerate(['1', '2', '3']):
        for coordinate_idx, coordinate in enumerate(['PX', 'PY', 'PZ']):

            if reco_var in sample.keys():
                sample_2[f'particle_{particle}_{coordinate}_DECODED'] = sample_2[reco_var][
                                                                       :, particle_idx, coordinate_idx]
            if 'momenta_upp' in sample.keys():
                sample_2[f'particle_{particle}_{coordinate}'] = sample_2['momenta_upp'][
                                                                       :, particle_idx, coordinate_idx]

    sample = sample_2

    pe_1 = engine.sqrt(
        sample_2['particle_1_M'] ** 2 + sample[f'particle_1_PX{tag}'] ** 2 + sample[f'particle_1_PY{tag}'] ** 2 + sample[
            f'particle_1_PZ{tag}'] ** 2)
    pe_2 = engine.sqrt(
        sample_2['particle_2_M'] ** 2 + sample[f'particle_2_PX{tag}'] ** 2 + sample[f'particle_2_PY{tag}'] ** 2 + sample[
            f'particle_2_PZ{tag}'] ** 2)
    pe_3 = engine.sqrt(
        sample_2['particle_3_M'] ** 2 + sample[f'particle_3_PX{tag}'] ** 2 + sample[f'particle_3_PY{tag}'] ** 2 + sample[
            f'particle_3_PZ{tag}'] ** 2)

    pe = pe_1 + pe_2 + pe_3
    px = sample[f'particle_1_PX{tag}'] + sample[f'particle_2_PX{tag}'] + sample[f'particle_3_PX{tag}']
    py = sample[f'particle_1_PY{tag}'] + sample[f'particle_2_PY{tag}'] + sample[f'particle_3_PY{tag}']
    pz = sample[f'particle_1_PZ{tag}'] + sample[f'particle_2_PZ{tag}'] + sample[f'particle_3_PZ{tag}']

    B_mass = engine.sqrt(pe ** 2 - px ** 2 - py ** 2 - pz ** 2)
    if tag == '':
        tag = '_DECODED'
    # print("YYY", tag, '%#05.5f'%float(engine.mean(pe)), '%#05.5f'%float(engine.mean(px)), '%#05.5f'%float(engine.mean(py)), '%#05.5f'%float(engine.mean(pz)), '%#05.5f'%float(engine.mean(sample[f'particle_3_PZ{tag}'])))
    return B_mass

def compute_dalitz_masses(sample, tag, nan_to_num=True, squared=False, _engine='torch', reco_var='momenta_reconstructed_upp'):
    # This is recomputing it twice at least. This is fine as these things are fast but could be cleaned up later on.
    sample_2 = {}
    sample_2.update(sample)
    # sample_2.pop('particle_1_PX')
    # sample_2.pop('particle_1_PY')
    # sample_2.pop('particle_1_PZ')
    # sample_2.pop('particle_2_PX')
    # sample_2.pop('particle_2_PY')
    # sample_2.pop('particle_2_PZ')
    # sample_2.pop('particle_3_PX')
    # sample_2.pop('particle_3_PY')
    # sample_2.pop('particle_3_PZ')
    for particle_idx, particle in enumerate(['1', '2', '3']):
        for coordinate_idx, coordinate in enumerate(['PX', 'PY', 'PZ']):

            if reco_var in sample.keys():
                sample_2[f'particle_{particle}_{coordinate}_DECODED'] = sample_2[reco_var][
                                                                       :, particle_idx, coordinate_idx]
            if 'momenta_upp' in sample.keys():
                sample_2[f'particle_{particle}_{coordinate}'] = sample_2['momenta_upp'][
                                                                       :, particle_idx, coordinate_idx]

            # # Check
            # if coordinate_idx == 2:
            #     sample_2[f'particle_{particle}_{coordinate}_DECODED']=sample_2['momenta_reconstructed_upp'][
            #                                                        :, particle_idx, coordinate_idx]*0.0

    if _engine == 'np':
        engine = np
    elif _engine == 'torch':
        engine = torch
    else:
        raise RuntimeError("Unknown engine")

    pe_1 = engine.sqrt(
        sample_2['particle_1_M'] ** 2 + sample_2[f'particle_1_PX{tag}'] ** 2 + sample_2[f'particle_1_PY{tag}'] ** 2 + sample_2[
            f'particle_1_PZ{tag}'] ** 2)
    pe_2 = engine.sqrt(
        sample_2['particle_2_M'] ** 2 + sample_2[f'particle_2_PX{tag}'] ** 2 + sample_2[f'particle_2_PY{tag}'] ** 2 + sample_2[
            f'particle_2_PZ{tag}'] ** 2)
    pe_3 = engine.sqrt(
        sample_2['particle_3_M'] ** 2 + sample_2[f'particle_3_PX{tag}'] ** 2 + sample_2[f'particle_3_PY{tag}'] ** 2 + sample_2[
            f'particle_3_PZ{tag}'] ** 2)

    pe = pe_3 + pe_2
    px = sample_2[f'particle_3_PX{tag}'] + sample_2[f'particle_2_PX{tag}']
    py = sample_2[f'particle_3_PY{tag}'] + sample_2[f'particle_2_PY{tag}']
    pz = sample_2[f'particle_3_PZ{tag}'] + sample_2[f'particle_2_PZ{tag}']
    # p_32 = vector.obj(px=px, py=py, pz=pz, E=pe)

    mass_32 = pe ** 2 - px ** 2 - py ** 2 - pz ** 2
    if not squared:
        mass_32 = engine.sqrt(mass_32)

    pe = pe_1 + pe_3
    px = sample_2[f'particle_1_PX{tag}'] + sample_2[f'particle_3_PX{tag}']
    py = sample_2[f'particle_1_PY{tag}'] + sample_2[f'particle_3_PY{tag}']
    pz = sample_2[f'particle_1_PZ{tag}'] + sample_2[f'particle_3_PZ{tag}']
    # p_13 = vector.obj(px=px, py=py, pz=pz, E=pe)

    mass_13 = pe ** 2 - px ** 2 - py ** 2 - pz ** 2
    if not squared:
        mass_13 = engine.sqrt(mass_13)

    if nan_to_num:
        mass_13 = engine.nan_to_num(mass_13)
        mass_32 = engine.nan_to_num(mass_32)

    return mass_13, mass_32




class OnlineThreeBodyDecayMomentaPreprocessor2(nn.Module, PreProcessor, PostProcessor):
    def __init__(self, estimation_sample=None):
        super(OnlineThreeBodyDecayMomentaPreprocessor2, self).__init__()
        self.min_decay_prods = torch.nn.Parameter(torch.zeros((1, 3,3)))
        self.max_decay_prods = torch.nn.Parameter(torch.zeros((1, 3,3)))
        self.min_mother = torch.nn.Parameter(torch.zeros((1, 1, 3)))
        self.max_mother = torch.nn.Parameter(torch.zeros((1, 1, 3)))

        if estimation_sample is not None:
            self.estimate(estimation_sample)

        # print(torch.min(estimation_sample['particle_3_PZ'], dim=0), torch.min(estimation_sample['mother_PZ_TRUE'], dim=0))
        # print(torch.min(estimation_sample['momenta'], dim=[0,1]), torch.min(estimation_sample['momenta_mother'], dim=0))

    def estimate(self, estimation_sample):
        if type(estimation_sample) is list:
            estimation_sample = tensors_dict_join(estimation_sample)
        self.get_limits_from_samples(estimation_sample['momenta'], estimation_sample['momenta_mother'])

    def forward(self, sample: dict, direction=1, on=None):
        """
        :param data: dict
        :param direction: 1 if forward, -1 if reverse
        :param on: only for postprocessing, 'sampled' for postprocessing sampled, 'reconstructed' for postprocessing
                   reconstructed output
        :return:
        """

        assert type(sample) is dict

        if direction == 1:
            momenta_preprocessed, momenta_mother_preprocessed = self.preprocess(sample['momenta'], sample['momenta_mother'])
            return {
                'momenta_pp': momenta_preprocessed,
                'momenta_mother_pp': momenta_mother_preprocessed,
            }
        elif direction == -1:
            if on is None:
                key_momenta = 'momenta'
                key_momenta_mother = 'momenta_mother'
                key_upp_momenta = 'momenta_upp'
                key_upp_momenta_mother = 'momenta_mother_upp'
            elif on == 'sampled':
                key_momenta = 'momenta_pp_sampled'
                key_momenta_mother = None
                key_upp_momenta = 'momenta_sampled_upp'
                key_upp_momenta_mother = None
            elif on == 'reconstructed':
                key_momenta = 'momenta_pp_reconstructed'
                key_momenta_mother = None
                key_upp_momenta = 'momenta_reconstructed_upp'
                key_upp_momenta_mother = None
            else:
                raise ValueError('Illegal value of on')

            # print(sample.keys())
            momenta_unpreprocessed, momenta_mother_unpreprocessed =  self.postprocess(sample[key_momenta], sample[key_momenta_mother] if key_momenta_mother is not None else None)
            # # Check
            # momenta_unpreprocessed, momenta_mother_unpreprocessed = sample[key_momenta], sample[
            #     key_momenta_mother] if key_momenta_mother is not None else None

            result_dict = {
                key_upp_momenta: momenta_unpreprocessed
            }
            if momenta_mother_unpreprocessed is not None:
                result_dict[key_upp_momenta_mother] = momenta_mother_unpreprocessed
            return result_dict
        else:
            raise ValueError('Direction value invalid.')

    def get_limits_from_samples(self, sample, sample_mother):
        def min_func(x):
            x,_ = torch.min(x, dim=0, keepdim=True)
            x = torch.where(x<0, x * 1.1, x * 0.9)
            return x
        def max_func(x):
            x,_ = torch.max(x, dim=0, keepdim=True)
            x = torch.where(x<0, x * 0.9, x * 1.1)
            return x

        assert len(sample.shape) == 3
        assert sample.shape[1] == 3
        assert sample.shape[2] == 3

        assert len(sample_mother.shape) == 3
        assert sample_mother.shape[1] == 1
        assert sample_mother.shape[2] == 3

        sample_copy = sample * 1.0
        sample_copy[:, :, 2] = torch.log(sample_copy[:, :, 2] + 5.0)
        # From [B, 3, 3] to [3, 3]
        self.min_decay_prods.data = min_func(sample_copy)
        self.max_decay_prods.data = max_func(sample_copy)

        sample_mother_copy = sample_mother * 1.0
        sample_mother_copy[:, :, 2] = torch.log(sample_mother_copy[:, :, 2] + 5)
        # From [B, 3, 3] to [3, 3]
        self.min_mother.data = min_func(sample_mother_copy)
        self.max_mother.data = max_func(sample_mother_copy)

    def preprocess(self, sample, sample_mother):
        return sample, sample_mother

        min_decay_prods = self.min_decay_prods
        max_decay_prods = self.max_decay_prods

        min_mother = self.min_mother
        max_mother = self.max_mother

        if sample.device != self.min_decay_prods.device:
            min_decay_prods = self.min_decay_prods.to(sample.device)
            max_decay_prods = self.max_decay_prods.to(sample.device)

            min_mother = self.min_mother.to(sample.device)
            max_mother = self.max_mother.to(sample.device)

        if sample is not None:
            sample = sample * 1.0
            sample[:, :, 2] = torch.log(sample[:, :, 2] + 5)
            sample = ((sample - min_decay_prods) / (max_decay_prods - min_decay_prods)) * 2.0 - 1.0

        if sample_mother is not None:
            sample_mother = sample_mother * 1.0
            sample_mother[:, :, 2] = torch.log(sample_mother[:, :, 2] + 5)
            sample_mother = ((sample_mother - min_mother) / (max_mother - min_mother)) * 2.0 - 1.0

        return sample, sample_mother

    def postprocess(self, sample, sample_mother):
        return sample, sample_mother

        min_decay_prods = self.min_decay_prods
        max_decay_prods = self.max_decay_prods

        min_mother = self.min_mother
        max_mother = self.max_mother

        if sample.device != self.min_decay_prods.device:
            min_decay_prods = self.min_decay_prods.to(sample.device)
            max_decay_prods = self.max_decay_prods.to(sample.device)

            min_mother = self.min_mother.to(sample.device)
            max_mother = self.max_mother.to(sample.device)

        if sample is not None:
            sample = sample * 1
            sample = (sample + 1) * 0.5 * (max_decay_prods - min_decay_prods) + min_decay_prods
            sample[:, :, 2] = torch.exp(sample[:, :, 2]) - 5
            sample = torch.nan_to_num(sample)

        if sample_mother is not None:
            sample_mother = sample_mother * 1
            sample_mother = (sample_mother + 1) * 0.5 * (max_mother - min_mother) + min_mother
            sample_mother[:, :, 2] = torch.exp(sample_mother[:, :, 2]) - 5
            sample_mother = torch.nan_to_num(sample_mother)

        return sample, sample_mother

class OnlineThreeBodyDecayMomentaPreprocessor(nn.Module, PreProcessor, PostProcessor):
    def __init__(self, estimation_sample=None):
        super(OnlineThreeBodyDecayMomentaPreprocessor, self).__init__()
        self.min_decay_prods = torch.nn.Parameter(torch.zeros((1, 3,3)), requires_grad=False)
        self.max_decay_prods = torch.nn.Parameter(torch.zeros((1, 3,3)), requires_grad=False)
        self.min_mother = torch.nn.Parameter(torch.zeros((1, 1, 3)), requires_grad=False)
        self.max_mother = torch.nn.Parameter(torch.zeros((1, 1, 3)), requires_grad=False)

        if estimation_sample is not None:
            self.estimate(estimation_sample)

        # print(torch.min(estimation_sample['particle_3_PZ'], dim=0), torch.min(estimation_sample['mother_PZ_TRUE'], dim=0))
        # print(torch.min(estimation_sample['momenta'], dim=[0,1]), torch.min(estimation_sample['momenta_mother'], dim=0))

    def estimate(self, estimation_sample):
        if type(estimation_sample) is list:
            estimation_sample = tensors_dict_join(estimation_sample)
        self.get_limits_from_samples(estimation_sample['momenta'], estimation_sample['momenta_mother'])

    def forward(self, sample: dict, direction=1, on=None):
        """
        :param data: dict
        :param direction: 1 if forward, -1 if reverse
        :param on: only for postprocessing, 'sampled' for postprocessing sampled, 'reconstructed' for postprocessing
                   reconstructed output
        :return:
        """

        assert type(sample) is dict

        if direction == 1:

            # # Check
            # momenta_preprocessed, momenta_mother_preprocessed = sample['momenta'], sample['momenta_mother']
            # return {
            #     'momenta_pp': momenta_preprocessed,
            #     'momenta_mother_pp': momenta_mother_preprocessed,
            # }

            momenta_preprocessed, momenta_mother_preprocessed = self.preprocess(sample['momenta'], sample['momenta_mother'])
            return {
                'momenta_pp': momenta_preprocessed,
                'momenta_mother_pp': momenta_mother_preprocessed,
            }
        elif direction == -1:
            if on is None:
                key_momenta = 'momenta_pp'
                key_momenta_mother = 'momenta_mother_pp'
                key_upp_momenta = 'momenta_upp'
                key_upp_momenta_mother = 'momenta_mother_upp'
            elif on == 'sampled':
                key_momenta = 'momenta_sampled'
                key_momenta_mother = None
                key_upp_momenta = 'momenta_sampled_upp'
                key_upp_momenta_mother = None
            elif on == 'reconstructed':
                key_momenta = 'momenta_reconstructed'
                key_momenta_mother = None
                key_upp_momenta = 'momenta_reconstructed_upp'
                key_upp_momenta_mother = None
            else:
                raise ValueError('Illegal value of on')

            momenta_unpreprocessed, momenta_mother_unpreprocessed =  self.postprocess(sample[key_momenta], sample[key_momenta_mother] if key_momenta_mother is not None else None)
            # # Check
            # momenta_unpreprocessed, momenta_mother_unpreprocessed = sample[key_momenta], sample[
            #     key_momenta_mother] if key_momenta_mother is not None else None

            result_dict = {
                key_upp_momenta: momenta_unpreprocessed
            }
            if momenta_mother_unpreprocessed is not None:
                result_dict[key_upp_momenta_mother] = momenta_mother_unpreprocessed
            return result_dict
        else:
            raise ValueError('Direction value invalid.')

    def get_limits_from_samples(self, sample, sample_mother):
        def min_func(x):
            x,_ = torch.min(x, dim=0, keepdim=True)
            x = torch.where(x<0, x * 1.1, x * 0.9)
            return x
        def max_func(x):
            x,_ = torch.max(x, dim=0, keepdim=True)
            x = torch.where(x<0, x * 0.9, x * 1.1)
            return x

        assert len(sample.shape) == 3
        assert sample.shape[1] == 3
        assert sample.shape[2] == 3

        assert len(sample_mother.shape) == 3
        assert sample_mother.shape[1] == 1
        assert sample_mother.shape[2] == 3

        sample_copy = sample * 1.0
        sample_copy[:, :, 2] = torch.log(sample_copy[:, :, 2] + 5.0)
        # From [B, 3, 3] to [3, 3]
        self.min_decay_prods.data = min_func(sample_copy)
        self.max_decay_prods.data = max_func(sample_copy)

        sample_mother_copy = sample_mother * 1.0
        sample_mother_copy[:, :, 2] = torch.log(sample_mother_copy[:, :, 2] + 5)
        # From [B, 3, 3] to [3, 3]
        self.min_mother.data = min_func(sample_mother_copy)
        self.max_mother.data = max_func(sample_mother_copy)

    def preprocess(self, sample, sample_mother):
        min_decay_prods = self.min_decay_prods
        max_decay_prods = self.max_decay_prods

        min_mother = self.min_mother
        max_mother = self.max_mother

        if sample.device != self.min_decay_prods.device:
            min_decay_prods = self.min_decay_prods.to(sample.device)
            max_decay_prods = self.max_decay_prods.to(sample.device)

            min_mother = self.min_mother.to(sample.device)
            max_mother = self.max_mother.to(sample.device)

        if sample is not None:
            sample = sample * 1.0
            sample[:, :, 2] = torch.log(sample[:, :, 2] + 5)
            # sample[:, :, 2] = (sample[:, :, 2] + 5)
            sample = ((sample - min_decay_prods) / (max_decay_prods - min_decay_prods)) * 2.0 - 1.0

        if sample_mother is not None:
            sample_mother = sample_mother * 1.0
            # sample_mother[:, :, 2] = (sample_mother[:, :, 2] + 5)
            sample_mother[:, :, 2] = torch.log(sample_mother[:, :, 2] + 5)
            sample_mother = ((sample_mother - min_mother) / (max_mother - min_mother)) * 2.0 - 1.0

        return sample, sample_mother

    def postprocess(self, sample, sample_mother):
        min_decay_prods = self.min_decay_prods
        max_decay_prods = self.max_decay_prods

        min_mother = self.min_mother
        max_mother = self.max_mother

        if sample.device != self.min_decay_prods.device:
            min_decay_prods = self.min_decay_prods.to(sample.device)
            max_decay_prods = self.max_decay_prods.to(sample.device)

            min_mother = self.min_mother.to(sample.device)
            max_mother = self.max_mother.to(sample.device)

        if sample is not None:
            sample = sample * 1
            sample = (sample + 1) * 0.5 * (max_decay_prods - min_decay_prods) + min_decay_prods
            sample[:, :, 2] = torch.exp(sample[:, :, 2]) - 5
            # sample[:, :, 2] = (sample[:, :, 2]) - 5
            sample = torch.nan_to_num(sample)

        if sample_mother is not None:
            sample_mother = sample_mother * 1
            sample_mother = (sample_mother + 1) * 0.5 * (max_mother - min_mother) + min_mother
            sample_mother[:, :, 2] = torch.exp(sample_mother[:, :, 2]) - 5
            # sample_mother[:, :, 2] = (sample_mother[:, :, 2]) - 5
            sample_mother = torch.nan_to_num(sample_mother)

        return sample, sample_mother

class ThreeBodyDecayDataset(LightningDataModule):
    def __init__(
            self,
            data_path: dict,
            legacy_rotation: bool,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = True,
            train_test_split: float = 0.8,
            split_seed = -1,
            use_root_reader = False,
            load_together=200000,
            shuffle=False,
            engine = 'torch',
            **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_test_split = train_test_split
        self.preprocessors = []
        self.engine = engine

        self.legacy_rotation = legacy_rotation
        self.split_seed = split_seed
        self.use_root_reader = use_root_reader
        self.load_together = load_together
        self.shuffle = shuffle # Only set to true if the whole dataset can be loaded into memory

        self._train_loader = None
        self._val_loader = None

    def get_data_simple(self, file):
        print("Attempting eta load")
        file = uproot.open(file)["DecayTree"]
        keys = file.keys()

        results = file.arrays(keys, library="np")

        if self.legacy_rotation:
            print("Legacy rotation")

            results = pd.DataFrame.from_dict(results)
            mother_P = np.sqrt(results.mother_PX ** 2 + results.mother_PY ** 2 + results.mother_PZ ** 2)
            mother_P_true = np.sqrt(
                results.mother_PX_TRUE ** 2 + results.mother_PY_TRUE ** 2 + results.mother_PZ_TRUE ** 2)

            shape = np.shape(results)
            training_parameters = {}

            pe_1 = np.sqrt(
                results.particle_1_M ** 2 + results.particle_1_PX ** 2 + results.particle_1_PY ** 2 + results.particle_1_PZ ** 2)
            pe_2 = np.sqrt(
                results.particle_2_M ** 2 + results.particle_2_PX ** 2 + results.particle_2_PY ** 2 + results.particle_2_PZ ** 2)
            pe_3 = np.sqrt(
                results.particle_3_M ** 2 + results.particle_3_PX ** 2 + results.particle_3_PY ** 2 + results.particle_3_PZ ** 2)

            pe = pe_1 + pe_2 + pe_3
            px = results.particle_1_PX + results.particle_2_PX + results.particle_3_PX
            py = results.particle_1_PY + results.particle_2_PY + results.particle_3_PY
            pz = results.particle_1_PZ + results.particle_2_PZ + results.particle_3_PZ

            B = vector.obj(px=px, py=py, pz=pz, E=pe)

            Bmass = np.sqrt(B.E ** 2 - B.px ** 2 - B.py ** 2 - B.pz ** 2)

            B_phi = np.arctan2(B.py, B.px)
            training_parameters["B_phi"] = B_phi
            B_theta = np.arctan2(B.px, B.pz)
            training_parameters["B_theta"] = B_theta
            B_p = np.sqrt(B.px ** 2 + B.py ** 2 + B.pz ** 2)
            training_parameters["B_P"] = B_p

            B_vec = np.swapaxes(norm(np.asarray([B.px, B.py, B.pz])), 0, 1)

            all_pz = np.swapaxes(
                norm(
                    np.asarray([np.zeros((np.shape(Bmass))), np.zeros((np.shape(Bmass))), np.ones((np.shape(Bmass)))])),
                0,
                1)

            ROT_matrix = rotation_matrix_from_vectors_vectorised(B_vec, all_pz)

            P1 = vector.obj(px=results.particle_1_PX, py=results.particle_1_PY, pz=results.particle_1_PZ,
                            E=results.particle_1_E)
            P2 = vector.obj(px=results.particle_2_PX, py=results.particle_2_PY, pz=results.particle_2_PZ,
                            E=results.particle_2_E)
            P3 = vector.obj(px=results.particle_3_PX, py=results.particle_3_PY, pz=results.particle_3_PZ,
                            E=results.particle_3_E)
            PM = vector.obj(px=results.mother_PX_TRUE, py=results.mother_PY_TRUE, pz=results.mother_PZ_TRUE,
                            E=results.mother_E_TRUE)

            P1_vec = [P1.px, P1.py, P1.pz]
            P2_vec = [P2.px, P2.py, P2.pz]
            P3_vec = [P3.px, P3.py, P3.pz]
            PM_vec = [PM.px, PM.py, PM.pz]

            P1_vec_ROT = rot_vectorised(P1_vec, ROT_matrix)
            P2_vec_ROT = rot_vectorised(P2_vec, ROT_matrix)
            P3_vec_ROT = rot_vectorised(P3_vec, ROT_matrix)
            PM_vec_ROT = rot_vectorised(PM_vec, ROT_matrix)

            E = np.sqrt(results.particle_1_M ** 2 + P1_vec_ROT[0] ** 2 + P1_vec_ROT[1] ** 2 + P1_vec_ROT[2] ** 2)
            P1_ROT = vector.obj(px=P1_vec_ROT[0], py=P1_vec_ROT[1], pz=P1_vec_ROT[2], E=E)

            E = np.sqrt(results.particle_2_M ** 2 + P2_vec_ROT[0] ** 2 + P2_vec_ROT[1] ** 2 + P2_vec_ROT[2] ** 2)
            P2_ROT = vector.obj(px=P2_vec_ROT[0], py=P2_vec_ROT[1], pz=P2_vec_ROT[2], E=E)

            E = np.sqrt(results.particle_3_M ** 2 + P3_vec_ROT[0] ** 2 + P3_vec_ROT[1] ** 2 + P3_vec_ROT[2] ** 2)
            P3_ROT = vector.obj(px=P3_vec_ROT[0], py=P3_vec_ROT[1], pz=P3_vec_ROT[2], E=E)

            E = np.sqrt(results.particle_3_M ** 2 + PM_vec_ROT[0] ** 2 + PM_vec_ROT[1] ** 2 + PM_vec_ROT[2] ** 2)
            PM_ROT = vector.obj(px=PM_vec_ROT[0], py=PM_vec_ROT[1], pz=PM_vec_ROT[2], E=E)

            training_parameters["P1_px"] = P1_ROT.px
            training_parameters["P1_py"] = P1_ROT.py
            training_parameters["P1_pz"] = P1_ROT.pz
            training_parameters["P2_px"] = P2_ROT.px
            training_parameters["P2_py"] = P2_ROT.py
            training_parameters["P2_pz"] = P2_ROT.pz
            training_parameters["P3_px"] = P3_ROT.px
            training_parameters["P3_py"] = P3_ROT.py
            training_parameters["P3_pz"] = P3_ROT.pz
            training_parameters["PM_px"] = PM_ROT.px
            training_parameters["PM_py"] = PM_ROT.py
            training_parameters["PM_pz"] = PM_ROT.pz
        else:
            print("No legacy rotation")

            training_parameters = {}

            training_parameters["P1_px"] = results['particle_1_PX']
            training_parameters["P1_py"] = results['particle_1_PY']
            training_parameters["P1_pz"] = results['particle_1_PZ']
            training_parameters["P2_px"] = results['particle_2_PX']
            training_parameters["P2_py"] = results['particle_2_PY']
            training_parameters["P2_pz"] = results['particle_2_PZ']
            training_parameters["P3_px"] = results['particle_3_PX']
            training_parameters["P3_py"] = results['particle_3_PY']
            training_parameters["P3_pz"] = results['particle_3_PZ']
            training_parameters["PM_px"] = results['mother_PX_TRUE']
            training_parameters["PM_py"] = results['mother_PY_TRUE']
            training_parameters["PM_pz"] = results['mother_PZ_TRUE']

        reshaped = np.asarray(
            [[training_parameters["P1_px"], training_parameters["P1_py"], training_parameters["P1_pz"]],
             [training_parameters["P2_px"], training_parameters["P2_py"], training_parameters["P2_pz"]],
             [training_parameters["P3_px"], training_parameters["P3_py"], training_parameters["P3_pz"]]])
        reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
        momenta = np.swapaxes(np.asarray(reshaped), 0, 2)

        reshaped = np.asarray(
            [[training_parameters["PM_px"], training_parameters["PM_py"], training_parameters["PM_pz"]]])
        reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
        momenta_mother = np.swapaxes(np.asarray(reshaped), 0, 2)
        print("Eta loaded")

        if self.legacy_rotation:
            results_dict = {
                'mass_mother': results.mother_M.to_numpy(),
                'momenta': momenta,
                'momenta_mother': momenta_mother,
                'pdgid_mother': results.mother_PID.to_numpy(),
                'pdgid_particle_1': results.particle_1_PID.to_numpy(),
                'pdgid_particle_2': results.particle_2_PID.to_numpy(),
                'pdgid_particle_3': results.particle_3_PID.to_numpy(),
                'mass_particle_1': results.particle_1_M_TRUE.to_numpy(),
                'mass_particle_2': results.particle_2_M_TRUE.to_numpy(),
                'mass_particle_3': results.particle_3_M_TRUE.to_numpy(),
            }
        else:
            results_dict = {
                'mass_mother': results['mother_M'],
                'momenta': momenta,
                'momenta_mother': momenta_mother,
                'pdgid_mother': results['mother_PID'],
                'pdgid_particle_1': results['particle_1_PID'],
                'pdgid_particle_2': results['particle_2_PID'],
                'pdgid_particle_3': results['particle_3_PID'],
                'mass_particle_1': results['particle_1_M_TRUE'],
                'mass_particle_2': results['particle_2_M_TRUE'],
                'mass_particle_3': results['particle_3_M_TRUE'],
            }

        return results_dict


    def split(self, data, split_at):
        return data[:split_at], data[split_at:]

    def _get_loader(self, data_param, batch_size):
        if type(data_param) is str:
            p = data_param
            block_size = 10000
            num_blocks = -1
        elif type(data_param) is dict:
            p = data_param['path']
            num_blocks = data_param['num_blocks']
            block_size = data_param['block_size']
        else:
            raise ValueError('The data param is of unknown type. It has to be either str showing the path of the'
                             'data file or a dict containing element data_path containing the path of the data file'
                             'and num_blocks and block size.')

        return RootBlockShuffledSubsetDataLoader(p,
                                                 block_size=block_size,
                                                 num_blocks=num_blocks,
                                                 batch_size=batch_size,
                                                 engine=self.engine)


    def setup(self, stage: Optional[str] = None) -> None:

        # full_dataset = Dataset({
        #     'momenta': torch.Tensor(momenta),
        #     'momenta_mother': torch.tensor(momenta_mother)}
        # )
        if not self.use_root_reader:
            raise NotImplementedError('Adapt code to to use RootTensorDataset')
            results_dict = self.get_data_simple(self.data_path)
            full_dataset = DictTensorDataset({k:torch.Tensor(v) for k,v in results_dict.items()})
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size

            generator = None if self.split_seed == -1 else torch.Generator().manual_seed(self.split_seed)
            self.dataset_train, self.dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size],
                                                                    generator=generator)
        else:
            # self.dataset_train = RootTensorDataset(self.data_path['train'], 'DecayTree', cache_size=-1)
            # self.dataset_test = RootTensorDataset(self.data_path['validate'], 'DecayTree', cache_size=-1)
            # train_size = int(0.8 * len(full_dataset))
            # test_size = len(full_dataset) - train_size
            self._train_loader = None

            self._val_loader = None

            # generator = None if self.split_seed == -1 else torch.Generator().manual_seed(self.split_seed)
            # self.dataset_train, self.dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size],
            #                                                                       generator=generator)

    def train_dataloader(self):
        print("train_dataloader() getting called!")

        if self._train_loader is None:
            self._train_loader = self._get_loader(self.data_path['train'], self.train_batch_size)
            self._train_loader.wait_to_load('Reading train data: ')
        # self._train_loader.prepare_next_epoch()
        return self._train_loader

    def check_val_loader(self):
        if self._val_loader is None:
            self._val_loader = self._get_loader(self.data_path['validate'], self.val_batch_size)
            self._val_loader.wait_to_load('Reading val data: ')


    def val_dataloader(self):
        self.check_val_loader()
        return self._val_loader


    def test_dataloader(self):
        self.check_val_loader()
        return self._val_loader

    def predict_dataloader(self):
        self.check_val_loader()
        return self._val_loader

    def exit(self):
        if self._train_loader is not None:
            self._train_loader.exit()
        if self._val_loader is not None:
            self._val_loader.exit()
