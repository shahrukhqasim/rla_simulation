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
from torch.utils.data import DataLoader, TensorDataset
from torch import nn


class DatasetAndLabels():
    def __init__(self, name, train_test_split=0.8, randomise=True):
        self.name = name
        self.dataset = {}
        self.dataset_pp = {}
        self.dataset_classes = {}
        self.data_idx = 0
        self.label_idx = 0
        self.dims = 0
        self.train_test_split = train_test_split
        self.randomise = randomise

    def add_data(self, data_string, dataset_i):
        if not isinstance(dataset_i, np.ndarray):
            raise Exception("Dataset object must be np.array")
        if self.data_idx != 0 and self.dims != np.shape(dataset_i)[0]:
            raise Exception(
                f"Dataset object has wrong length in first dimension, {np.shape(dataset_i)}, length expected {self.dims}")

        if self.data_idx == 0:
            self.dims = np.shape(dataset_i)[0]
            self.idx = np.arange(0, self.dims)
            split_at = int(self.train_test_split * self.dims)
            if self.randomise:
                rand_idx = self.idx.copy()
                np.random.shuffle(rand_idx)
                self.training_idxes = rand_idx[:split_at]
                self.testing_idxes = rand_idx[split_at:]
            else:
                self.training_idxes = rand_idx[:split_at]
                self.testing_idxes = rand_idx[split_at:]
            training_labels = np.concatenate((np.ones(split_at), np.zeros(self.dims - split_at)))
            training_idxes = np.concatenate((self.training_idxes, self.testing_idxes))
            order_idxs = training_idxes.argsort()
            self.training_labels = training_labels[order_idxs]

        elif self.dims != np.shape(dataset_i)[0]:
            raise Exception(
                f"Datset object has wrong length in first dimension, {np.shape(dataset_i)}, length expected {self.dims}")

        print(f"Adding dataset {data_string} (shape {np.shape(dataset_i)}) to {self.name}")

        self.dataset[data_string] = dataset_i
        self.dataset_pp[data_string] = dataset_i
        self.dataset_classes[data_string] = "dataset"
        self.data_idx += 1

    def add_label(self, label_string, label_i):

        if self.dims == 0:
            raise Exception("use add_data first")
        if not isinstance(label_i, np.ndarray):
            raise Exception("Label object must be np.array")
        if self.dims != np.shape(label_i)[0]:
            raise Exception(
                f"Label object has wrong length in first dimension, {np.shape(label_i)}, length expected {self.dims}")

        print(f"Adding label {label_string} (shape {np.shape(label_i)}) to {self.name}")

        self.dataset[label_string] = label_i
        self.dataset_pp[label_string] = label_i
        self.dataset_classes[label_string] = "label"
        self.label_idx += 1

    def get_keys(self, labels=None):

        if labels == None:
            return list(self.dataset.keys())
        if labels:
            out = []
            for key in list(self.dataset.keys()):
                if self.dataset_classes[key] == 'label': out.append(key)
            return out
        else:
            out = []
            for key in list(self.dataset.keys()):
                if self.dataset_classes[key] == 'dataset': out.append(key)
            return out

    def get_data(self, mode=None, preprocessed=False):
        if mode == 'train':
            where = np.where(self.training_labels == 1.)
        elif mode == 'test':
            where = np.where(self.training_labels != 1.)
        else:
            where = self.idx
        out = {}
        for key in list(self.dataset.keys()):
            if preprocessed:
                out[key] = self.dataset_pp[key][where]
            else:
                out[key] = self.dataset[key][where]
        return out

    def get_data_i(self, key, mode=None, preprocessed=False):
        if mode == 'train':
            where = np.where(self.training_labels == 1.)
        elif mode == 'test':
            where = np.where(self.training_labels != 1.)
        else:
            where = self.idx
        if preprocessed:
            return self.dataset_pp[key][where]
        else:
            return self.dataset[key][where]

    def update_preprocess_i(self, key, new_array):
        self.dataset_pp[key] = new_array

    def shuffle(self):
        shuffle_idx = self.idx.copy()
        np.random.shuffle(shuffle_idx)
        self.shuffled_array = self.idx[shuffle_idx]

        self.training_labels = self.training_labels[shuffle_idx]
        for key in list(self.dataset.keys()):
            self.dataset[key] = self.dataset[key][shuffle_idx]
            self.dataset_pp[key] = self.dataset_pp[key][shuffle_idx]

    def permutate_particle_number(self):

        shuffle_idx = self.idx.copy()

        a = np.array([0, 1, 2])
        perms = np.empty((0, np.shape(a)[0]))

        for perm in set(permutations(a)):
            perm = np.expand_dims(np.asarray(perm), 0).astype(int)
            perms = np.append(perms, perm, axis=0)
        n_groups = np.shape(perms)[0]

        print(perms, n_groups)
        choice = np.random.choice(np.arange(n_groups), size=np.shape(shuffle_idx)[0], replace=True)

        for key in list(self.dataset.keys()):

            print('permutate_particle_number:', key)
            # print(np.shape(self.dataset[key]))
            if np.shape(self.dataset[key])[1] == 3:
                for n_group in range(n_groups):
                    perm = perms[n_group]
                    where = np.where(choice == n_group)

                    in_array_where = self.dataset[key][where]
                    in_array_where_permed = in_array_where.copy()

                    in_array_where_permed[:, 0] = in_array_where[:, int(perm[0])]
                    in_array_where_permed[:, 1] = in_array_where[:, int(perm[1])]
                    in_array_where_permed[:, 2] = in_array_where[:, int(perm[2])]

                    self.dataset[key][where] = in_array_where_permed

                    in_array_where = self.dataset_pp[key][where]
                    in_array_where_permed = in_array_where.copy()

                    in_array_where_permed[:, 0] = in_array_where[:, int(perm[0])]
                    in_array_where_permed[:, 1] = in_array_where[:, int(perm[1])]
                    in_array_where_permed[:, 2] = in_array_where[:, int(perm[2])]

                    self.dataset_pp[key][where] = in_array_where_permed

    def reset_indexes(self):
        reset_idxs = self.shuffled_array.argsort()

        self.training_labels = self.training_labels[reset_idxs]
        for key in list(self.dataset.keys()):
            self.dataset[key] = self.dataset[key][reset_idxs]
            self.dataset_pp[key] = self.dataset_pp[key][reset_idxs]


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


(80000, 1, 3)


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


def get_data(file):
    file = uproot.open(file)["DecayTree"]
    keys = file.keys()
    results = file.arrays(keys, library="np")
    results = pd.DataFrame.from_dict(results)

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

    B_pt = np.sqrt(B.px ** 2 + B.py ** 2)
    training_parameters["B_pt"] = B_pt
    B_phi = np.arctan2(B.py, B.px)
    training_parameters["B_phi"] = B_phi
    training_parameters["B_px"] = B.px
    training_parameters["B_py"] = B.py
    training_parameters["B_pz"] = B.pz
    B_p = np.sqrt(B.px ** 2 + B.py ** 2 + B.pz ** 2)
    training_parameters["B_P"] = B_p

    P1 = vector.obj(px=results.particle_1_PX, py=results.particle_1_PY, pz=results.particle_1_PZ,
                    E=results.particle_1_E)
    P2 = vector.obj(px=results.particle_2_PX, py=results.particle_2_PY, pz=results.particle_2_PZ,
                    E=results.particle_2_E)
    P3 = vector.obj(px=results.particle_3_PX, py=results.particle_3_PY, pz=results.particle_3_PZ,
                    E=results.particle_3_E)

    P1_CoM = P1.boost_beta3(-B.to_beta3())
    P2_CoM = P2.boost_beta3(-B.to_beta3())
    P3_CoM = P3.boost_beta3(-B.to_beta3())

    P1_CoM_vec = np.swapaxes(norm(np.asarray([P1_CoM.px, P1_CoM.py, P1_CoM.pz])), 0, 1)
    P2_CoM_vec = np.swapaxes(norm(np.asarray([P2_CoM.px, P2_CoM.py, P2_CoM.pz])), 0, 1)
    P3_CoM_vec = np.swapaxes(norm(np.asarray([P3_CoM.px, P3_CoM.py, P3_CoM.pz])), 0, 1)
    all_pz = np.swapaxes(
        norm(np.asarray([np.zeros((np.shape(Bmass))), np.zeros((np.shape(Bmass))), np.ones((np.shape(Bmass)))])), 0, 1)
    ROT_matrix_CoM = rotation_matrix_from_vectors_vectorised(P1_CoM_vec, all_pz)

    P1_CoM_vec = [P1_CoM.px, P1_CoM.py, P1_CoM.pz]
    P2_CoM_vec = [P2_CoM.px, P2_CoM.py, P2_CoM.pz]
    P3_CoM_vec = [P3_CoM.px, P3_CoM.py, P3_CoM.pz]
    P1_CoM_vec_ROT = rot_vectorised(P1_CoM_vec, ROT_matrix_CoM)
    P2_CoM_vec_ROT = rot_vectorised(P2_CoM_vec, ROT_matrix_CoM)
    P3_CoM_vec_ROT = rot_vectorised(P3_CoM_vec, ROT_matrix_CoM)

    E = np.sqrt(results.particle_1_M ** 2 + P1_CoM_vec_ROT[0] ** 2 + P1_CoM_vec_ROT[1] ** 2 + P1_CoM_vec_ROT[2] ** 2)
    P1_CoM = vector.obj(px=P1_CoM_vec_ROT[0], py=P1_CoM_vec_ROT[1], pz=P1_CoM_vec_ROT[2], E=E)

    E = np.sqrt(results.particle_2_M ** 2 + P2_CoM_vec_ROT[0] ** 2 + P2_CoM_vec_ROT[1] ** 2 + P2_CoM_vec_ROT[2] ** 2)
    P2_CoM = vector.obj(px=P2_CoM_vec_ROT[0], py=P2_CoM_vec_ROT[1], pz=P2_CoM_vec_ROT[2], E=E)

    E = np.sqrt(results.particle_3_M ** 2 + P3_CoM_vec_ROT[0] ** 2 + P3_CoM_vec_ROT[1] ** 2 + P3_CoM_vec_ROT[2] ** 2)
    P3_CoM = vector.obj(px=P3_CoM_vec_ROT[0], py=P3_CoM_vec_ROT[1], pz=P3_CoM_vec_ROT[2], E=E)

    P1_theta = np.arctan2(P1_CoM.py, P1_CoM.px)
    P1_phi = np.arctan2(P1_CoM.px, P1_CoM.pz)
    P1_p = np.sqrt(P1_CoM.px ** 2 + P1_CoM.py ** 2 + P1_CoM.pz ** 2)

    P2_theta = np.arctan2(P2_CoM.py, P2_CoM.px)
    P2_phi = np.arctan2(P2_CoM.px, P2_CoM.pz)
    P2_p = np.sqrt(P2_CoM.px ** 2 + P2_CoM.py ** 2 + P2_CoM.pz ** 2)

    P3_theta = np.arctan2(P3_CoM.py, P3_CoM.px)
    P3_phi = np.arctan2(P3_CoM.px, P3_CoM.pz)
    P3_p = np.sqrt(P3_CoM.px ** 2 + P3_CoM.py ** 2 + P3_CoM.pz ** 2)

    training_parameters["phi_P2"] = P2_phi  # -P1_phi
    training_parameters["phi_P3"] = P3_phi  # -P1_phi

    training_parameters["theta_P2"] = P2_theta  # -P1_theta
    training_parameters["theta_P3"] = P3_theta  # -P1_theta

    training_parameters["mom_CoM_P1"] = P1_p
    training_parameters["mom_CoM_P2"] = P2_p
    training_parameters["mom_CoM_P3"] = P3_p

    # training_parameters["P1_p"] = P1_p
    # training_parameters["P2_px"] = P2_CoM_vec_ROT[0]
    # training_parameters["P2_py"] = P2_CoM_vec_ROT[1]
    # training_parameters["P2_pz"] = P2_CoM_vec_ROT[2]
    # training_parameters["P3_px"] = P3_CoM_vec_ROT[0]
    # training_parameters["P3_py"] = P3_CoM_vec_ROT[1]
    # training_parameters["P3_pz"] = P3_CoM_vec_ROT[2]

    training_samples = DatasetAndLabels("dataset")

    # reshaped = [training_parameters["B_pt"],training_parameters["B_phi"],training_parameters["B_px"],training_parameters["B_py"],training_parameters["B_pz"],training_parameters["B_P"]]
    reshaped = [training_parameters["B_pt"], training_parameters["B_phi"], training_parameters["B_pz"]]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
    training_samples.add_data("B_properties", reshaped)

    reshaped = [training_parameters["phi_P2"], training_parameters["theta_P2"], training_parameters["phi_P3"],
                training_parameters["theta_P3"]]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
    training_samples.add_data("CoM_angles", reshaped)

    # reshaped = [training_parameters["P1_p"],training_parameters["P2_px"],training_parameters["P2_py"],training_parameters["P2_pz"],training_parameters["P3_px"],training_parameters["P3_py"],training_parameters["P3_pz"]]
    # reshaped = np.swapaxes(np.asarray(reshaped),0,1)
    # training_samples.add_data("CoM_momenta", reshaped)

    reshaped = [training_parameters["mom_CoM_P1"], training_parameters["mom_CoM_P2"], training_parameters["mom_CoM_P3"]]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
    training_samples.add_data("CoM_momenta", reshaped)

    training_samples.add_data("mass_P1", np.asarray(results.particle_1_M))
    training_samples.add_data("mass_P2", np.asarray(results.particle_2_M))
    training_samples.add_data("mass_P3", np.asarray(results.particle_3_M))
    reshaped = [results.particle_1_M, results.particle_2_M, results.particle_3_M]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
    training_samples.add_data("masses", reshaped)

    _B_properties_preprocessor = B_properties_preprocessor([training_samples.get_data_i("B_properties")])
    training_samples.update_preprocess_i("B_properties", _B_properties_preprocessor.preprocess(
        training_samples.get_data_i("B_properties")))

    _CoM_angles_preprocessor = CoM_angles_preprocessor(training_samples.get_data_i("CoM_angles"))
    training_samples.update_preprocess_i("CoM_angles",
                                         _CoM_angles_preprocessor.preprocess(training_samples.get_data_i("CoM_angles")))

    _CoM_momenta_preprocessor = CoM_momenta_preprocessor(training_samples.get_data_i("CoM_momenta"))
    training_samples.update_preprocess_i("CoM_momenta", _CoM_momenta_preprocessor.preprocess(
        training_samples.get_data_i("CoM_momenta")))

    preprocessors = {}
    preprocessors["B_properties"] = _B_properties_preprocessor
    preprocessors["CoM_angles"] = _CoM_angles_preprocessor
    preprocessors["CoM_momenta"] = _CoM_momenta_preprocessor

    return training_samples, preprocessors


def get_data_simple(file):
    file = uproot.open(file)["DecayTree"]
    keys = file.keys()
    results = file.arrays(keys, library="np")
    results = pd.DataFrame.from_dict(results)

    mother_P = np.sqrt(results.mother_PX**2+results.mother_PY**2+results.mother_PZ**2)
    mother_P_true = np.sqrt(results.mother_PX_TRUE**2+results.mother_PY_TRUE**2+results.mother_PZ_TRUE**2)


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
        norm(np.asarray([np.zeros((np.shape(Bmass))), np.zeros((np.shape(Bmass))), np.ones((np.shape(Bmass)))])), 0, 1)

    ROT_matrix = rotation_matrix_from_vectors_vectorised(B_vec, all_pz)

    P1 = vector.obj(px=results.particle_1_PX, py=results.particle_1_PY, pz=results.particle_1_PZ,
                    E=results.particle_1_E)
    P2 = vector.obj(px=results.particle_2_PX, py=results.particle_2_PY, pz=results.particle_2_PZ,
                    E=results.particle_2_E)
    P3 = vector.obj(px=results.particle_3_PX, py=results.particle_3_PY, pz=results.particle_3_PZ,
                    E=results.particle_3_E)
    PM= vector.obj(px=results.mother_PX_TRUE, py=results.mother_PY_TRUE, pz=results.mother_PZ_TRUE, E=results.mother_E_TRUE)



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

    # pe_1 = np.sqrt(results.particle_1_M**2 + P1_ROT.px**2 + P1_ROT.py**2 + P1_ROT.pz**2)
    # pe_2 = np.sqrt(results.particle_2_M**2 + P2_ROT.px**2 + P2_ROT.py**2 + P2_ROT.pz**2)
    # pe_3 = np.sqrt(results.particle_3_M**2 + P3_ROT.px**2 + P3_ROT.py**2 + P3_ROT.pz**2)
    # pe = pe_1 + pe_2 + pe_3
    # px = P1_ROT.px + P2_ROT.px + P3_ROT.px
    # py = P1_ROT.py + P2_ROT.py + P3_ROT.py
    # pz = P1_ROT.pz + P2_ROT.pz + P3_ROT.pz
    # mass = np.sqrt(pe**2 - px**2 - py**2 - pz**2)
    # print(mass)

    # pe_1 = np.sqrt(results.particle_1_M**2 + P1.px**2 + P1.py**2 + P1.pz**2)
    # pe_2 = np.sqrt(results.particle_2_M**2 + P2.px**2 + P2.py**2 + P2.pz**2)
    # pe_3 = np.sqrt(results.particle_3_M**2 + P3.px**2 + P3.py**2 + P3.pz**2)
    # pe = pe_1 + pe_2 + pe_3
    # px = P1.px + P2.px + P3.px
    # py = P1.py + P2.py + P3.py
    # pz = P1.pz + P2.pz + P3.pz
    # mass = np.sqrt(pe**2 - px**2 - py**2 - pz**2)
    # print(mass)

    # quit()

    training_samples = DatasetAndLabels("dataset")

    reshaped = np.asarray([[training_parameters["P1_px"], training_parameters["P1_py"], training_parameters["P1_pz"]],
                           [training_parameters["P2_px"], training_parameters["P2_py"], training_parameters["P2_pz"]],
                           [training_parameters["P3_px"], training_parameters["P3_py"], training_parameters["P3_pz"]]])
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 2)
    training_samples.add_data("momenta", reshaped)


    reshaped = np.asarray([[training_parameters["PM_px"], training_parameters["PM_py"], training_parameters["PM_pz"]]])
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 2)
    training_samples.add_data("mother_momenta", reshaped)

    reshaped = np.asarray([[training_parameters["B_phi"], training_parameters["B_theta"], training_parameters["B_P"]]])
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 2)
    training_samples.add_data("B_angles", reshaped)

    _momenta_preprocessor = MomentaPreprocessor(training_samples.get_data_i("momenta"), training_samples.get_data_i("momenta_mother"))
    training_samples.update_preprocess_i("momenta",
                                         _momenta_preprocessor.preprocess(training_samples.get_data_i("momenta")))
    training_samples.update_preprocess_i("mother_momenta",
                                         _momenta_preprocessor.preprocess(training_samples.get_data_i("mother_momenta")))

    _B_angles_preprocessor = B_angles_preprocessor(training_samples.get_data_i("B_angles"))
    training_samples.update_preprocess_i("B_angles",
                                         _B_angles_preprocessor.preprocess(training_samples.get_data_i("B_angles")))

    reshaped = [results.particle_1_M, results.particle_2_M, results.particle_3_M]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 1)
    reshaped = np.expand_dims(reshaped, 2)
    training_samples.add_data("masses", reshaped)

    preprocessors = {}

    preprocessors["momenta"] = _momenta_preprocessor
    preprocessors["B_angles"] = _B_angles_preprocessor

    return training_samples, preprocessors

    '''
    random_phi = np.random.uniform(low=-math.pi,high=math.pi, size=(np.shape(Bmass)))
    random_theta = np.random.uniform(low=-math.pi,high=math.pi, size=(np.shape(Bmass)))

    rot_pz = np.ones((np.shape(Bmass)))
    rot_py = rot_pz*np.tan(random_phi)*np.tan(random_theta)
    rot_px = rot_py*(1./np.tan(random_theta))

    all_pz = np.swapaxes(norm(np.asarray([np.zeros((np.shape(Bmass))),np.zeros((np.shape(Bmass))),np.ones((np.shape(Bmass)))])),0,1)
    rot_vec = np.swapaxes(norm(np.asarray([rot_px, rot_py, rot_pz])),0,1)
    ROT_matrix_random = rotation_matrix_from_vectors_vectorised(all_pz, rot_vec)

    P1_vec = [np.zeros((np.shape(Bmass))), np.zeros((np.shape(Bmass))), P1_p]
    P1_vec_ROT = rot_vectorised(P1_vec, ROT_matrix_random)
    # P1_vec_ROT = P1_vec

    E = np.sqrt(results.particle_1_M**2 + P1_vec_ROT[0]**2 + P1_vec_ROT[1]**2 + P1_vec_ROT[2]**2)
    P1 = vector.obj(px=P1_vec_ROT[0], py=P1_vec_ROT[1], pz=P1_vec_ROT[2], E=E) 


    angles = np.swapaxes(np.asarray([training_parameters["phi_P2"], training_parameters["theta_P2"], training_parameters["phi_P3"], training_parameters["theta_P3"]]),0,1)

    sign_pz_P2 = np.ones(np.shape(Bmass))
    sign_pz_P3 = np.ones(np.shape(Bmass))

    where = np.where((angles[:,0]<0)&(angles[:,2]>math.pi/2.)&(angles[:,0]>-math.pi/2.))
    sign_pz_P2[where[0]] *= -1.
    where = np.where((angles[:,0]>0)&(angles[:,2]<-math.pi/2.)&(angles[:,0]<math.pi/2.))
    sign_pz_P2[where[0]] *= -1.

    where = np.where((angles[:,0]<0)&(angles[:,2]<math.pi/2.)&(angles[:,0]<-math.pi/2.))
    sign_pz_P3[where[0]] *= -1.
    where = np.where((angles[:,0]>0)&(angles[:,2]>-math.pi/2.)&(angles[:,0]>math.pi/2.))
    sign_pz_P3[where[0]] *= -1.

    ##### 

    rot_pz = sign_pz_P2*np.ones((np.shape(Bmass)))
    rot_py = rot_pz*np.tan(angles[:,0])*np.tan(angles[:,1])
    rot_px = rot_py*(1./np.tan(angles[:,1]))

    all_pz = np.swapaxes(norm(np.asarray([np.zeros((np.shape(Bmass))),np.zeros((np.shape(Bmass))),np.ones((np.shape(Bmass)))])),0,1)
    rot_vec = np.swapaxes(norm(np.asarray([rot_px, rot_py, rot_pz])),0,1)
    ROT_matrix_P2 = rotation_matrix_from_vectors_vectorised(all_pz, rot_vec)

    P2_vec = [np.zeros((np.shape(Bmass))), np.zeros((np.shape(Bmass))), P2_p]
    P2_vec_ROT = rot_vectorised(-rot_vectorised(P2_vec, ROT_matrix_P2), ROT_matrix_random)
    # P2_vec_ROT = rot_vectorised(P2_vec, ROT_matrix_P2)

    E = np.sqrt(results.particle_2_M**2 + P2_vec_ROT[0]**2 + P2_vec_ROT[1]**2 + P2_vec_ROT[2]**2)
    P2 = vector.obj(px=P2_vec_ROT[0], py=P2_vec_ROT[1], pz=P2_vec_ROT[2], E=E) 

    ##### 

    rot_pz = sign_pz_P3*np.ones((np.shape(Bmass)))
    rot_py = rot_pz*np.tan(angles[:,2])*np.tan(angles[:,3])
    rot_px = rot_py*(1./np.tan(angles[:,3]))

    all_pz = np.swapaxes(norm(np.asarray([np.zeros((np.shape(Bmass))),np.zeros((np.shape(Bmass))),np.ones((np.shape(Bmass)))])),0,1)
    rot_vec = np.swapaxes(norm(np.asarray([rot_px, rot_py, rot_pz])),0,1)
    ROT_matrix_P3 = rotation_matrix_from_vectors_vectorised(all_pz, rot_vec)

    P3_vec = [np.zeros((np.shape(Bmass))), np.zeros((np.shape(Bmass))), P3_p]
    P3_vec_ROT = rot_vectorised(-rot_vectorised(P3_vec, ROT_matrix_P3), ROT_matrix_random)
    # P3_vec_ROT = rot_vectorised(P3_vec, ROT_matrix_P3)

    E = np.sqrt(results.particle_3_M**2 + P3_vec_ROT[0]**2 + P3_vec_ROT[1]**2 + P3_vec_ROT[2]**2)
    P3 = vector.obj(px=P3_vec_ROT[0], py=P3_vec_ROT[1], pz=P3_vec_ROT[2], E=E) 


    P1 = P1.boost_beta3(B.to_beta3())
    P2 = P2.boost_beta3(B.to_beta3())
    P3 = P3.boost_beta3(B.to_beta3())


    pe = P1.E + P2.E + P3.E
    px = P1.px + P2.px + P3.px
    py = P1.py + P2.py + P3.py
    pz = P1.pz + P2.pz + P3.pz
    B = vector.obj(px=px, py=py, pz=pz, E=pe)
    Bmass = np.sqrt(B.E**2 - B.px**2 - B.py**2 - B.pz**2)
    '''

    training_samples = DatasetAndLabels("dataset")

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
    Bmass = np.sqrt(pe ** 2 - px ** 2 - py ** 2 - pz ** 2)

    pe = pe_1 + pe_2
    px = results.particle_1_PX + results.particle_2_PX
    py = results.particle_1_PY + results.particle_2_PY
    pz = results.particle_1_PZ + results.particle_2_PZ
    mass_12 = np.sqrt(pe ** 2 - px ** 2 - py ** 2 - pz ** 2)

    pe = pe_1 + pe_3
    px = results.particle_1_PX + results.particle_3_PX
    py = results.particle_1_PY + results.particle_3_PY
    pz = results.particle_1_PZ + results.particle_3_PZ
    mass_13 = np.sqrt(pe ** 2 - px ** 2 - py ** 2 - pz ** 2)

    pe = pe_2 + pe_3
    px = results.particle_2_PX + results.particle_3_PX
    py = results.particle_2_PY + results.particle_3_PY
    pz = results.particle_2_PZ + results.particle_3_PZ
    mass_23 = np.sqrt(pe ** 2 - px ** 2 - py ** 2 - pz ** 2)

    B_momenta_PX = results.particle_1_PX + results.particle_2_PX + results.particle_3_PX
    B_momenta_PY = results.particle_1_PY + results.particle_2_PY + results.particle_3_PY
    B_momenta_PZ = results.particle_1_PZ + results.particle_2_PZ + results.particle_3_PZ

    frac_1 = np.sqrt(results.particle_1_PX ** 2 + results.particle_1_PY ** 2 + results.particle_1_PZ ** 2) / (
        np.sqrt(B_momenta_PX ** 2 + B_momenta_PY ** 2 + B_momenta_PZ ** 2))

    frac_2 = np.sqrt(results.particle_2_PX ** 2 + results.particle_2_PY ** 2 + results.particle_2_PZ ** 2) / (
        np.sqrt(B_momenta_PX ** 2 + B_momenta_PY ** 2 + B_momenta_PZ ** 2))

    frac_3 = np.sqrt(results.particle_3_PX ** 2 + results.particle_3_PY ** 2 + results.particle_3_PZ ** 2) / (
        np.sqrt(B_momenta_PX ** 2 + B_momenta_PY ** 2 + B_momenta_PZ ** 2))

    B_momenta_PT = np.sqrt(B_momenta_PX ** 2 + B_momenta_PY ** 2)
    B_momenta_P = np.sqrt(B_momenta_PX ** 2 + B_momenta_PY ** 2 + B_momenta_PZ ** 2)

    frac_pt = B_momenta_PT / B_momenta_P

    auxiliary_variables = np.swapaxes(np.asarray([Bmass, mass_12, mass_13, mass_23, frac_1, frac_2, frac_3, frac_pt]),
                                      0, 1)

    reshaped = [[results.particle_1_PX, results.particle_2_PX, results.particle_3_PX],
                [results.particle_1_PY, results.particle_2_PY, results.particle_3_PY],
                [results.particle_1_PZ, results.particle_2_PZ, results.particle_3_PZ]]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 2)

    training_samples.add_data("daughter_momenta", reshaped)

    reshaped = [[results.particle_1_M_TRUE, results.particle_2_M_TRUE, results.particle_3_M_TRUE]]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 2)

    training_samples.add_label("daughter_masses", reshaped)

    reshaped = [[results.mother_M_TRUE]]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 2)

    training_samples.add_label("mother_masses", reshaped)

    reshaped = [[results.mother_PX_TRUE],
                [results.mother_PY_TRUE],
                [results.mother_PZ_TRUE]]
    reshaped = np.swapaxes(np.asarray(reshaped), 0, 2)

    training_samples.add_label("mother_momentum", reshaped)

    training_samples.add_label("auxiliary_variables", auxiliary_variables)

    # training_samples.update_preprocess_i("daughter_momenta", preprocess(training_samples.get_data_i("daughter_momenta")))

    # print(training_samples.get_data_i("daughter_momenta",mode='train')[0])
    # print(training_samples.get_data_i("daughter_momenta",mode='train',preprocessed=True)[0])
    # quit()
    _momentum_preprocessor = momentum_preprocessor(
        [training_samples.get_data_i("daughter_momenta"), training_samples.get_data_i("mother_momentum")])
    training_samples.update_preprocess_i("daughter_momenta", _momentum_preprocessor.preprocess(
        training_samples.get_data_i("daughter_momenta")))
    training_samples.update_preprocess_i("mother_momentum", _momentum_preprocessor.preprocess(
        training_samples.get_data_i("mother_momentum")))

    _auxilary_preprocessor = auxilary_preprocessor(auxiliary_variables)

    training_samples.update_preprocess_i("auxiliary_variables", _auxilary_preprocessor.preprocess(
        training_samples.get_data_i("auxiliary_variables")))

    preprocessors = {}
    preprocessors["momentum"] = _momentum_preprocessor
    preprocessors["aux"] = _auxilary_preprocessor

    return training_samples, preprocessors


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


def get_proccessing_limits(dataset=None, plotting=False):
    if plotting:
        try:
            rd.processing_limits_plotting = rd.processing_limits
        except:
            pass

    try:
        return rd.processing_limits
    except:
        processing_limits = {}
        processing_limits['px'] = {}
        processing_limits['py'] = {}
        processing_limits['pz'] = {}

        dataset_copy = dataset.copy()
        dataset_copy[:, :, 2] = dataset_copy[:, :, 2] + 5
        dataset_copy[:, :, 2] = np.log(dataset_copy[:, :, 2])

        for i, particle_i in enumerate([1, 2, 3]):
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

        if plotting:
            rd.processing_limits_plotting = processing_limits
            return

        rd.processing_limits = processing_limits

        return rd.processing_limits


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


class VaeDataset(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = True,
            **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        dataset, preprocessors = get_data_simple(self.data_path)
        dataset.permutate_particle_number()

        train_momenta_pp = dataset.get_data_i("momenta", mode="train", preprocessed=True).astype("float32")
        train_mother_momenta_pp = dataset.get_data_i("momenta_mother", mode="train", preprocessed=True).astype("float32")

        test_momenta_pp = dataset.get_data_i("momenta", mode="test", preprocessed=True).astype("float32")
        test_mother_momenta_pp = dataset.get_data_i("momenta_mother", mode="test", preprocessed=True).astype("float32")

        train_momenta_pp = np.reshape(train_momenta_pp, (-1, 9))
        train_mother_momenta_pp = np.reshape(train_mother_momenta_pp, (-1, 3))

        test_momenta_pp = np.reshape(test_momenta_pp, (-1, 9))
        test_mother_momenta_pp = np.reshape(test_mother_momenta_pp, (-1, 3))

        self.dataset_train = TensorDataset(torch.Tensor(train_momenta_pp), torch.Tensor(train_mother_momenta_pp))
        self.dataset_test = TensorDataset(torch.Tensor(test_momenta_pp), torch.Tensor(test_mother_momenta_pp))



    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test,
            batch_size=100,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )



class OnlineThreeBodyDecayMomentaPreprocessor(nn.Module):
    def __init__(self, estimation_sample, estimation_sample_mother):
        super(OnlineThreeBodyDecayMomentaPreprocessor, self).__init__()
        self.limits = self.get_limits_from_samples(estimation_sample, estimation_sample_mother)


    def forward(self, sample, sample_mother, direction=1):
        """

        :param data: tuple: sample, sample_mother
        :param direction: 1 if forward, -1 if reverse
        :return:
        """

        if direction == 1:
            return self.preprocess(sample, sample_mother)
        elif direction == -1:
            return self.postprocess(sample, sample_mother)
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
        self.min_decay_prods = min_func(sample_copy)
        self.max_decay_prods = max_func(sample_copy)

        sample_mother_copy = sample_mother * 1.0
        sample_mother_copy[:, :, 2] = torch.log(sample_mother_copy[:, :, 2] + 5)
        # From [B, 3, 3] to [3, 3]
        self.min_mother = min_func(sample_mother_copy)
        self.max_mother = max_func(sample_mother_copy)

    def preprocess(self, sample, sample_mother):
        if sample.device != self.min_decay_prods.device:
            self.min_decay_prods = self.min_decay_prods.to(sample.device)
            self.max_decay_prods = self.max_decay_prods.to(sample.device)

            self.min_mother = self.min_mother.to(sample.device)
            self.max_mother = self.max_mother.to(sample.device)


        if sample is not None:
            sample = sample * 1.0
            sample[:, :, 2] = torch.log(sample[:, :, 2] + 5)
            sample = ((sample - self.min_decay_prods) / (self.max_decay_prods - self.min_decay_prods)) * 2.0 - 1.0

        if sample_mother is not None:
            sample_mother = sample_mother * 1.0
            sample_mother[:, :, 2] = torch.log(sample_mother[:, :, 2] + 5)
            sample_mother = ((sample_mother - self.min_mother) / (self.max_mother - self.min_mother)) * 2.0 - 1.0

        return sample, sample_mother

    def postprocess(self, sample, sample_mother):
        if sample is not None:
            sample = sample * 1
            sample = (sample + 1) * 0.5 * (self.max_decay_prods - self.min_decay_prods) + self.min_decay_prods
            sample[:, :, 2] = torch.exp(sample[:, :, 2]) - 5

        if sample_mother is not None:
            sample_mother = sample_mother * 1
            sample_mother = (sample_mother + 1) * 0.5 * (self.max_mother - self.min_mother) + self.min_mother
            sample_mother[:, :, 2] = torch.exp(sample_mother[:, :, 2]) - 5

        return sample, sample_mother

class ThreeBodyDecayDataset(LightningDataModule):
    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            num_workers: int = 0,
            pin_memory: bool = True,
            train_test_split: float = 0.8,
            **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_test_split = train_test_split

    def get_data_simple(self, file):
        file = uproot.open(file)["DecayTree"]
        keys = file.keys()
        results = file.arrays(keys, library="np")
        results = pd.DataFrame.from_dict(results)

        mother_P = np.sqrt(results.mother_PX ** 2 + results.mother_PY ** 2 + results.mother_PZ ** 2)
        mother_P_true = np.sqrt(results.mother_PX_TRUE ** 2 + results.mother_PY_TRUE ** 2 + results.mother_PZ_TRUE ** 2)

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
            norm(np.asarray([np.zeros((np.shape(Bmass))), np.zeros((np.shape(Bmass))), np.ones((np.shape(Bmass)))])), 0,
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

        return momenta, momenta_mother


    def split(self, data, split_at):
        return data[:split_at], data[split_at:]


    def setup(self, stage: Optional[str] = None) -> None:
        momenta, momenta_mother = self.get_data_simple(self.data_path)

        split_at = int(self.train_test_split * momenta.shape[0])

        momenta_train, momenta_test = self.split(momenta, split_at)
        momenta_mother_train, momenta_mother_test = self.split(momenta_mother, split_at)

        momenta_train, momenta_test = torch.Tensor(momenta_train), torch.Tensor(momenta_test)
        momenta_mother_train, momenta_mother_test = torch.Tensor(momenta_mother_train), torch.Tensor(momenta_mother_test)

        self.dataset_train = TensorDataset(momenta_train, momenta_mother_train)
        self.dataset_test = TensorDataset(momenta_test, momenta_mother_test)

        self.preprocessor = OnlineThreeBodyDecayMomentaPreprocessor(momenta_train, momenta_mother_train)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def predict_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataset_test,
            batch_size=100,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
