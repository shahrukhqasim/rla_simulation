import pickle

import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import vector
import math
from typing import Iterable
from typing import Dict


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
import tensorflow as tf


class LoadedDataset(Dataset):
    def __init__(self, mothers:dict, daughters:dict):
        # Initialize your dataset here
        self.mothers = mothers
        self.daughters = daughters

        lens = set()
        for k,v in mothers.items():
            lens.add(int(v.shape[0]))
        for k,v in daughters.items():
            lens.add(int(v.shape[0]))
        assert len(lens) == 1
        self.len = lens.pop()

        self.keys_mother = self.mothers.keys()
        self.keys_daughters = self.daughters.keys()

    def __len__(self):
        # Return the total number of samples in your dataset
        return self.len

    def __getitem__(self, index):
        return {k: self.mothers[k][index] for k in self.keys_mother}, {k: self.daughters[k][index] for k in self.keys_daughters}

    def get_batch(self, indices):
        # batch_mothers = {k: [self.mothers[k][idx] for idx in indices] for k in self.keys_mother}
        # batch_daughters = {k: [self.daughters[k][idx] for idx in indices] for k in self.keys_daughters}
        # return batch_mothers, batch_daughters

        batch_mothers = {k: tf.gather_nd(self.mothers[k], indices[:, tf.newaxis]) for k in self.keys_mother}
        batch_daughters = {k: tf.gather_nd(self.daughters[k], indices[:, tf.newaxis]) for k in self.keys_daughters}
        return batch_mothers, batch_daughters


class RaggedDatasetLoader(Iterable):
    def __init__(self, dataset: LoadedDataset, batch_size: int, pad_processor, convert_to_torch=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length = len(self.dataset) // self.batch_size
        self.pad_processor = pad_processor
        self.convert_to_torch = convert_to_torch


    def __iter__(self):
        for i in range(self.length):
            indices = tf.constant(np.arange(self.batch_size) + (i * self.batch_size))
            batch = self.dataset.get_batch(indices)
            mother, daughters = batch
            if self.pad_processor is not None:
                mother, daughters = self.pad_processor.forward(mother, daughters)
            if self.convert_to_torch:
                mother, daughters = convert_tf_to_torch(mother), convert_tf_to_torch(daughters)
            yield mother, daughters

    def __len__(self):
        return self.length


class ShipDisDataset(LightningDataModule):
    def __init__(self, data_path_mothers : str, data_path_daughters: str, seed: int,
                 split_train, split_validate, split_test, batch_size, zero_pad: bool, max_num_daughters: int, **kwargs):
        super().__init__()

        assert data_path_mothers.endswith('.pkl')
        assert data_path_daughters.endswith('.pkl')

        self.data_path_mothers = data_path_mothers
        self.data_path_daughters = data_path_daughters
        self.seed = seed
        self.split_train = split_train
        self.split_validate = split_validate
        self.split_test = split_test
        self.batch_size = batch_size

        if zero_pad:
            self.pad_processor = RaggedToZeroPadded(max_length=max_num_daughters)
        else:
            self.pad_processor = None

    def _construct_soa(self, df, df_daughters):
        df_sorted = df.sort_values(by='track_eventNumber')
        soa = {col: tf.constant(df_sorted[col].values) for col in df_sorted.columns}


        df_daughters = df_daughters.sort_values(by='dau_eventNumber')
        daughters_soa = {col: tf.constant(df_daughters[col].values) for col in df_daughters.columns}
        daughters_soa = {k: tf.RaggedTensor.from_value_rowids(v, daughters_soa['dau_eventNumber']) for k,v in daughters_soa.items()}


        # print((daughters_soa['dau_eventNumber'].shape), len(soa['track_eventNumber']))
        # print(daughters_soa)

        return soa, daughters_soa

    def setup(self, stage: Optional[str] = None) -> None:
        try:
            with open(self.data_path_mothers, 'rb') as f:
                df_mothers = pickle.load(f)
        except Exception as e:
            print(f"Error loading mothers data: {e} with path {self.data_path_mothers}")
            raise RuntimeError('Error loading mother data')

        try:
            with open(self.data_path_daughters, 'rb') as f:
                df_daughters = pickle.load(f)
        except Exception as e:
            print(f"Error loading daughters data: {e} with path {self.data_path_daughters}")
            raise RuntimeError('Error loading daughters data')


        # keys in data_mother
        # 'track_vx', 'track_vy', 'track_vz', 'track_px', 'track_py', 'track_pz',
        #        'track_energy', 'track_weight', 'track_ID', 'track_theta_x',
        #        'track_theta_y', 'track_theta', 'track_charge', 'track_eventNumber'],
        #       dtype='object'

        # keys in data_daughters
        # 'dau_vx', 'dau_vy', 'dau_vz', 'dau_px', 'dau_py', 'dau_pz',
        #        'dau_energy', 'dau_weight', 'dau_ID', 'dau_theta_x', 'dau_theta_y',
        #        'dau_theta', 'dau_charge', 'dau_eventNumber'],
        #       dtype='object'

        df_daughters['dau_mask'] = (df_daughters['dau_ID']!=0).astype('int32')


        data_mothers, data_daughters = self._construct_soa(df_mothers, df_daughters)


        rng = np.random.default_rng(seed=self.seed)
        total = self.split_validate + self.split_test + self.split_train
        fraction_train = self.split_train / total
        fraction_validate = self.split_validate / total
        fraction_test = self.split_test / total

        # Shuffle indices
        indices = np.arange(len(df_mothers))
        rng.shuffle(indices)

        # Split indices based on the fractions
        num_train = int(fraction_train * len(indices))
        num_validate = int(fraction_validate * len(indices))

        train_indices = indices[:num_train]
        validate_indices = indices[num_train:num_train + num_validate]
        test_indices = indices[num_train + num_validate:]

        self.data_daughters_train = {k: tf.gather_nd(v, tf.constant(train_indices)[:, tf.newaxis]) for k,v in data_daughters.items()}
        self.data_mothers_train = {k: tf.gather_nd(v, tf.constant(train_indices)[:, tf.newaxis]) for k,v in data_mothers.items()}

        self.data_daughters_validate = {k: tf.gather_nd(v, tf.constant(validate_indices)[:, tf.newaxis]) for k,v in data_daughters.items()}
        self.data_mothers_validate = {k: tf.gather_nd(v, tf.constant(validate_indices)[:, tf.newaxis]) for k,v in data_mothers.items()}

        self.data_daughters_test = {k: tf.gather_nd(v, tf.constant(test_indices)[:, tf.newaxis]) for k,v in data_daughters.items()}
        self.data_mothers_test = {k: tf.gather_nd(v, tf.constant(test_indices)[:, tf.newaxis]) for k,v in data_mothers.items()}

        self.train_loader = RaggedDatasetLoader(LoadedDataset(self.data_mothers_train, self.data_daughters_train), self.batch_size, pad_processor=self.pad_processor)
        self.validate_loader = RaggedDatasetLoader(LoadedDataset(self.data_mothers_validate, self.data_daughters_validate), self.batch_size, pad_processor=self.pad_processor)
        self.test_loader = RaggedDatasetLoader(LoadedDataset(self.data_mothers_test, self.data_daughters_test), self.batch_size, pad_processor=self.pad_processor)

    def train_dataloader(self):
        return self.train_loader

    def test_dataloader(self):
        return self.test_loader

    def val_dataloader(self):
        return self.validate_loader



class RaggedToZeroPadded():
    def __init__(self, max_length=50):
        self.max_length = max_length

    def forward(self, mother: Dict[str, tf.RaggedTensor], daughters: Dict[str, tf.RaggedTensor]):
        daughters_2 = dict()
        for k, v in daughters.items():
            regular_tensor = v.to_tensor(shape=(v.shape[0], self.max_length), default_value=0)
            daughters_2[k] = regular_tensor

            # resulting_tensor = tf.zeros(shape=(v.shape[0], self.max_length), dtype=v.dtype)
            #
            # segment_ids = tf.ragged.row_splits_to_segment_ids(v.row_splits)
            # # tf.ragged.
            #
            # index_batch = tf.RaggedTensor.from_row_splits(segment_ids, v.row_splits).values
            # index_secondary = tf.RaggedTensor.from_row_splits(tf.ones_like(segment_ids), v.row_splits)
            # index_secondary = tf.math.cumsum(index_secondary, axis=1, exclusive=True).values
            # print(index_batch.shape, index_secondary.shape)
            # indexing_tensor = tf.RaggedTensor.from_row_splits(tf.concat((index_batch[:, tf.newaxis], index_secondary[:, tf.newaxis]), axis=-1), v.row_splits)
            #
            # print(indexing_tensor)
            # tf.scatter_nd(resulting_tensor, indexing_tensor, v.shape)
            # print(resulting_tensor)


            # tf.scatter_nd()
            # print(segment_ids)
            # print(x)


        return mother, daughters_2


def convert_tf_to_torch(tf_tensor_dict : Dict[str, tf.Tensor], double_to_float=True):
    pt_tensor_dict = {}

    for name, tf_tensor in tf_tensor_dict.items():
        # Convert TensorFlow tensor to NumPy array
        np_array = tf_tensor.numpy()

        # Convert NumPy array to PyTorch tensor
        if np_array.dtype == np.double and double_to_float:
            np_array = np_array.astype(np.float32)

        pt_tensor = torch.from_numpy(np_array)

        # Store the converted PyTorch tensor in the dictionary
        pt_tensor_dict[name] = pt_tensor

    return pt_tensor_dict




