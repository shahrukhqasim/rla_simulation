import math
import os
import time
import unittest
import uuid
from typing import Optional

import numpy as np
import random

import torch
import uproot
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import threading

from rlasim.lib.data_core import tensors_dict_join, RootBlockShuffledSubsetDataset, RootBlockShuffledSubsetDataLoader
from queue import Queue
from torch.utils.data._utils.collate import default_collate


class TestAddFunction(unittest.TestCase):
    def setUp(self) -> None:
        self.total_length = 1000000
        data = {
            'first': np.arange(self.total_length),
            'second': np.arange(self.total_length),
        }

        my_file = output_folder = str(uuid.uuid4())+'.root'
        file2 = uproot.recreate(my_file, compression=uproot.ZLIB(0))
        file2['DecayTree'] = data
        file2.close()
        self.test_file = my_file

    def doCleanups(self) -> None:
        os.remove(self.test_file)

    def test_RootBlockShuffledSubsetDataset(self):

        # my_file = '/Users/shahrukhqasim/Workspace/UZH/rla_simulation/data/decay_modes_572.root'
        block_size = 1000
        num_blocks = 10
        dataset = RootBlockShuffledSubsetDataset(self.test_file, block_size=block_size, num_blocks=num_blocks)

        collected = []
        for i in tqdm(range(self.total_length//(block_size*num_blocks))):
            for j, x in enumerate(dataset):
                collected += [x['first']]

        assert len(np.unique(collected)) == self.total_length
        dataset.exit()


    def test_RootBlockShuffledSubsetDataLoader(self):

        # my_file = '/Users/shahrukhqasim/Workspace/UZH/rla_simulation/data/decay_modes_572.root'
        block_size = 1000
        num_blocks = 10
        dataloader = RootBlockShuffledSubsetDataLoader(self.test_file, block_size=block_size, num_blocks=num_blocks, batch_size=10)

        collected = []
        for i in tqdm(range(self.total_length//(block_size*num_blocks))):
            for j, x in enumerate(dataloader):
                collected += x['first']

        collected = torch.stack(collected, dim=0)

        assert len(np.unique(collected)) == self.total_length
        dataloader.exit()




if __name__ == '__main__':
    unittest.main()


