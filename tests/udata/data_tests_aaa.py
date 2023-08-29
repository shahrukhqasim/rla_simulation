import math
import os
import shutil
import time
import unittest
import uuid
import warnings
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


class TestSubetDataLoader(unittest.TestCase):
    def _my_setup(self, total_length=1000000) -> None:
        self.total_length = total_length
        data = {
            'first': np.arange(self.total_length),
            'second': np.arange(self.total_length),
        }

        self.test_file = os.path.join('temp_files', str(uuid.uuid4())+'.root')
        os.system('mkdir -p temp_files')
        file2 = uproot.recreate(self.test_file, compression=uproot.ZLIB(0))
        file2['DecayTree'] = data
        file2.close()
        print("Setup", self.test_file)

    def _my_cleanup(self) -> None:
        # os.remove(self.test_file)
        # pass
        shutil.rmtree('temp_files')

    def test_RootBlockShuffledSubsetDataset(self):
        self._my_setup()

        # my_file = '/Users/shahrukhqasim/Workspace/UZH/rla_simulation/data/decay_modes_572.root'
        block_size = 1000
        num_blocks = 10
        dataset = RootBlockShuffledSubsetDataset(self.test_file, block_size=block_size, num_blocks=num_blocks)

        collected = []
        for i in tqdm(range(self.total_length//(block_size*num_blocks))):
            for j, x in enumerate(dataset):
                collected += [x['first']]
            dataset.prepare_next_epoch()

        assert len(np.unique(collected)) == self.total_length
        dataset.exit()

        self._my_cleanup()


    def test_RootBlockShuffledSubsetDataLoader(self):
        self._my_setup()

        # my_file = '/Users/shahrukhqasim/Workspace/UZH/rla_simulation/data/decay_modes_572.root'
        block_size = 1000
        num_blocks = 10
        dataloader = RootBlockShuffledSubsetDataLoader(self.test_file, block_size=block_size, num_blocks=num_blocks, batch_size=10)

        collected = []
        for i in tqdm(range(self.total_length//(block_size*num_blocks))):
            for j, x in enumerate(dataloader):
                collected += x['first']
            dataloader.prepare_next_epoch()

        collected = torch.stack(collected, dim=0)

        assert len(np.unique(collected)) == self.total_length
        dataloader.exit()

        self._my_cleanup()

    def test_RootBlockShuffledSubsetDataLoader_overdraw(self):
        ll = 1000000
        ll2 = 50
        self._my_setup(total_length=ll+ll2)

        # my_file = '/Users/shahrukhqasim/Workspace/UZH/rla_simulation/data/decay_modes_572.root'
        block_size = 1000
        num_blocks = 10
        dataloader = RootBlockShuffledSubsetDataLoader(self.test_file, block_size=block_size, num_blocks=num_blocks,
                                                       batch_size=10)

        collected = []
        for i in tqdm(range(self.total_length // (block_size * num_blocks) + 30)):
            for j, x in enumerate(dataloader):
                collected += x['first']
            dataloader.prepare_next_epoch()

        collected = torch.stack(collected, dim=0)

        assert len(np.unique(collected)) == ll
        dataloader.exit()

        self._my_cleanup()

    def test_TooManyBlocks(self):
        block_size = 1000
        num_blocks = 20
        dataloader = None

        self._my_setup(total_length=10000)
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")
            # Trigger a warning.
            dataloader = RootBlockShuffledSubsetDataLoader(self.test_file, block_size=block_size, num_blocks=num_blocks,
                                                           batch_size=10)
            # Verify some things
            assert len(w) == 1
            assert issubclass(w[-1].category, UserWarning)
            assert "blocks" in str(w[-1].message)

        collected = []
        vv = 0
        for j, x in enumerate(dataloader):
            vv += 1
            collected += x['first']

        print("Enumerated for iter %d. Total blocks: %d. Len of loader: %d." % (vv,dataloader.num_blocks, len(dataloader)))
        print("Batch size", dataloader.batch_size)

        collected = torch.stack(collected, dim=0)

        assert len(np.unique(collected)) == self.total_length
        dataloader.exit()

        self._my_cleanup()




if __name__ == '__main__':
    unittest.main()


