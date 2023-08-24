import collections
import math
from typing import Dict, Union, Type, Tuple, Callable, Optional, Iterable

import uproot

torch_installed = True
try:
    import torch
    from torch.utils.data import Dataset, TensorDataset
except:
    torch_installed = False
    pass
import numpy as np
from particle import Particle
import warnings


def get_pdgid(name_):
    rapidsim2pdg_mapping = {
        'p+': 'p',
        'p-': 'p~',
        'nue': 'nu(e)',
        'anti-nue': 'nu(e)~',
        'Bs0': 'B(s)0',
        'Bs0b': 'B(s)~0',
        'Bc+': 'B(c)+',
        'Bc-': 'B(c)-',
        'D0b': 'D~0',
        'Ds+': 'D(s)+',
        'Ds-': 'D(s)-',
    }

    if name_ in rapidsim2pdg_mapping:
        name = rapidsim2pdg_mapping[name_]
    else:
        name = name_

    respective = Particle.findall(name=name)
    if len(respective) == 0:
        raise ValueError('Unable to find a match for the particle %s.'%name_)


    if len(respective) > 1:
        verified_multiple_ok = {'p'}
        if name not in verified_multiple_ok:
            warnings.warn(RuntimeWarning('Multiple particles found for the input %s: '%name_+str(respective)))

    return int(respective[0].pdgid)

if torch_installed:
    import re

    default_collate_err_msg_format = (
        "nb_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    np_str_obj_array_pattern = re.compile(r'[SaUO]')


    def collate(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        elem = batch[0]
        elem_type = type(elem)

        if collate_fn_map is not None:
            if elem_type in collate_fn_map:
                return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)

            for collate_type in collate_fn_map:
                if isinstance(elem, collate_type):
                    return collate_fn_map[collate_type](batch, collate_fn_map=collate_fn_map)

        if isinstance(elem, collections.abc.Mapping):
            try:
                return elem_type({key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem})
            except TypeError:
                # The mapping type may not support `__init__(iterable)`.
                return {key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(collate(samples, collate_fn_map=collate_fn_map) for samples in zip(*batch)))
        elif isinstance(elem, collections.abc.Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError('each element in list of batch should be of equal size')
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

            if isinstance(elem, tuple):
                return [collate(samples, collate_fn_map=collate_fn_map) for samples in
                        transposed]  # Backwards compatibility.
            else:
                try:
                    return elem_type([collate(samples, collate_fn_map=collate_fn_map) for samples in transposed])
                except TypeError:
                    # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                    return [collate(samples, collate_fn_map=collate_fn_map) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))

    def collate_tensor_fn_no_batch_extension(batch, *, collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        elem = batch[0]
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem._typed_storage()._new_shared(numel, device=elem.device)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.cat(batch, 0, out=out)

    def collate_numpy_array_fn(batch, *,
                               collate_fn_map: Optional[Dict[Union[Type, Tuple[Type, ...]], Callable]] = None):
        elem = batch[0]
        # array of string classes and object
        if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
            raise TypeError(default_collate_err_msg_format.format(elem.dtype))

        return collate([torch.as_tensor(b) for b in batch], collate_fn_map=collate_fn_map)


    nbe_collate_fn_map: Dict[Union[Type, Tuple[Type, ...]], Callable] = {torch.Tensor: collate_tensor_fn_no_batch_extension}
    nbe_collate_fn_map[np.ndarray] = collate_numpy_array_fn


    def nbe_default_collate(batch):
        return collate(batch, collate_fn_map=nbe_collate_fn_map)



    def tensors_dict_join(list_of_dicts):
        collected_tensors = {}

        for dictionary in list_of_dicts:
            for key, tensor in dictionary.items():
                if key in collected_tensors:
                    collected_tensors[key].append(tensor)
                else:
                    collected_tensors[key] = [tensor]

        concatenated_dict = {}

        for key, tensor_list in collected_tensors.items():
            if isinstance(tensor_list[0], np.ndarray):
                concatenated_dict[key] = np.concatenate(tensor_list, axis=0)
            else:
                concatenated_dict[key] = torch.cat(tensor_list, dim=0)

        return concatenated_dict


    class RootTensorDataset2(Dataset):
        def __init__(self, file, tree_name, cache_size=1000):
            self.tree = uproot.open(file)[tree_name]
            self.keys = self.tree.keys()
            self.block_size = cache_size
            self.cache = []
            self.cache_start = 0
            self.cache_end = 0

            lens = set()
            for k in self.keys:
                num_entries = self.tree[k].num_entries
                lens = lens.union({num_entries})
            assert len(lens) == 1
            self.length = list(lens)[0]
            self.length = math.floor(self.length/cache_size)

        def get_block(self, start):
            start = start * self.block_size
            end = min(start+self.block_size, self.length*self.block_size)
            print('x',start, end)
            data = self.tree.arrays(self.keys, entry_start=start, entry_stop=end, library='np')
            return data

        def __getitem__(self, item):
            data_block = self.get_block(item)

            return data_block

        def __len__(self):
            return self.length


    import math
    import time
    from typing import Optional

    import numpy as np
    import random

    import uproot
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import threading

    from rlasim.lib.data_core import tensors_dict_join
    from queue import Queue
    from torch.utils.data._utils.collate import default_collate


    class RootBlockShuffledSubsetDataset(Dataset):
        def shuffle_dict_arrays(self, dict_of_arrays):
            # Assuming all arrays have the same length
            array_length = len(next(iter(dict_of_arrays.values())))
            shuffling_indices = np.arange(array_length)
            np.random.shuffle(shuffling_indices)

            shuffled_dict = {key: array[shuffling_indices] for key, array in dict_of_arrays.items()}

            return shuffled_dict

        def select_and_delete_elements(self, input_list, N):
            if N >= len(input_list):
                selected_elements = input_list[:]
                input_list.clear()
            else:
                selected_elements = input_list[:N]
                del input_list[:N]

            return selected_elements

        def read_root_file(self, file_path, tree_name, start_index, end_index):
            with uproot.open(file_path) as file:
                tree = file[tree_name]
                branches = tree.arrays(library="np", entry_start=start_index, entry_stop=end_index)

            return branches

        # Define the function that will be executed in each thread
        def read_blocks_thread(self, sampled_blocks, data_read):
            while True:
                with self.block_lock:
                    if not sampled_blocks:
                        break
                    b = sampled_blocks.pop()

                start = b * self.block_size
                end = min((b + 1) * self.block_size, self.length_full)
                x = self.read_root_file(self.input_file, 'DecayTree', start, end)
                if self.debug:
                    print("Done", len(data_read))

                cache_2 = {}
                for k, v in x.items():
                    if v.dtype == np.float64:
                        cache_2[k] = v.astype(np.float32)
                    elif v.dtype == np.int64:
                        cache_2[k] = v.astype(np.int32)
                    else:
                        cache_2[k] = v
                x = cache_2

                # Lock the data_read list to avoid concurrent modification
                with self.data_lock:
                    data_read.append(x)

        def _background_loading_thread(self):
            while True:
                if self.kill_signal:
                    break
                if self.sampled_subsets_queue.qsize() > 2:
                    time.sleep(0.1)
                    continue

                self.data_lock = threading.Lock()
                self.block_lock = threading.Lock()

                if self.all_blocks is None:
                    self.all_blocks = list(range(0, math.floor(self.length_full / self.block_size)))
                    random.shuffle(self.all_blocks)
                elif len(self.all_blocks) < self.num_blocks:
                    prev_blocks = [x for x in self.all_blocks]
                    new_all_blocks = list(
                        set(range(0, math.floor(self.length_full / self.block_size))) - set(prev_blocks))
                    random.shuffle(new_all_blocks)
                    self.all_blocks = prev_blocks + new_all_blocks

                sampled_blocks = self.select_and_delete_elements(self.all_blocks, self.num_blocks)

                data_read = []
                num_threads = 4

                threads = []
                t1 = time.time()
                for _ in range(num_threads):
                    thread = threading.Thread(target=self.read_blocks_thread, args=(sampled_blocks, data_read))
                    thread.start()
                    threads.append(thread)

                # Wait for all threads to finish
                for thread in threads:
                    thread.join()

                print("Took", time.time() - t1, "seconds.")

                sampled_subset = tensors_dict_join(data_read)
                sampled_subset = self.shuffle_dict_arrays(sampled_subset)
                self.sampled_subsets_queue.put(sampled_subset)

        def load_subsets(self):
            self.all_blocks = None
            self.reading_thread_main = threading.Thread(target=self._background_loading_thread, args=())
            self.reading_thread_main.start()

        def __init__(self, input_file, block_size=4096, num_blocks=10):
            self.block_size = block_size
            self.num_blocks = num_blocks
            self.input_file = input_file
            self.kill_signal = False
            self.debug = False

            # length = 1000000
            tree = uproot.open(input_file)['DecayTree']
            self.keys = tree.keys()
            lens = set()
            for k in self.keys:
                num_entries = tree[k].num_entries
                lens = lens.union({num_entries})
            assert len(lens) == 1
            # self.length_full = min(length, list(lens)[0])
            self.length_full = list(lens)[0]
            self.length_sampled = num_blocks * block_size
            self.sampled_subsets_queue = Queue()

            self.current_sampled_subset = None

            self.num_retrieved = 0
            self.load_subsets()

        def __len__(self):
            return self.length_sampled

        def __getitem__(self, item):
            if self.current_sampled_subset is None:
                while self.sampled_subsets_queue.qsize() == 0:
                    time.sleep(0.1)
                self.current_sampled_subset = self.sampled_subsets_queue.get()

            results = {key: self.current_sampled_subset[key][item] for key in self.keys}
            self.num_retrieved += 1

            if self.num_retrieved >= self.length_sampled:
                self.current_sampled_subset = None
                self.num_retrieved = 0

            return results

        def exit(self):
            self.kill_signal = True


    class RootBlockShuffledSubsetDataLoader(Iterable):
        def __init__(self, dataset: str, block_size, num_blocks, batch_size):
            # super().__init__(dataset)
            self.dataset = RootBlockShuffledSubsetDataset(dataset, block_size=block_size, num_blocks=num_blocks)

            self.length = math.floor(len(self.dataset) / batch_size)
            self.block_size = block_size
            self.num_blocks = num_blocks
            self.batch_size = batch_size

        def __len__(self):
            return self.length

        def __iter__(self):
            for i in range(self.length):
                batch = []
                for j in range(self.batch_size):
                    batch += [self.dataset.__getitem__(i*self.batch_size + j)]
                yield default_collate(batch)

        def exit(self):
            self.dataset.exit()

    class RootTensorDataset(Dataset):
        def __init__(self, file, tree_name, cache_size=1000):
            self.tree = uproot.open(file)[tree_name]
            self.keys = self.tree.keys()
            self.cache_size = cache_size
            self.cache = []
            self.cache_start = 0
            self.cache_end = 0

            lens = set()
            for k in self.keys:
                num_entries = self.tree[k].num_entries
                lens = lens.union({num_entries})
            assert len(lens) == 1
            self.length = list(lens)[0]

            self.cache_size = min(self.cache_size, self.length)

        def _load_cache(self, item):
            self.cache_start = item
            self.cache_end = min(self.length, self.cache_start + self.cache_size)
            self.cache = self.tree.arrays(self.keys, library="np", entry_start=self.cache_start,
                                          entry_stop=self.cache_end)

            cache_2 = {}
            for k,v in self.cache.items():
                if v.dtype==np.float64:
                    cache_2[k] = v.astype(np.float32)
                elif v.dtype==np.int64:
                    cache_2[k] = v.astype(np.int32)
                else:
                    cache_2[k] = v
            self.cache = cache_2

        def __getitem__(self, item):
            if item < self.cache_start or item >= self.cache_end:
                self._load_cache(item)
            cache_item = item - self.cache_start
            results = {key: self.cache[key][cache_item] for key in self.keys}

            return results

        def __len__(self):
            return self.length

    class DictTensorDataset(Dataset):
        def __init__(self, dict: dict):
            self.dataset_dict = {}
            lens = set()
            self.one_key = None
            for k, v in dict.items():
                assert type(v) is not list
                self.dataset_dict[k] = TensorDataset(v)
                lens = lens.union({len(v)})
                self.one_key = k

            assert len(lens) == 1

        def __getitem__(self, item):
            return_dict = {}
            for k,v in self.dataset_dict.items():
                return_dict[k] = v[item][0] # It returns output as a list
            return return_dict

        def __len__(self):
            return len(self.dataset_dict[self.one_key])
