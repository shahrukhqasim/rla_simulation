import torch
from torch.utils.data import Dataset, TensorDataset
import numpy as np

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
