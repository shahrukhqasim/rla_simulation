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
    assert len(respective) >0

    if len(respective) > 1:
        verified_multiple_ok = {'p'}
        if name not in verified_multiple_ok:
            warnings.warn(RuntimeWarning('Multiple particles found for the input %s: '%name_+str(respective)))

    return int(respective[0].pdgid)

if torch_installed:
    
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
