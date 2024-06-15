# Adapted from
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
from functools import reduce

import numpy as np
import torch

from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple, Dict
from torch.nn import functional as F
from rlasim.lib.networks import BaseVAE
from operator import mul

from rlasim.lib.organise_data import compute_dalitz_masses, compute_parent_mass

Tensor = TypeVar('torch.tensor')

class MlpConditionalEncoder(nn.Module):
    def __init__(self, in_features, out_features, condition_dims, layer_units=None, activation=torch.relu):
        super(MlpConditionalEncoder, self).__init__()

        if layer_units is None:
            layer_units = [1024, 1024, 1024, 1024, 1024]

        assert isinstance(layer_units, list) and len(layer_units) >= 1, \
               "layer_units should be a list of length at least 1"
        self.layer_units = [in_features + condition_dims] + layer_units
        self.activation = activation

        # Define layers dynamically based on layer_units
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_units) - 1):
            self.layers.append(nn.Linear(self.layer_units[i], self.layer_units[i+1]))

        self.fcl = nn.Linear(self.layer_units[-1], out_features)

    def forward(self, x):
        assert len(x.shape) == 2

        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        x = self.fcl(x)
        return x


class MlpConditionalDecoder(nn.Module):
    def __init__(self, in_features, out_features, condition_dims, layer_units=None, activation=torch.relu):
        super(MlpConditionalDecoder, self).__init__()

        if layer_units is None:
            layer_units = [1024, 1024, 1024, 1024, 1024]

        assert isinstance(layer_units, list) and len(layer_units) >= 1, \
               "layer_units should be a list of length at least 1"
        self.layer_units = [in_features + condition_dims] + layer_units
        self.activation = activation

        # Define layers dynamically based on layer_units
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_units) - 1):
            self.layers.append(nn.Linear(self.layer_units[i], self.layer_units[i+1]))

        self.fcl = nn.Linear(self.layer_units[-1], out_features)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)

        x = self.fcl(x)
        return x


class VaeShipLoss(nn.Module):
    def __init__(self, kld_weight, dalitz_weight=0.0, parent_mass_weight=0.0, loss_type='mse',
                 mask_loss='bce',
                 truth_var='momenta', predicted_var='momenta_reconstructed', **kwards):
        super().__init__()

        self.kld_weight = kld_weight

        self.loss_type = loss_type
        self.predicted_var = predicted_var
        self.truth_var = truth_var
        self.dalitz_weight = dalitz_weight
        self.parent_mass_weight = parent_mass_weight

    def quantile(self, reco, truth, quantile=0.5):
        errors = truth - reco
        loss = torch.max(quantile * errors, (quantile - 1) * errors)
        return torch.mean(loss)

    def forward(self, sample, iteration=-1):
        if self.loss_type == 'mse':
            distance_func = F.mse_loss
        elif self.loss_type == 'huber':
            distance_func = F.huber_loss
        elif self.loss_type == 'quantile':
            distance_func = self.quantile
        else:
            raise ValueError('Unknown value of loss_type')

        mu = sample['mu']
        log_var = sample['log_var']

        # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        # recons2 = torch.nan_to_num(recons)

        # recons_loss = distance_func(recons2, input)
        the_dict = {}

        # sample['dau_mask_reconstructed'] = torch.sigmoid(sample['dau_mask_reconstructed'])

        # print(sample['dau_mask_reconstructed'][:, :, 0].dtype, sample['dau_mask'].dtype)
        recons_loss = F.binary_cross_entropy(sample['dau_mask_reconstructed'][:, :, 0], sample['dau_mask'].float())

        # Calculate binary predictions (0 or 1) based on threshold (e.g., 0.5)
        predicted = (sample['dau_mask_reconstructed'][:, :, 0] >= 0.5).float()

        # Convert sample['dau_mask'] to float tensor if necessary (assuming it's not already)
        target = sample['dau_mask'].float()

        # Calculate accuracy
        accuracy = torch.mean((predicted == target).float()).item()

        # the_dict['loss_reco_p'] = recons_loss

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_weight = self.kld_weight
        print("XXX", float(recons_loss), float(kld_loss), float(accuracy))

        loss = recons_loss + kld_weight * kld_loss

        the_dict.update({'loss': loss, 'loss_reco': recons_loss.detach(), 'loss_kld': -kld_loss.detach()})
        return the_dict

class MlpConditionalVAE(BaseVAE):
    def __init__(self, latent_dim=64,
                 conditional_feats=None,
                 data_feats=None,
                 encoder_layer_units=None,
                 decoder_layer_units=None,
                 sigmoid_on='dau_mask',
                 **kwargs):
        super().__init__()

        # if data_feature_dim is None:
        #     data_feature_dim = [3, 3]

        self.latent_dim = latent_dim
        # self.data_feature_dim = data_feature_dim
        # reduced_data_feature_dim = reduce(mul, self.data_feature_dim)

        if conditional_feats is None:
            conditional_feats = {'momenta_mother_pp': [1, 3]}

        if data_feats is None:
            data_feats = {'momenta': [3, 3]}

        assert type(conditional_feats) is dict

        self.conditional_feats = {}
        reduced_conditional_dim = 0
        for k,v in sorted(conditional_feats.items()):
            reduced_dims = reduce(mul, v)
            self.conditional_feats[k] = reduced_dims
            reduced_conditional_dim += reduced_dims

        self.data_feats = {}
        reduced_data_feature_dim = 0
        for k,shape in sorted(data_feats.items()):
            reduced_shape = reduce(mul, shape)
            self.data_feats[k] = (reduced_shape, shape)
            reduced_data_feature_dim += reduced_shape


        self.encoder = MlpConditionalEncoder(in_features=reduced_data_feature_dim, out_features=self.latent_dim * 4, condition_dims=reduced_conditional_dim, layer_units=encoder_layer_units)
        self.decoder = MlpConditionalDecoder(in_features=latent_dim, out_features=reduced_data_feature_dim, condition_dims=reduced_conditional_dim, layer_units=decoder_layer_units)


        self.fc_mu = nn.Linear(self.latent_dim*4, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim*4, self.latent_dim)

        self._test_exponential_distribution = torch.distributions.Exponential(0.05)

        self.sigmoid_on = sigmoid_on

    def encode(self, x: Tensor, c: Tensor) -> List[Tensor]:
        # assert len(x.shape) == 3
        # assert x.shape[1] == self.data_feature_dim[0]
        # assert x.shape[2] == self.data_feature_dim[1]

        # x = torch.reshape(x, (-1, x.shape[1]*x.shape[2]))
        if c is not None:
            x = torch.cat((x, c), dim=-1)

        result = self.encoder(x)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        # print("Softplusinggg...")
        # log_var = F.softplus(log_var)


        return [mu, log_var]

    def decode(self, z: Tensor, condition: Tensor, tag='_reconstructed') -> Tensor:

        x = z
        if condition is not None:
            x = torch.cat((z, condition), dim=-1)

        x = self.decoder(x)

        result = {}
        start = 0
        for k,v in sorted(self.data_feats.items()):
            reduced_shape, shape = v
            assert len(shape) <= 2

            if tag != '_reconstructed':
                print("Sampling", k + tag)

            result[k+tag] = x[:, start:start+reduced_shape]
            # print("XXX", k, result[k+tag].shape)
            if len(shape) == 2:
                result[k+tag] = result[k+tag].reshape((len(z), shape[0], shape[1]))
            start += reduced_shape

            if self.sigmoid_on is not None:
                if k.startswith(self.sigmoid_on):
                    result[k + tag] = torch.sigmoid(result[k + tag])
            # reduced_shape = reduce(mul, shape)
            # self.data_feats[k] = (reduced_shape, shape)


        # x = torch.reshape(x, (-1, self.data_feature_dim[0], self.data_feature_dim[1]))
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Multi variate normal sampling
        return eps * std + mu


    def _get_cat_condition(self, data_samples):
        conditions = []
        for k,v in sorted(self.conditional_feats.items()):
            x = data_samples[k].reshape((-1, v))
            assert x.shape[1] == v
            conditions += [x]

        if len(conditions) == 0:
            return None
        cat_c = torch.cat(conditions, dim=1)
        return cat_c

    def _get_feats(self, data_samples):
        # data_samples['momenta'] = torch.normal(15, 2, size=data_samples['momenta'].shape).to('cuda:0')
        # data_samples['momenta'] = self._test_exponential_distribution.sample(data_samples['momenta'].shape).to('cuda:0')

        feats = []
        for k,v in sorted(self.data_feats.items()):
            reduced_shape, shape = v
            x = data_samples[k].reshape((-1, reduced_shape))
            assert x.shape[1] == reduced_shape
            feats += [x]
        cat_c = torch.cat(feats, dim=1)
        return cat_c

    def forward(self, input: dict, **kwargs) -> dict:
        # features = input['momenta_pp']

        features = self._get_feats(input)
        condition = self._get_cat_condition(input)

        mu, log_var = self.encode(features, condition)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z, condition)

        all_dict = {'features': features, 'mu': mu, 'log_var': log_var}
        all_dict.update(decoded)
        return all_dict
        # return {'features_reconstructed': decoded, 'features': features, 'mu': mu, 'log_var': log_var}

    def sample(self,
               num_samples: int,
               current_device: int, data_dict: dict, **kwargs) -> Tensor:

        condition = self._get_cat_condition(data_dict)

        if condition is not None:
            assert condition.shape[0] == num_samples

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z, condition, tag='_sampled')
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]


class Permuter(nn.Module):
    def __init__(self, permuter_type='max_momenta', **kwargs):
        super().__init__()
        self.permuter_type = permuter_type

    def forward(self, data_dict: Dict[str, torch.Tensor]):
        if self.permuter_type == 'max_momenta':
            px = data_dict['dau_px']
            py = data_dict['dau_py']
            pz = data_dict['dau_pz']
            p_mag = torch.sqrt(px**2 + py**2 + pz**2)
            indexing_tensor = torch.argsort(-p_mag, dim=1)
        else:
            raise NotImplementedError('Permuter type not recognized')

        data_dict_2 = {}
        for k, v in data_dict.items():
            if k.startswith('dau_'):
                v2 = torch.gather(v, dim=1, index=indexing_tensor)
                data_dict_2[k] = v2
            else:
                data_dict_2[k] = v
        return data_dict_2