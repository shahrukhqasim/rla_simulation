# Adapted from
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
from functools import reduce

import numpy as np
import torch

from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch.nn import functional as F
from rlasim.lib.networks import BaseVAE
from operator import mul

from rlasim.lib.organise_data import compute_dalitz_masses, compute_parent_mass

Tensor = TypeVar('torch.tensor')

class MlpConditionalEncoder2(nn.Module):
    def __init__(self, in_features, out_features, condition_dims):
        super(MlpConditionalEncoder2, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features+condition_dims, 1024)  # 5*5 from image dimension
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fcl = nn.Linear(1024, out_features)

        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        assert len(x.shape) == 2

        x = self.fc1(x)
        # x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        # x = self.bn1(x)

        x = self.fc3(x)
        x = F.relu(x)

        # x = self.bn2(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        x = self.fc6(x)
        x = F.relu(x)

        x = self.fcl(x)
        return x

class MlpConditionalEncoder(nn.Module):
    def __init__(self, in_features, out_features, condition_dims):
        super(MlpConditionalEncoder, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features+condition_dims, 1024)  # 5*5 from image dimension
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fcl = nn.Linear(1024, out_features)

        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        assert len(x.shape) == 2

        x = self.fc1(x)
        # x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        # x = self.bn1(x)

        x = self.fc3(x)
        x = F.relu(x)

        # x = self.bn2(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        x = self.fc6(x)
        x = F.relu(x)

        x = self.fcl(x)
        return x


class MlpConditionalDecoder(nn.Module):
    def __init__(self, in_features, out_features, condition_dims):
        super(MlpConditionalDecoder, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features+condition_dims, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 1024)
        self.fc5 = nn.Linear(1024, 1024)
        self.fc6 = nn.Linear(1024, 1024)
        self.fcl = nn.Linear(1024, out_features)

        # self.bn1 = nn.BatchNorm1d(1024)
        # self.bn2 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        # x = self.bn1(x)

        x = self.fc3(x)
        x = F.relu(x)

        # x = self.bn2(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fc5(x)
        x = F.relu(x)

        x = self.fc6(x)
        x = F.relu(x)

        x = self.fcl(x)
        return x


class ThreeBodyVaeLoss(nn.Module):
    def __init__(self, kld_weight, dalitz_weight=0.0, parent_mass_weight=0.0, loss_type='mse', truth_var='momenta', predicted_var='momenta_reconstructed', **kwards):
        super().__init__()

        self.kld_weight = kld_weight

        self.loss_type = loss_type
        self.predicted_var = predicted_var
        self.truth_var = truth_var
        self.dalitz_weight = dalitz_weight
        self.parent_mass_weight = parent_mass_weight


    def forward(self, sample, iteration=-1):
        recons = sample[self.predicted_var]

        if self.loss_type == 'mse':
            distance_func = F.mse_loss
        elif self.loss_type == 'huber':
            distance_func = F.huber_loss
        else:
            raise ValueError('Unknown value of loss_type')

        input = sample[self.truth_var]
        mu = sample['mu']
        log_var = sample['log_var']

        # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons2 = torch.nan_to_num(recons)

        recons_loss = distance_func(recons2, input)
        the_dict = {}

        one = float(distance_func(recons[:, :, 0:2], input[:, :, 0:2]))
        two = float(distance_func(recons[:, :, 2], input[:, :, 2]))

        the_dict['loss_reco_p'] = recons_loss

        mass_13, mass_32 = compute_dalitz_masses(sample, '', reco_var=self.predicted_var)
        parent_mass = compute_parent_mass(sample, '', reco_var=self.predicted_var)
        parent_mass_decoded = compute_parent_mass(sample, '_DECODED', reco_var=self.predicted_var)
        mass_13_decoded, mass_32_decoded = compute_dalitz_masses(sample, '_DECODED', squared=True, reco_var=self.predicted_var)
        # print(mass_13, mass_13_decoded, mass_32, mass_32_decoded)
        # dalitz_loss = (F.mse_loss(mass_13, mass_13_decoded) + F.mse_loss(mass_32, mass_32_decoded))*0.001
        dalitz_loss = (distance_func(mass_13, mass_13_decoded) + distance_func(mass_32, mass_32_decoded))
        parent_mass_loss = distance_func(parent_mass, parent_mass_decoded)
        the_dict['loss_dalitz'] = dalitz_loss
        if self.dalitz_weight > 0.0:
            recons_loss += dalitz_loss * self.dalitz_weight
        if self.parent_mass_weight > 0.0:
            recons_loss += parent_mass_loss * self.parent_mass_weight

        print("XXA", torch.mean(input, dim=0).detach().cpu().numpy(), torch.mean(recons2, dim=0).detach().cpu().numpy())
        print("XXB", torch.max(input, dim=0)[0].detach().cpu().numpy(), torch.max(recons2, dim=0)[0].detach().cpu().numpy())

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_weight = self.kld_weight
        print("XXX", float(recons_loss), float(kld_loss), float(dalitz_loss), float(torch.mean(parent_mass)), float(torch.mean(parent_mass_decoded)), one, two, float(torch.mean(mass_13)), float(torch.mean(mass_13_decoded)), float(torch.mean(mass_32)), float(torch.mean(mass_32_decoded)))

        loss = recons_loss + kld_weight * kld_loss

        the_dict.update({'loss': loss, 'loss_reco': recons_loss.detach(), 'loss_kld': -kld_loss.detach()})
        return the_dict


class MlpConditionalVAE(BaseVAE):
    def __init__(self, latent_dim=64, conditional_feats=None, data_feats=None, **kwargs):
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

        self.encoder = MlpConditionalEncoder(in_features=reduced_data_feature_dim, out_features=self.latent_dim * 4, condition_dims=reduced_conditional_dim)
        self.decoder = MlpConditionalDecoder(in_features=latent_dim, out_features=reduced_data_feature_dim, condition_dims=reduced_conditional_dim)


        self.fc_mu = nn.Linear(self.latent_dim*4, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim*4, self.latent_dim)

        self._test_exponential_distribution = torch.distributions.Exponential(0.05)

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
            # reduced_shape = reduce(mul, shape)
            # self.data_feats[k] = (reduced_shape, shape)


        # x = torch.reshape(x, (-1, self.data_feature_dim[0], self.data_feature_dim[1]))
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        # return mu + logvar # TODO: Comment this
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
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

    # def loss_function(self,
    #                   results:dict,
    #                   **kwargs) -> dict:
    #
    #     recons = results['features_reconstructed']
    #     input = results['features']
    #     mu = results['mu']
    #     log_var = results['log_var']
    #
    #
    #     kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
    #
    #     recons_loss = F.mse_loss(recons, input)
    #
    #     kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    #
    #     loss = recons_loss + kld_weight * kld_loss
    #     return {'loss': loss, 'loss_reco': recons_loss.detach(), 'loss_kld': -kld_loss.detach()}

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