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

Tensor = TypeVar('torch.tensor')


class MlpConditionalEncoder(nn.Module):
    def __init__(self, in_features, out_features, condition_dims):
        super(MlpConditionalEncoder, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features+condition_dims, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fcl = nn.Linear(128, out_features)

    def forward(self, x, c):
        assert len(x.shape) == 2
        assert len(c.shape) == 2

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fcl(x)
        return x


class MlpConditionalDecoder(nn.Module):
    def __init__(self, in_features, out_features, condition_dims):
        super(MlpConditionalDecoder, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features+condition_dims, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fcl = nn.Linear(256, out_features)

    def forward(self, x, c):
        x = torch.cat((x, c), dim=-1)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        x = F.relu(x)

        x = self.fcl(x)
        return x



class MlpConditionalVAE(BaseVAE):
    def __init__(self, latent_dim=64, data_feature_dim=None, conditional_feats=None, **kwargs):
        super().__init__()

        if data_feature_dim is None:
            data_feature_dim = [3, 3]

        self.latent_dim = latent_dim
        self.data_feature_dim = data_feature_dim
        reduced_data_feature_dim = reduce(mul, self.data_feature_dim)

        if conditional_feats is None:
            conditional_feats = {'momenta_mother_pp': [1, 3]}

        assert type(conditional_feats) is dict

        self.conditional_feats = {}
        reduced_conditional_dim = 0
        for k,v in sorted(conditional_feats.items()):
            reduced_dims = reduce(mul, v)
            self.conditional_feats[k] = reduced_dims
            reduced_conditional_dim += reduced_dims

        self.encoder = MlpConditionalEncoder(in_features=reduced_data_feature_dim, out_features=self.latent_dim * 4, condition_dims=reduced_conditional_dim)
        self.decoder = MlpConditionalDecoder(in_features=latent_dim, out_features=reduced_data_feature_dim, condition_dims=reduced_conditional_dim)


        self.fc_mu = nn.Linear(self.latent_dim*4, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim*4, self.latent_dim)

    def encode(self, x: Tensor, c: Tensor) -> List[Tensor]:
        assert len(x.shape) == 3

        assert x.shape[1] == self.data_feature_dim[0]
        assert x.shape[2] == self.data_feature_dim[1]

        x = torch.reshape(x, (-1, x.shape[1]*x.shape[2]))
        x = torch.cat((x, c), dim=-1)

        result = self.encoder(x, c)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor, condition: Tensor) -> Tensor:
        x = self.decoder(z, condition)

        x = torch.reshape(x, (-1, self.data_feature_dim[0], self.data_feature_dim[1]))
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def _get_cat_condition(self, data_samples):
        conditions = []
        for k,v in sorted(self.conditional_feats.items()):
            x = data_samples[k].reshape((-1, v))
            assert x.shape[1] == v
            conditions += [x]

        return torch.cat(conditions, dim=1)

    def forward(self, input: dict, **kwargs) -> dict:
        features = input['momenta_pp']

        condition = self._get_cat_condition(input)

        mu, log_var = self.encode(features, condition)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z, condition)

        return {'momenta_reconstructed': decoded, 'features': features, 'mu': mu, 'log_var': log_var}

    def loss_function(self,
                      results:dict,
                      **kwargs) -> dict:

        recons = results['momenta_reconstructed']
        input = results['features']
        mu = results['mu']
        log_var = results['log_var']


        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'loss_reco': recons_loss.detach(), 'loss_kld': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, data_dict: dict, **kwargs) -> Tensor:

        condition = self._get_cat_condition(data_dict)

        assert condition.shape[0] == num_samples

        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z, condition)
        return {'momenta_sampled': samples}

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]