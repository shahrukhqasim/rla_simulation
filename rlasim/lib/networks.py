# Adapted from
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py


import numpy as np
import torch
from torch import nn

from torch import nn
from abc import abstractmethod
from typing import List, Callable, Union, Any, TypeVar, Tuple
from torch.nn import functional as F

Tensor = TypeVar('torch.tensor')

class BaseVAE(nn.Module):

    def __init__(self, **kwargs) -> None:
        super(BaseVAE, self).__init__(**kwargs)

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class MlpEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(MlpEncoder, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fcl = nn.Linear(128, out_features)

    def forward(self, x):
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



class MlpDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(MlpDecoder, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(in_features, 128)  # 5*5 from image dimension
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fcl = nn.Linear(256, out_features)

    def forward(self, x):
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



class VanillaVae(BaseVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = 64
        self.data_feature_dims = 9

        self.encoder = MlpEncoder(in_features=self.data_feature_dims, out_features=self.latent_dim*4)
        self.decoder = MlpDecoder(in_features=self.latent_dim, out_features=self.data_feature_dims)


        self.fc_mu = nn.Linear(self.latent_dim*4, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim*4, self.latent_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder(z)
        return x

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        decoded = self.decode(z)
        return [decoded, input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:

        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]



        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        # with torch.no_grad():
        #     y = input.cpu().numpy()
        #     y = y.reshape(-1, 9)
        #     y = np.sum(y, axis=1)
        #
        #     x = recons.cpu().numpy()
        #     x = x.reshape(-1, 9)
        #     x = np.sum(x, axis=1)
        #     print("X", np.min(x), np.mean(x), np.max(x), np.min(y), np.mean(y), np.max(y))

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'loss_reco': recons_loss.detach(), 'loss_kld': -kld_loss.detach()}

    def sample(self,
               num_samples: int,
               current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]