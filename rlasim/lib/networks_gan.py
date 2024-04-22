import torch
from torch import nn
from torch.nn import functional as F
from functools import reduce
from typing import List, Callable, Union, Any, TypeVar, Tuple
from operator import mul
from rlasim.lib.organise_data import compute_dalitz_masses, compute_parent_mass

Tensor = TypeVar('torch.tensor')



class MlpGanGenerator(nn.Module):
    def __init__(self, in_features, out_features, condition_dims):
        super(MlpGanGenerator, self).__init__()
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

class MlpGanCritic(nn.Module):
    def __init__(self, in_features, out_features, condition_dims):
        super(MlpGanCritic, self).__init__()
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



class MlpGan(nn.Module):
    def __init__(self, n_z_dim, n_features):
        super().__init__()


class MlpConditionalWGAN(nn.Module):
    def __init__(self, latent_dim=64, conditional_feats=None, data_feats=None, **kwargs):
        super().__init__()


        self.latent_dim = latent_dim

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

        self.gen = MlpGanGenerator(in_features=self.latent_dim, out_features=reduced_data_feature_dim, condition_dims=reduced_conditional_dim)
        self.critic = MlpGanCritic(in_features=reduced_data_feature_dim, out_features=self.latent_dim, condition_dims=reduced_conditional_dim)


    def criticize(self, x: Tensor, c: Tensor) -> List[Tensor]:
        # assert len(x.shape) == 3
        # assert x.shape[1] == self.data_feature_dim[0]
        # assert x.shape[2] == self.data_feature_dim[1]

        # x = torch.reshape(x, (-1, x.shape[1]*x.shape[2]))
        if c is not None:
            x = torch.cat((x, c), dim=-1)

        result = self.critic(x)

        return result

    def gen_w(self, z: Tensor, condition: Tensor, tag='_reconstructed') -> Tensor:
        x = z
        if condition is not None:
            x = torch.cat((z, condition), dim=-1)
        x = self.gen(x)

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

        return x, result


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
        feats = []
        for k,v in sorted(self.data_feats.items()):
            reduced_shape, shape = v
            x = data_samples[k].reshape((-1, reduced_shape))
            assert x.shape[1] == reduced_shape
            feats += [x]
        cat_c = torch.cat(feats, dim=1)
        return cat_c

    def forward(self, input: dict, base_dist:torch.distributions.Distribution, device=None, **kwargs) -> dict:

        features = self._get_feats(input)
        condition = self._get_cat_condition(input)
        if device is None:
            device = features.device
        condition = condition.to(device)

        sampled, sampled_dict = self.sample(len(features), base_dist, input, tag='_sampled', device=device)

        print(sampled.shape, features.shape, self.data_feats, self.conditional_feats)
        critic_output_von_sampled = self.criticize(sampled, condition)
        critic_output_von_real = self.criticize(features, condition)


        # TODO: Put everything in here!
        all_dict = {'features': features,
                    'critic_real': critic_output_von_sampled,
                    'critic_sampled': critic_output_von_real}
        all_dict.update(sampled_dict)

        return all_dict
        # return {'features_reconstructed': decoded, 'features': features, 'mu': mu, 'log_var': log_var}

    def sample(self,
               num_samples: int,
               base_dist : torch.distributions.Distribution,
               data_dict: dict,device=None,tag='_sampled', **kwargs) -> Tensor:

        condition = self._get_cat_condition(data_dict)

        if condition is not None:
            assert condition.shape[0] == num_samples

        if device is None:
            device = condition.device

        z = base_dist.sample((num_samples, self.latent_dim)).to(device)

        samples, samples_dict = self.gen_w(z, condition, tag=tag)

        return samples, samples_dict


class ThreeBodyGanLoss(nn.Module):
    def __init__(self, secondary_weight=0.0, dalitz_weight=0.0, parent_mass_weight=0.0, loss_type='mse', truth_var='momenta', predicted_var='momenta_reconstructed', **kwards):
        super().__init__()

        self.secondary_weight = secondary_weight

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

        critic_real = sample['critic_real'].reshape(-1)
        critic_fake = sample['critic_sampled'].reshape(-1)
        loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake))
        the_dict = {'loss':loss_critic}


        input = sample[self.truth_var]

        # kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        # recons2 = torch.nan_to_num(recons)
        # recons_loss = distance_func(recons2, input)


        loss_secondary = torch.tensor(0.0)

        one = float(distance_func(recons[:, :, 0:2], input[:, :, 0:2]))
        two = float(distance_func(recons[:, :, 2], input[:, :, 2]))

        # the_dict['loss_reco_p'] = recons_loss

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
            loss_secondary += dalitz_loss * self.dalitz_weight
        if self.parent_mass_weight > 0.0:
            loss_secondary += parent_mass_loss * self.parent_mass_weight

        # print("XXA", torch.mean(input, dim=0).detach().cpu().numpy(), torch.mean(recons2, dim=0).detach().cpu().numpy())
        # print("XXB", torch.max(input, dim=0)[0].detach().cpu().numpy(), torch.max(recons2, dim=0)[0].detach().cpu().numpy())

        secondary_weight = self.secondary_weight
        print("XXX", float(loss_critic), float(loss_secondary), float(dalitz_loss), float(torch.mean(parent_mass)), float(torch.mean(parent_mass_decoded)), one, two, float(torch.mean(mass_13)), float(torch.mean(mass_13_decoded)), float(torch.mean(mass_32)), float(torch.mean(mass_32_decoded)))

        loss = loss_critic + secondary_weight * loss_secondary

        the_dict.update({'loss': loss, 'loss_secondary': loss_secondary.detach(), 'loss_critic': loss_critic.detach()})
        return the_dict