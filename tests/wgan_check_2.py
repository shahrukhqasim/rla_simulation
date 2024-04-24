import os

import matplotlib.pyplot as plt
import numpy as np
import torch.distributions
import torch.optim as optim
from datetime import datetime

learning_rate = 5e-5
learning_rate_critic = 5e-5
num_iterations = 10 ** 6
batch_size = 8192
noise_dim = 3
device = 'mps'
weight_clip = 0.01
current_time = datetime.now()
random_string = current_time.strftime("%Y%m%d%H%M%S%f")[:-3]
output_dir = 'logs/wgan_check/'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = output_dir+random_string
os.mkdir(output_dir)
to_pdf=True
after = 40

data_dim = 4

from rlasim.lib.networks_gan import MlpConditionalWGAN

network = MlpConditionalWGAN(noise_dim=noise_dim, data_feats={'x': [data_dim]}).to(device)
gen = network.gen
critic = network.critic


# opt_gen = optim.RMSprop(gen.parameters(), lr=learning_rate)
# opt_critic = optim.RMSprop(critic.parameters(), lr=learning_rate_critic)
opt_gen = optim.Adam(gen.parameters(), lr=learning_rate)
opt_critic = optim.Adam(critic.parameters(), lr=learning_rate)


means = np.random.uniform(-500, 500, size=(data_dim,))
means = torch.tensor(means)
# means = means*0.0 + 1

distribution_a = torch.distributions.Normal(loc=means, scale=(means*0+1))

# base_dist = torch.distributions.Uniform(0,1)
base_dist = torch.distributions.Normal(0,1)
critic_iterations = 5

enable_gp = True
lambda_gp = 10.

for it in range(num_iterations):
    batch_dict = distribution_a.sample((batch_size,)).float().to(device)
    batch_dict = {'x':batch_dict}

    assert critic_iterations >= 1
    for _ in range(critic_iterations):
        # fake = gen(noise)
        # critic_real = critic(real).reshape(-1)
        # critic_fake = critic(fake).reshape(-1)

        sampled_data, sampled_dict = network.generate(batch_dict, base_dist)
        critic_sampled = network.criticize(sampled_data, sampled_dict['condition'])
        critic_real = network.criticize(sampled_dict['features'], sampled_dict['condition'])

        the_gradient_penalty = network.compute_gradient_penalty(sampled_dict['features'], sampled_data, sampled_dict['condition'])

        loss_critic = -(torch.mean(critic_real) - torch.mean(critic_sampled))
        if enable_gp:
            loss_critic += the_gradient_penalty * lambda_gp

        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        if not enable_gp: # If its weight clamping method
            for p in critic.parameters():
                p.data.clamp_(-weight_clip, weight_clip)

    output = critic(sampled_data).reshape(-1)
    loss_gen = -torch.mean(output) # TODO: Not sure if there should be a minus sign here.
    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    if it % after == 0:
        N = 40000
        sampled_data = network.sample(N, base_dist, {}, device=device)[0].detach().cpu().numpy()
        real_data = distribution_a.sample((N,)).cpu().numpy()

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        for i in range(4):
            # Plot histogram for sampled_data
            axs[i // 2, i % 2].hist(sampled_data[:, i], bins=30, alpha=0.5, label='Sampled Data')

            # Plot histogram for real_data
            axs[i // 2, i % 2].hist(real_data[:, i], bins=30, alpha=0.5, label='Real Data')

            # Add labels and title
            axs[i // 2, i % 2].set_xlabel('Value')
            axs[i // 2, i % 2].set_ylabel('Frequency')
            axs[i // 2, i % 2].set_title(f'Histogram of Sampled Data and Real Data (Column {i})')
            axs[i // 2, i % 2].legend()

        plt.tight_layout()

        if to_pdf:
            plt.savefig(os.path.join(output_dir, 'it_%05d.pdf'%it))
            plt.close(fig)
        else:
            plt.show()

        print(sampled_data.shape, real_data.shape)


    print("Iteration", it, float(loss_gen), float(loss_critic), float(the_gradient_penalty))
