import matplotlib.pyplot as plt
import numpy as np
import torch.distributions
import torch.optim as optim

learning_rate = 5e-5
num_iterations = 10 ** 6
batch_size = 8192
noise_dim = 3
device = 'mps'
weight_clip = 0.01

data_dim = 4

from rlasim.lib.networks_gan import MlpConditionalWGAN

network = MlpConditionalWGAN(noise_dim=noise_dim, data_feats={'x': [data_dim]}).to(device)
gen = network.gen
critic = network.critic


opt_gen = optim.RMSprop(gen.parameters(), lr=learning_rate)
opt_critic = optim.RMSprop(critic.parameters(), lr=learning_rate)


means = np.random.uniform(1, 10, size=(data_dim,))
means = torch.tensor(means)

distribution_a = torch.distributions.Normal(loc=means, scale=(means*0+1.0))

base_dist = torch.distributions.Uniform(0,1)
critic_iterations = 1

for it in range(num_iterations):
    batch_dict = distribution_a.sample((batch_size,)).float().to(device)
    batch_dict = {'x':batch_dict}

    assert critic_iterations >= 1
    for _ in range(critic_iterations):
        noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

        # fake = gen(noise)
        # critic_real = critic(real).reshape(-1)
        # critic_fake = critic(fake).reshape(-1)

        o_dict = network.forward(batch_dict, base_dist)

        # print(o_dict.keys())

        critic_real = o_dict['critic_real']
        critic_sampled = o_dict['critic_sampled']
        fake = o_dict['x_sampled']


        loss_critic = -(torch.mean(critic_real) - torch.mean(critic_sampled))
        critic.zero_grad()
        loss_critic.backward(retain_graph=True)
        opt_critic.step()

        for p in critic.parameters():
            p.data.clamp_(-weight_clip, weight_clip)

    output = critic(fake).reshape(-1)
    loss_gen = torch.mean(output) # TODO: Not sure if there should be a minus sign here.
    gen.zero_grad()
    loss_gen.backward()
    opt_gen.step()

    if it % 10 == 0:
        sampled_data = network.sample(1000, base_dist, {}, device=device)[0].detach().cpu().numpy()
        real_data = distribution_a.sample((1000,)).cpu().numpy()

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
        # Show plot
        plt.show()

        print(sampled_data.shape, real_data.shape)


    print("Iteration", it, float(loss_gen), float(loss_critic))
