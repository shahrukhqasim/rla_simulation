import matplotlib.pyplot as plt




for particle in range(3):
    plt.figure(figsize=(12, 4))
    subplot_idx = 0
    for i in range(3):
        for j in range(i + 1, 3):
            subplot_idx += 1
            plt.subplot(1, 3, subplot_idx)

            print(particle, i, j)

            # plt.hist2d(samples[:, particle, i], samples[:, particle, j], bins=15, norm=LogNorm(),
            #            range=[[-1, 1], [-1, 1]])
