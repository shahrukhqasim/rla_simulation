import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from matplotlib.colors import LogNorm

from rlasim.lib.plt_settings import set_sizing




class ThreeBodyDecayPlotter:
    def __init__(self, ranges=None, unit=None, **kwargs):
        # set_sizing()
        self.ranges = ranges

        self.range_min = {0:self.ranges['px_min'], 1:self.ranges['py_min'], 2:self.ranges['pz_min']}
        self.range_max = {0:self.ranges['px_max'], 1:self.ranges['py_max'], 2:self.ranges['pz_max']}
        self.unit = {0:'', 1:'', 2:''}
        if unit is not None:
            if type(unit) is str:
                if len(unit) > 0:
                    self.unit = {0:' %s'%unit, 1:' %s'%unit, 2:' %s'%unit}
            else:
                NotImplementedError('Non str units not implemented')

    def _make_histo_three_body_prods(self, samples):
        fig, axg = plt.subplots(3, 3, figsize=(9, 7))

        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.1)
        fig.suptitle('Decay products', fontsize=20)
        fig.tight_layout(pad=2.0)

        axl_dict = {0: 'x', 1: 'y', 2: 'z'}
        for p in range(3):
            for i in range(3):
                ax = axg[p][i]
                dat = samples[:, p, i]
                dat[dat < self.range_min[i]] = self.range_min[i]
                dat[dat > self.range_max[i]] = self.range_max[i]
                hist, bins = np.histogram(dat, bins=50, range=(self.range_min[i], self.range_max[i]))
                bin_centers = (bins[:-1] + bins[1:]) / 2
                ax.step(bin_centers, hist, where='mid', color='tab:red', linewidth=1)
                ax.set_yscale('log')
                ax.set_xlabel('${p_%d}_%s$%s' % (p + 1, axl_dict[i],self.unit[i]))
                ax.set_ylabel('Frequency')

        return fig, axg

    def _make_2d_histo_three_body_prods(self, samples):
        fig, axg = plt.subplots(3, 3, figsize=(9, 7))

        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.1)
        fig.suptitle('Decay products', fontsize=20)

        combos = [(0, 1), (0, 2), (1, 2)]

        axl_dict = {0: 'x', 1: 'y', 2: 'z'}
        for p in range(3):
            for i in range(3):
                ax = axg[p][i]

                samples_x = samples[:, p, combos[i][0]]*1
                samples_y = samples[:, p, combos[i][1]]*1

                samples_x[samples_x < self.range_min[combos[i][0]]] = self.range_min[combos[i][0]]
                samples_x[samples_x > self.range_max[combos[i][0]]] = self.range_max[combos[i][0]]

                samples_y[samples_y < self.range_min[combos[i][1]]] = self.range_min[combos[i][1]]
                samples_y[samples_y > self.range_max[combos[i][1]]] = self.range_max[combos[i][1]]

                h = ax.hist2d(samples_x,
                              samples_y,
                              range=[
                                  [self.range_min[combos[i][0]], self.range_max[combos[i][0]]],
                                  [self.range_min[combos[i][1]], self.range_max[combos[i][1]]]
                              ],
                              norm=LogNorm(),
                              bins=30)
                fig.colorbar(h[3], ax=ax)
                ax.set_xlabel('${p_%d}_%s$%s' % (p + 1, axl_dict[combos[i][0]], self.unit[combos[i][0]]))
                ax.set_ylabel('${p_%d}_%s$%s' % (p + 1, axl_dict[combos[i][1]], self.unit[combos[i][1]] ))
                # ax.set_ylabel('Frequency')
        fig.tight_layout(pad=2.0)
        return fig, axg

    def _make_histo_mother(self, samples):
        fig, axg = plt.subplots(1, 3, figsize=(9, 3))
        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.1)
        fig.suptitle('Mother', fontsize=20)
        fig.tight_layout(pad=2.0)

        axl_dict = {0: 'x', 1: 'y', 2: 'z'}
        for i in range(3):
            ax = axg[i]
            dat = samples[:, 0, i]
            dat[dat < self.range_min[i]] = self.range_min[i]
            dat[dat > self.range_max[i]] = self.range_max[i]
            hist, bins = np.histogram(dat, bins=50, range=(self.range_min[i], self.range_max[i]))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.step(bin_centers, hist, where='mid', color='tab:red', linewidth=1)
            ax.set_yscale('log')
            ax.set_xlabel('${p_m}_%s$%s' % (axl_dict[i], self.unit[i]))
            ax.set_ylabel('Frequency')

        return fig, axg

    def _make_hist_pz_pz(self, samples, samples_mother):
        fig, axg = plt.subplots(3, 1, figsize=(4, 9))
        fig.suptitle('Mother', fontsize=20)
        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.1)
        for i in range(3):
            ax = axg[i]
            samples_x = samples_mother[:, 0, 2]*1
            samples_y = samples[:, i, 2].reshape(-1)*1

            samples_x[samples_x < self.range_min[2]] = self.range_min[2]
            samples_x[samples_x > self.range_max[2]] = self.range_max[2]

            samples_y[samples_y < self.range_min[2]] = self.range_min[2]
            samples_y[samples_y > self.range_max[2]] = self.range_max[2]

            h = ax.hist2d(samples_x,
                          samples_y,
                          range=[
                              [self.range_min[2], self.range_max[2]],
                              [self.range_min[2], self.range_max[2]]
                          ],
                          norm=LogNorm(),
                          bins=30)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel('${p_m}_z$%s'%(self.unit[2]))
            ax.set_ylabel('${p_%d}_z$%s' % (i + 1, self.unit[2]))

        fig.tight_layout(pad=2.0)

        return fig, axg

    def plot(self, samples, samples_mother, file):
        if torch.is_tensor(samples):
            samples = samples.numpy()
        if torch.is_tensor(samples_mother):
            samples_mother = samples_mother.numpy()

        assert len(samples.shape) == 3
        assert len(samples_mother.shape) == 3
        assert samples.shape[1] == 3
        assert samples.shape[2] == 3
        assert samples_mother.shape[2] == 3
        assert samples_mother.shape[1] == 1

        with PdfPages(file) as pdf:
            fig, ax = self._make_2d_histo_three_body_prods(samples)
            pdf.savefig(fig)
            fig.clear()

            fig, ax = self._make_histo_three_body_prods(samples)
            pdf.savefig(fig)
            fig.clear()

            fig, ax = self._make_histo_mother(samples_mother)
            pdf.savefig(fig)
            fig.clear()

            fig, ax = self._make_hist_pz_pz(samples, samples_mother)
            pdf.savefig(fig)
            fig.clear()
