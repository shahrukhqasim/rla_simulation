import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
from matplotlib.colors import LogNorm

from rlasim.lib.data_core import tensors_dict_join
from rlasim.lib.plt_settings import set_sizing

import re


class ThreeBodyDecayPlotter:
    def __init__(self, ranges=None, unit=None, conditions=None, **kwargs):
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

        self.conditions = conditions
        if self.conditions is not None:
            assert type(self.conditions) is list

    def _make_histo_three_body_prods(self, samples, str='Decay products'):
        fig, axg = plt.subplots(3, 3, figsize=(9, 7))

        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.1)
        fig.suptitle(str, fontsize=16)
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


    def _make_three_body_prods_hist_diff(self, samples, samples_2, str='Decay products', ylabel='true', ylabel_2='sampled'):
        fig, axg = plt.subplots(3, 3, figsize=(9, 7))

        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.15)
        fig.suptitle(str, fontsize=16)
        fig.tight_layout(pad=2.5)

        axl_dict = {0: 'x', 1: 'y', 2: 'z'}
        for p in range(3):
            for i in range(3):
                ax = axg[p][i]

                dat = samples[:, p, i]
                dat[dat < self.range_min[i]] = self.range_min[i]
                dat[dat > self.range_max[i]] = self.range_max[i]

                dat_2 = samples_2[:, p, i]
                dat_2[dat_2 < self.range_min[i]] = self.range_min[i]
                dat_2[dat_2 > self.range_max[i]] = self.range_max[i]

                hist, bins = np.histogram(dat, bins=50, range=(self.range_min[i], self.range_max[i]))
                hist2, _ = np.histogram(dat_2, bins=50, range=(self.range_min[i], self.range_max[i]))

                bin_centers = (bins[:-1] + bins[1:]) / 2

                ax.step(bin_centers, np.log10(hist2/hist), where='mid', color='tab:red', linewidth=1)

                # ax.set_yscale('log')
                ax.set_xlabel('${p_%d}_%s$%s' % (p + 1, axl_dict[i],self.unit[i]))
                ax.set_ylabel('log10$(N_{\\mathrm{%s}}/N_{\\mathrm{%s}})$'%(ylabel_2, ylabel))

        return fig, axg

    def _make_2d_histo_three_body_prods(self, samples, str='Decay products'):
        fig, axg = plt.subplots(3, 3, figsize=(9, 7))

        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.1)

        fig.suptitle(str, fontsize=16)

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

    def _make_2d_three_body_prods_hist_diff(self, samples, samples_2, str='Sampled vs true distr. diff.'):
        fig, axg = plt.subplots(3, 3, figsize=(9, 7))

        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.1)

        fig.suptitle(str, fontsize=16)

        combos = [(0, 1), (0, 2), (1, 2)]

        axl_dict = {0: 'x', 1: 'y', 2: 'z'}
        for p in range(3):
            for i in range(3):
                ax = axg[p][i]

                samples_x = samples[:, p, combos[i][0]]*1
                samples_y = samples[:, p, combos[i][1]]*1

                samples_2_x = samples_2[:, p, combos[i][0]]*1
                samples_2_y = samples_2[:, p, combos[i][1]]*1

                samples_x[samples_x < self.range_min[combos[i][0]]] = self.range_min[combos[i][0]]
                samples_x[samples_x > self.range_max[combos[i][0]]] = self.range_max[combos[i][0]]

                samples_y[samples_y < self.range_min[combos[i][1]]] = self.range_min[combos[i][1]]
                samples_y[samples_y > self.range_max[combos[i][1]]] = self.range_max[combos[i][1]]

                samples_2_x[samples_2_x < self.range_min[combos[i][0]]] = self.range_min[combos[i][0]]
                samples_2_x[samples_2_x > self.range_max[combos[i][0]]] = self.range_max[combos[i][0]]

                samples_2_y[samples_2_y < self.range_min[combos[i][1]]] = self.range_min[combos[i][1]]
                samples_2_y[samples_2_y > self.range_max[combos[i][1]]] = self.range_max[combos[i][1]]

                H, x_edges, y_edges = np.histogram2d(samples_x,
                              samples_y,
                              range=[
                                  [self.range_min[combos[i][0]], self.range_max[combos[i][0]]],
                                  [self.range_min[combos[i][1]], self.range_max[combos[i][1]]]
                              ],
                              bins=30)
                H2, _, _ = np.histogram2d(samples_2_x,
                              samples_2_y,
                              range=[
                                  [self.range_min[combos[i][0]], self.range_max[combos[i][0]]],
                                  [self.range_min[combos[i][1]], self.range_max[combos[i][1]]]
                              ],
                              bins=30)

                ax.set_xlabel('${p_%d}_%s$%s' % (p + 1, axl_dict[combos[i][0]], self.unit[combos[i][0]]))
                ax.set_ylabel('${p_%d}_%s$%s' % (p + 1, axl_dict[combos[i][1]], self.unit[combos[i][1]] ))

                extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

                divergence = np.log10(H2/H)
                # divergence[np.logical_not(np.isreal(divergence))] = 0?

                # ax.imshow(divergence, extent=extent, origin='lower', aspect='auto', cmap='viridis')
                fig.colorbar(
                    ax.imshow(divergence.T, extent=extent, origin='lower', aspect='auto', cmap='PiYG',
                    vmin=-3, vmax=+3)
                )

                # ax.set_ylabel('Frequency')
        fig.tight_layout(pad=2.0)
        return fig, axg

    def _make_histo_mother(self, samples, str='Mother'):
        fig, axg = plt.subplots(1, 3, figsize=(9, 3))
        plt.subplots_adjust(left=0.12,
                            bottom=0.13,
                            right=0.95,
                            top=0.9,
                            wspace=0.3,
                            hspace=0.1)
        fig.suptitle(str, fontsize=16)
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

    def _make_hist_pz_pz(self, samples, samples_mother, str='Mother vs Decay'):
        fig, axg = plt.subplots(1, 3, figsize=(9, 4))
        fig.suptitle(str, fontsize=16)
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

    def _make_pz_pz_hist_diff(self, samples, samples_2, samples_mother, str='Mother vs Decay'):
        fig, axg = plt.subplots(1, 3, figsize=(9, 4))
        fig.suptitle(str, fontsize=16)
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
            samples_y_2 = samples_2[:, i, 2].reshape(-1)*1

            samples_x[samples_x < self.range_min[2]] = self.range_min[2]
            samples_x[samples_x > self.range_max[2]] = self.range_max[2]

            samples_y[samples_y < self.range_min[2]] = self.range_min[2]
            samples_y[samples_y > self.range_max[2]] = self.range_max[2]

            samples_y_2[samples_y_2 < self.range_min[2]] = self.range_min[2]
            samples_y_2[samples_y_2 > self.range_max[2]] = self.range_max[2]

            H, x_edges, y_edges = np.histogram2d(samples_x,
                                                 samples_y,
                                                 range=[
                                                     [self.range_min[2], self.range_max[2]],
                                                     [self.range_min[2], self.range_max[2]]
                                                 ],
                                                 bins=30)

            H2, _, _ = np.histogram2d(samples_x,
                                                 samples_y_2,
                                                 range=[
                                                     [self.range_min[2], self.range_max[2]],
                                                     [self.range_min[2], self.range_max[2]]
                                                 ],
                                                 bins=30)

            extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

            divergence = np.log10(H2 / H)
            # divergence[np.logical_not(np.isreal(divergence))] = 0?

            # ax.imshow(divergence, extent=extent, origin='lower', aspect='auto', cmap='viridis')
            fig.colorbar(
                ax.imshow(divergence.T, extent=extent, origin='lower', aspect='auto', cmap='PiYG',
                          vmin=-3, vmax=+3)
            )

            ax.set_xlabel('${p_m}_z$%s'%(self.unit[2]))
            ax.set_ylabel('${p_%d}_z$%s' % (i + 1, self.unit[2]))

        fig.tight_layout(pad=2.0)

        return fig, axg

    def _do_plot(self, file, data_samples_, condition=None):
        if condition is not None:
            def replace_text_with_data(match):
                var_name = match.group(1)
                return f"data_samples_['{var_name}']"

            if type(condition) is list:
                assert len(condition) == 2
                condition_code_, condition_render_text = condition
            condition_code = re.sub(r"\{(\w+)\}", replace_text_with_data, condition_code_)

            condition_evaluated = eval(condition_code)
            print(np.unique(data_samples_['pdgid_particle_1']))
            print(np.unique(data_samples_['pdgid_particle_2']))
            print(np.unique(data_samples_['pdgid_particle_3']))
            print(condition_code, np.mean(condition_evaluated))
            condition_render_text += ' [%.2f%%]'%(np.mean(condition_evaluated)*100.0)
            data_samples = {}
            for k,v in data_samples_.items():
                data_samples[k] = v[condition_evaluated]
        else:
            condition_render_text = 'No condition'
            data_samples = data_samples_

        def add_condition_text(fig):
            fig.text(0.05, 0.92, condition_render_text, fontsize=8,
                     verticalalignment='bottom', horizontalalignment='left', color='black')

        with PdfPages(file) as pdf:
            # Create a new figure
            fig, ax = plt.subplots(figsize=(9, 4))  # A4 size in inches

            # Add text to the middle of the page
            ax.text(0.5, 0.5, condition_render_text, ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            # Save the figure to the PDF
            pdf.savefig(fig)
            fig.clear()

            fig, ax = self._make_2d_histo_three_body_prods(data_samples['momenta'], 'True decay products')
            add_condition_text(fig)
            pdf.savefig(fig)
            fig.clear()

            if 'momenta_reconstructed_upp' in data_samples:
                fig, ax = self._make_2d_histo_three_body_prods(data_samples['momenta_reconstructed_upp'], 'Reconstructed decay products')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()


                fig, ax = self._make_2d_three_body_prods_hist_diff(data_samples['momenta'], data_samples['momenta_reconstructed_upp'],
                          'Reconstructed vs true dist. diff. $\\log_{10}(N_{\\mathrm{reco}} / N_{\\mathrm{true}})$')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

            if 'momenta_sampled_upp' in data_samples:
                fig, ax = self._make_2d_histo_three_body_prods(data_samples['momenta_sampled_upp'], 'Sampled decay products')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

                fig, ax = self._make_2d_three_body_prods_hist_diff(data_samples['momenta_sampled_upp'], data_samples['momenta_reconstructed_upp'],
                          'Sampled vs true dist. diff. $\\log_{10}(N_{\\mathrm{sampled}} / N_{\\mathrm{true}})$')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()


            fig, ax = self._make_histo_three_body_prods(data_samples['momenta'], str='True decay products')
            add_condition_text(fig)
            pdf.savefig(fig)
            fig.clear()

            if 'momenta_reconstructed_upp' in data_samples:
                fig, ax = self._make_histo_three_body_prods(data_samples['momenta_reconstructed_upp'], str='Reconstructed decay products')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

                fig, ax = self._make_three_body_prods_hist_diff(data_samples['momenta'], data_samples['momenta_reconstructed_upp'],
                                                                str='Reconstructed vs true decay products', ylabel='true', ylabel_2='reco')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

            if 'momenta_sampled_upp' in data_samples:
                fig, ax = self._make_histo_three_body_prods(data_samples['momenta_sampled_upp'], str='Sampled decay products')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

                fig, ax = self._make_three_body_prods_hist_diff(data_samples['momenta'], data_samples['momenta_sampled_upp'],
                                                                str='Sampled vs true decay products', ylabel='true', ylabel_2='sampled')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

            fig, ax = self._make_histo_mother(data_samples['momenta_mother'], str='Mother')
            add_condition_text(fig)
            pdf.savefig(fig)
            fig.clear()

            fig, ax = self._make_hist_pz_pz(data_samples['momenta'], data_samples['momenta_mother'], str='Mother vs true decay products ')
            add_condition_text(fig)
            pdf.savefig(fig)
            fig.clear()

            if 'momenta_reconstructed_upp' in data_samples:
                fig, ax = self._make_hist_pz_pz(data_samples['momenta_reconstructed_upp'], data_samples['momenta_mother'], str='Mother vs reconstructed decay products ')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

                fig, ax = self._make_pz_pz_hist_diff(data_samples['momenta'], data_samples['momenta_reconstructed_upp'], data_samples['momenta_mother'], str='Mother vs reconstructed decay products hist. diff. $\\log_{10}(N_{\\mathrm{reco}} / N_{\\mathrm{true}})$')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

            if 'momenta_sampled_upp' in data_samples:
                fig, ax = self._make_hist_pz_pz(data_samples['momenta_sampled_upp'], data_samples['momenta_mother'], str='Mother vs sampled decay products ')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

                fig, ax = self._make_pz_pz_hist_diff(data_samples['momenta'], data_samples['momenta_sampled_upp'], data_samples['momenta_mother'], str='Mother vs sampled decay products hist. diff. $\\log_{10}(N_{\\mathrm{sampled}} / N_{\\mathrm{true}})$')
                add_condition_text(fig)
                pdf.savefig(fig)
                fig.clear()

    def plot(self, data_samples, file):
        if type(data_samples) is list:
            data_samples = tensors_dict_join(data_samples)

        assert type(data_samples) is dict

        data_samples_2 = {}
        for k,v in data_samples.items():
            if isinstance(v, torch.Tensor):
                data_samples_2[k] = v.cpu().numpy()
            else:
                data_samples_2[k] = v

        assert len(data_samples_2['momenta'].shape) == 3
        assert len(data_samples_2['momenta_mother'].shape) == 3
        assert data_samples_2['momenta'].shape[1] == 3
        assert data_samples_2['momenta'].shape[2] == 3
        assert data_samples_2['momenta_mother'].shape[2] == 3
        assert data_samples_2['momenta_mother'].shape[1] == 1


        self._do_plot(file+'_%03d.pdf'%0, data_samples_2)
        if self.conditions is not None:
            for i,c in enumerate(self.conditions):
                self._do_plot(file + '_%03d.pdf' % (i+1), data_samples_2, c)


