import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict


class SpectralPlot:
    def __init__(self, obs_data, freq_per_row=1000., width=10., height=3., sharey=True):
        obs_data = obs_data.copy()
        obs_data.sort(key=lambda item: item[0, 0])
        freq_min = obs_data[0][0, 0]
        freq_max = obs_data[-1][-1, 0]

        bounds_dict = {}
        slice_dict = defaultdict(list)
        i_segment = 0
        i_ax = 0
        freq_curr = freq_min + freq_per_row
        idx_b = 0
        while i_segment < len(obs_data) and freq_curr < freq_max + freq_per_row:
            freq = obs_data[i_segment][:, 0]

            idx_e = np.searchsorted(freq, freq_curr)
            if idx_e != 0 and idx_e - idx_b > 1:
                bounds_dict[i_ax] = (freq_curr - freq_per_row, freq_curr)
                slice_dict[i_ax].append((i_segment, slice(idx_b, idx_e)))

            if idx_e != len(freq):
                freq_curr += freq_per_row
                i_ax += 1
                idx_b = idx_e
            else:
                i_segment += 1
                idx_b = 0
        bounds = [args for args in bounds_dict.values()]

        n_axe = len(slice_dict)
        fig, axes = plt.subplots(figsize=(width, n_axe*height), nrows=n_axe, sharey=sharey)
        axes = np.ravel(axes)
        for i_ax, slice_list in enumerate(slice_dict.values()):
            ax = axes[i_ax]
            for i_segment, inds in slice_list:
                freq, spec = obs_data[i_segment].T
                ax.plot(freq[inds], spec[inds], "k")
                ax.set_xlim(*bounds[i_ax])

        self._fig = fig
        self._axes = axes
        self._bounds = bounds

    def _get_axe_idx(self, freq):
        idx = 0
        for lower, upper in self._bounds:
            if freq >= lower and freq <= upper:
                break
            idx += 1
        else:
            raise ValueError
        return idx

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._axes

    @property
    def bounds(self):
        return self._bounds

    def plot_spec(self, freq_list, spec_list, *args, **kwargs):
        sort_list = list(zip(freq_list, spec_list))
        sort_list.sort(key=lambda item: item[0][0])
        freq_list, spec_list = list(zip(*sort_list))

        i_segment = 0
        i_ax = 0
        idx_b = 0

        while i_segment < len(freq_list) and i_ax < len(self.axes):
            freq = freq_list[i_segment]
            spec = spec_list[i_segment]

            idx_e = np.searchsorted(freq, self.bounds[i_ax][-1])
            if idx_e - idx_b > 1 and spec is not None:
                self.axes[i_ax].plot(freq[idx_b:idx_e], spec[idx_b:idx_e], *args, **kwargs)

            if idx_e != len(freq):
                i_ax += 1
                idx_b = idx_e
            else:
                i_segment += 1
                idx_b = 0

    def plot_names(self, freqs, name_list, y_min, y_max,
                   color="k", color_blen="r", linestyles="--",
                   offset_0=1.5, offset_1=4, frac=.95, fontsize=12):
        for freq_c, names in zip(freqs, name_list):
            idx_ax = self._get_axe_idx(freq_c)
            ax = self.axes[idx_ax]
            if names is None:
                continue

            c = color if len(names) == 1 else color_blen
            ax.vlines(freq_c, y_min, y_max, c, linestyles)
            y_show = frac*(y_max + y_min)
            x_show = freq_c + offset_0
            for text in names:
                ax.text(
                    x_show, y_show, text,
                    rotation="vertical", verticalalignment="top",
                    fontsize=fontsize, c=c
                )
                x_show += offset_1

    def plot_errors(self, freqs, errors, y_min, y_max, colors="k", linestyles="--", offset=3):
        for freq_c, err in zip(freqs, errors):
            idx_ax = self._get_axe_idx(freq_c)
            ax = self.axes[idx_ax]
            ax.vlines(freq_c, y_min, y_max, colors, linestyles)
            ax.text(
                freq_c + offset, y_max, "{:.3f}".format(err), rotation="vertical")

    def vlines(self, freqs, *args, **kwargs):
        for freq_c in freqs:
            idx_ax = self._get_axe_idx(freq_c)
            self.axes[idx_ax].vlines(freq_c, *args, **kwargs)

    def set_ylim(self, *args, **kwargs):
        for ax in self.axes:
            ax.set_ylim(*args, **kwargs)