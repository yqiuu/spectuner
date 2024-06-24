import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict

from .preprocess import get_freq_data, get_T_data


class PeakPlot:
    def __init__(self, freqs, delta_v=100, n_col=5, plot_width=4, plot_height=3):
        bounds = []
        for freq in freqs:
            bounds.append([freq*(1 - delta_v/3e5), freq*(1 + delta_v/3e5)])

        # Merge inter peaks
        bounds_new = []
        for lower, upper in bounds:
            if len(bounds_new) == 0 or bounds_new[-1][-1] < lower:
                bounds_new.append([lower, upper])
            else:
                bounds_new[-1][-1] = max(bounds_new[-1][-1], upper)
        bounds = bounds_new
        self._bounds = bounds

        n_plot = len(bounds)
        if n_plot < n_col:
            n_row = 1
            n_col = len(bounds)
        else:
            n_row = n_plot//n_col + int(n_plot%n_col != 0)
        self.n_plot = n_plot

        self._fig, self._axes = plt.subplots(
            figsize=(n_col*plot_width, n_row*plot_height), nrows=n_row, ncols=n_col
        )
        if n_row == 1 and n_col == 1:
            self._axes = np.ravel(self.axes)

        for ax in np.ravel(self._axes)[n_plot:]:
            ax.axis("off")

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._axes

    @property
    def bounds(self):
        return self._bounds

    def plot_spec(self, freq_list, spec_list, *args, apply_ylim=False, **kwargs):
        for i_a, ax in enumerate(self._axes.flat):
            if i_a >= self.n_plot:
                continue

            y_max = 0.
            lower, upper = self.bounds[i_a]
            freq_c = .5*(lower + upper)
            for i_segment, freq in enumerate(freq_list):
                if (freq_c >= freq[0]) & (freq_c <= freq[-1]):
                    cond = (freq >= lower) & (freq <= upper)
                    T_data = spec_list[i_segment][cond]
                    ax.plot(freq[cond], T_data, *args, **kwargs)
                    y_max = max(y_max, max(T_data))
            if apply_ylim:
                ax.set_ylim(-1e-2*y_max, top=1.5*y_max)

    def plot_prominence(self, freq_list, prom_list):
        for i_a, ax in enumerate(self._axes.flat):
            if i_a >= self.n_plot:
                continue

            lower, upper = self.bounds[i_a]
            for freq, prom in zip(freq_list, prom_list):
                if freq[0] > upper or freq[-1] < upper:
                    continue
                x_min = max(freq[0], lower)
                x_max = min(freq[-1], upper)
                ax.hlines(prom, x_min, x_max, "grey")

    def vlines(self, freqs, y_min, y_max, *args, **kwargs):
        for i_a, ax in enumerate(self._axes.flat):
            if i_a >= self.n_plot:
                continue

            for freq in freqs:
                lower, upper = self.bounds[i_a]
                if freq >= lower and freq <= upper:
                    ax.vlines(freq, y_min, y_max, *args, **kwargs)


class SpectralPlot:
    def __init__(self, freq_data, freq_per_row=1000., width=15., height=3., sharey=True):
        bounds = self._derive_bounds(freq_data, freq_per_row)
        n_axe = len(bounds)
        fig, axes = plt.subplots(figsize=(width, n_axe*height), nrows=n_axe, sharey=sharey)
        axes = np.ravel(axes)
        for i_ax, ax in enumerate(axes):
            ax.set_xlim(*bounds[i_ax])

        self._fig = fig
        self._axes = axes
        self._bounds = bounds

    def _derive_bounds(self, freq_data, freq_per_row):
        freq_data = freq_data.copy()
        freq_data.sort(key=lambda item: item[0])
        freq_min = freq_data[0][0]
        freq_max = freq_data[-1][-1]
        if freq_max - freq_min < freq_per_row:
            return [(freq_min, freq_max),]

        bounds_dict = {}
        slice_dict = defaultdict(list)
        i_segment = 0
        i_ax = 0
        freq_curr = freq_min + freq_per_row
        idx_b = 0
        while i_segment < len(freq_data) and freq_curr < freq_max + freq_per_row:
            freq = freq_data[i_segment]

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
        return bounds

    def _get_axe_idx(self, freq):
        idx = 0
        for lower, upper in self._bounds:
            if freq >= lower and freq <= upper:
                break
            idx += 1
        else:
            raise ValueError
        return idx

    @classmethod
    def from_config(cls, config, freq_per_row=1000., width=15., height=3., sharey=True,
                    color="k", **kwargs):
        obs_data = [np.loadtxt(fname) for fname in config["files"]]
        freq_data = get_freq_data(obs_data)
        plot = cls(
            freq_data=freq_data,
            freq_per_row=freq_per_row,
            width=width,
            height=height,
            sharey=sharey,
        )
        plot.plot_spec(freq_data, get_T_data(obs_data), color=color, **kwargs)
        return plot

    @property
    def fig(self):
        return self._fig

    @property
    def axes(self):
        return self._axes

    @property
    def bounds(self):
        return self._bounds

    def plot_T_pred(self, ident_result, y_min, y_max, key=None, name=None,
                    show_lines=True, offset_0=1.5,
                    color="k", color_blen="r", color_fp="b",
                    fontsize=12, T_base_data=None, kwargs_spec=None):
        T_data = ident_result.get_T_pred(key, name)
        if T_base_data is not None:
            for i_segment, T_base in enumerate(T_base_data):
                if T_base_data is None or T_data[i_segment] is None:
                    continue
                T_data[i_segment] = T_data[i_segment] \
                    + T_base - ident_result.T_back
        if kwargs_spec is None:
            kwargs_spec = {}
        self.plot_spec(ident_result.freq_data, T_data, **kwargs_spec)

        if not show_lines:
            return

        if key is None:
            self.plot_names(
                ident_result.line_table.freq,
                ident_result.line_table.name,
                y_min, y_max,
                color=color, color_blen=color_blen,
                offset_0=offset_0, fontsize=fontsize
            )
            self.plot_names(
                ident_result.line_table_fp.freq,
                ident_result.line_table_fp.name,
                y_min, y_max,
                color=color_fp, color_blen=color_fp,
                offset_0=offset_0, fontsize=fontsize
            )
            return

        if name is None:
            name_set = set(ident_result.T_single_dict[key])
        else:
            name_set = set((name,))

        line_table = ident_result.line_table
        line_table_fp = ident_result.line_table_fp
        inds = ident_result.filter_name_list(name_set, line_table.name)
        spans = line_table.freq[inds]
        name_list = np.array(line_table.name, dtype=object)[inds]
        self.plot_names(
            spans, name_list, y_min, y_max,
            color=color, color_blen=color_blen,
            offset_0=offset_0, fontsize=fontsize
        )
        inds = ident_result.filter_name_list(name_set, line_table_fp.name)
        spans = line_table_fp.freq[inds]
        name_list = np.array(line_table_fp.name, dtype=object)[inds]
        self.plot_names(
            spans, name_list, y_min, y_max,
            color=color_fp, color_blen=color_fp,
            offset_0=offset_0, fontsize=fontsize
        )

    def plot_unknown_lines(self, ident_result, y_min, y_max, color="grey", linestyle="-"):
        freqs = ident_result.get_unknown_lines()
        self.vlines(freqs, y_min, y_max, colors=color, linestyles=linestyle)

    def plot_spec(self, freq_list, spec_list, *args, color="C0", **kwargs):
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
                self.axes[i_ax].plot(
                    freq[idx_b:idx_e], spec[idx_b:idx_e], *args, color=color, **kwargs
                )

            if idx_e != len(freq):
                i_ax += 1
                idx_b = idx_e
            else:
                i_segment += 1
                idx_b = 0

    def plot_names(self, freqs, name_list, y_min, y_max, key=None,
                   color="k", color_blen="r", linestyles="--",
                   offset_0=1.5, frac=.95, fontsize=12):
        for freq_c, names in zip(freqs, name_list):
            if names is None or (key is not None and key not in names):
                continue

            idx_ax = self._get_axe_idx(freq_c)
            ax = self.axes[idx_ax]
            c = color if len(names) == 1 else color_blen
            ax.vlines(freq_c, y_min, y_max, c, linestyles)
            y_show = y_min + frac*(y_max - y_min)
            x_show = freq_c + offset_0
            ax.text(
                x_show, y_show, "\n".join(names),
                rotation="vertical", verticalalignment="top",
                fontsize=fontsize, c=c
            )

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