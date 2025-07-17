from typing import Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .config import Config
from .preprocess import load_preprocess, get_freq_data, get_T_data
from .identify import IdentResult


class PeakPlot:
    """Multi-window plot for visualizing the spectrum.

    Args:
        freqs: Cenetral frequencies for each window.
        delta_v: Velocity width in km/s of each window. Windows with overlaps
            will be merged.
        n_col: Number of windows per row.
        plot_width: Figure width of each sub-plot.
        plot_height: Figure height of each sub-plot.
    """
    def __init__(self,
                 freqs: np.ndarray,
                 delta_v: float=100.,
                 n_col: int=5,
                 plot_width: float=4,
                 plot_height: float=3):
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

    def plot_spec_from_config(self,
                              config: Config,
                              step_plot: bool=True,
                              ylim_factor: Optional[float]=None,
                              y_top_min: float=0.,
                              color="k",
                              **kwargs):
        """Plot the spectrum defined in the config dict."""
        obs_data = load_preprocess(config["obs_info"], clip=False)
        freq_data = get_freq_data(obs_data)
        T_data = get_T_data(obs_data)
        self.plot_spec(
            freq_data, T_data,
            step_plot=step_plot,
            ylim_factor=ylim_factor,
            y_top_min=y_top_min,
            color=color,
            **kwargs
        )

    def plot_spec(self,
                  freq_data: list,
                  spec_data: list,
                  step_plot: bool=False,
                  ylim_factor: Optional[float]=None,
                  y_top_min: float=0.,
                  **kwargs):
        """Plot a spectrum."""
        for i_a, ax in enumerate(self._axes.flat):
            if i_a >= self.n_plot:
                continue

            y_max = 0.
            lower, upper = self.bounds[i_a]
            for i_segment, freq in enumerate(freq_data):
                cond = (freq >= lower) & (freq <= upper)
                if np.count_nonzero(cond) == 0:
                    continue
                T_data = spec_data[i_segment][cond]
                if step_plot:
                    ax.step(freq[cond], T_data, where="mid", **kwargs)
                else:
                    ax.plot(freq[cond], T_data, **kwargs)
                y_max = max(y_max, max(T_data))
            if ylim_factor is not None:
                y_top = ylim_factor*y_max
                y_top = max(y_top, y_top_min)
                ax.set_ylim(top=y_top)

    def plot_prominence(self, freq_data, prom_list):
        for i_a, ax in enumerate(self._axes.flat):
            if i_a >= self.n_plot:
                continue

            lower, upper = self.bounds[i_a]
            for freq, prom in zip(freq_data, prom_list):
                if freq[0] > upper or freq[-1] < upper:
                    continue
                x_min = max(freq[0], lower)
                x_max = min(freq[-1], upper)
                ax.hlines(prom, x_min, x_max, "grey")

    def vlines(self, freqs: np.ndarray, **kwargs):
        """Plot vertical lines."""
        kwargs_ = {"linestyle": "--", "color": "k"}
        kwargs_.update(kwargs)
        for i_a, ax in enumerate(self._axes.flat):
            if i_a >= self.n_plot:
                continue

            y_min_, y_max_ = ax.get_ylim()
            for freq in freqs:
                lower, upper = self.bounds[i_a]
                if freq >= lower and freq <= upper:
                    ax.vlines(freq, y_min_, y_max_, **kwargs_)
                    ax.set_ylim(y_min_, y_max_)

    def vtexts(self,
               freqs: np.ndarray,
               texts: list,
               h_txt_offset: float=1.5e-2,
               v_txt_offset: float=.95,
               **kwargs):
        """Plot vertical texts."""
        for i_a, ax in enumerate(self._axes.flat):
            if i_a >= self.n_plot:
                continue

            x_min, x_max = ax.get_xlim()
            y_min_, y_max_ = ax.get_ylim()
            for freq, text in zip(freqs, texts):
                lower, upper = self.bounds[i_a]
                if freq >= lower and freq <= upper:
                    y_show = y_min_ + v_txt_offset*(y_max_ - y_min_)
                    x_show = freq + h_txt_offset*(x_max - x_min)
                    ax.text(
                        x_show, y_show, text, rotation="vertical", va="top", **kwargs
                    )


class SpectralPlot:
    """Multi-row plot for visualizing the spectrum of multiple spectral windows.

    Args:
        freq_data: List of 1D array that specifies the frequency values for each
            spectral window.
        freq_per_row: Frequency range to show in each row. The unit should be
            the same as ``freq_data``.
        width: Figure width.
        height: Figure height of each row.
        axes: Axes to plot. If ``None``, create a new figure.
    """
    def __init__(self,
                 freq_data: list,
                 freq_per_row: float=1000.,
                 width: float=20.,
                 height: float=3.,
                 axes: Optional[np.ndarray]=None):
        bounds = self._derive_bounds(freq_data, freq_per_row)
        n_axe = len(bounds)
        if axes is None:
            fig, axes = plt.subplots(figsize=(width, n_axe*height), nrows=n_axe,)
            axes = np.ravel(axes)
        else:
            assert len(axes) == n_axe, f"Number of input axes must be equal to {n_axe}."
            fig = None
        for i_ax, ax in enumerate(axes):
            ax.set_xlim(*bounds[i_ax])

        self._fig = fig
        self._axes = axes
        self._bounds = bounds
        self._y_min = None
        self._y_max = None
        self._freq_per_row = freq_per_row

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
    def from_config(cls,
                    config: Config,
                    freq_per_row: float=1000.,
                    width: float=20.,
                    height: float=3.,
                    axes:np.ndarray=None,
                    color: str="k",
                    **kwargs):
        """Create a plot from a ``Config`` instance.

        Args:
            config: ``Config`` instance.
            freq_per_row: Frequency range to show in each row. The unit should
                be the same as defined in ``config``.
            width: Figure width.
            height: Figure height of each row.
            axes: Axes to plot. If ``None``, create a new figure.
            color: Color of the spectrum defined in ``config``.
            **kwargs: Other arguments passed to ``plt.plot`` to plot the
                spectrum defined in ``config``.
        """
        obs_data = load_preprocess(config["obs_info"], clip=False)
        freq_data = get_freq_data(obs_data)
        plot = cls(
            freq_data=freq_data,
            freq_per_row=freq_per_row,
            width=width,
            height=height,
            axes=axes,
        )
        plot.plot_spec(freq_data, get_T_data(obs_data), color=color, **kwargs)
        noise = np.mean([item["noise"] for item in config["obs_info"]])
        plot.set_ylim(-10.*noise, 100.*noise)
        for ax in plot.axes:
            ax.set_xlabel("Frequency [MHz]")
            ax.set_ylabel("Intensity [K]")
        plt.subplots_adjust(hspace=.3)
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

    def plot_ident_result(self,
                          ident_result: IdentResult,
                          key: Optional[int]=None,
                          name: Optional[str]=None,
                          show_lines: bool=True,
                          color: str="k",
                          color_blen: str="r",
                          color_fp: str="b",
                          h_txt_offset: float=2.5e-3,
                          v_txt_offset: float=.95,
                          fontsize: float=12,
                          T_base_data: Optional[list]=None,
                          kwargs_spec: Optional[dict]=None):
        """Plot a identification result.

        This method is a combination of ``plot_spec`` and ``plot_names``.

        Args:
            ident_result: Identification result.
            key: Molecule ID. This is used to plot the result of a single
                molecule in a combined result.
            name: Molecule name. This is used to plot the result of a single
                molecule in a combined result.
            show_lines: Whether to show the vertical lines that indicate the
                molecules.
            txt_offset: Text offset of the lines. Larger values mean farther
                from the line.
            color: Line color of the peaks that match the observed spectrum.
            color_blen: Line color of the peaks that match the observed
                spectrum but contributed by multiple species.
            color_fp: Color of the peaks found in the fitted spectrum but
                missing from the observed spectrum.
            fontsize: Font size of the molecules.
            T_base_data: Base intensity data.
            kwargs_spec: Keyword arguments passed to ``plt.plot`` to plot the
                spectrum.
        """
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
                color=color,
                color_blen=color_blen,
                h_txt_offset=h_txt_offset,
                v_txt_offset=v_txt_offset,
                fontsize=fontsize
            )
            self.plot_names(
                ident_result.line_table_fp.freq,
                ident_result.line_table_fp.name,
                color=color_fp,
                color_blen=color_fp,
                h_txt_offset=h_txt_offset,
                v_txt_offset=v_txt_offset,
                fontsize=fontsize
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
            spans, name_list,
            color=color,
            color_blen=color_blen,
            h_txt_offset=h_txt_offset,
            v_txt_offset=v_txt_offset,
            fontsize=fontsize
        )
        inds = ident_result.filter_name_list(name_set, line_table_fp.name)
        spans = line_table_fp.freq[inds]
        name_list = np.array(line_table_fp.name, dtype=object)[inds]
        self.plot_names(
            spans, name_list,
            color=color_fp,
            color_blen=color_fp,
            h_txt_offset=h_txt_offset,
            v_txt_offset=v_txt_offset,
            fontsize=fontsize
        )

    def plot_spec(self,
                  freq_data: list,
                  spec_data: list,
                  *args,
                  color: str="C0",
                  **kwargs):
        """Plot a spectrum.

        Args:
            freq_data: List of 1D array specifiyng the frequency of each
                spectral window.
            spec_data: List of 1D array specifiyng the intensity of each
                spectral window.
            *args: Arguments passed to ``plt.plot``.
            color: Color of the spectrum.
            **kwargs: Keyword arguments passed to ``plt.plot``.
        """
        sort_list = list(zip(freq_data, spec_data))
        sort_list.sort(key=lambda item: item[0][0])
        freq_data, spec_data = list(zip(*sort_list))

        i_segment = 0
        i_ax = 0
        idx_b = 0

        while i_segment < len(freq_data) and i_ax < len(self.axes):
            freq = freq_data[i_segment]
            spec = spec_data[i_segment]

            idx_e = np.searchsorted(freq, self.bounds[i_ax][-1])
            if idx_e - idx_b > 1 and spec is not None:
                self.axes[i_ax].plot(
                    freq[idx_b:idx_e], spec[idx_b:idx_e], *args, color=color, **kwargs
                )
                # Only use one label
                if "label" in kwargs:
                    kwargs["label"] = None

            if idx_e != len(freq):
                i_ax += 1
                idx_b = idx_e
            else:
                i_segment += 1
                idx_b = 0

    def plot_names(self,
                   freqs: np.ndarray,
                   name_list: list,
                   key: Optional[int]=None,
                   color: str="k",
                   color_blen: str="r",
                   linestyles: str="--",
                   h_txt_offset: float=2.5e-3,
                   v_txt_offset: float=.95,
                   fontsize: float=12):
        """Plot the identitied names of the lines.

        Args:
            freqs: Frequencies of the lines.
            name_list: Names of the lines.
            key: Molecule ID. This is used to plot the result of a single
                molecule in a combined result.
            color: Line color of the peaks that match the observed spectrum.
            color_blen: Line color of the peaks that match the observed
                spectrum but contributed by multiple species.
            linestyles: Line styles.
            h_txt_offset: Horizontal text offset. Larger values mean farther
                from the line.
            v_txt_offset: Vertical text offset. Smaller values mean farther
                from the top.
            fontsize: Font size of the molecules.
        """
        for freq_c, names in zip(freqs, name_list):
            if names is None or (key is not None and key not in names):
                continue

            idx_ax = self._get_axe_idx(freq_c)
            ax = self.axes[idx_ax]
            y_min, y_max = self.get_ylim(ax)
            c = color if len(names) == 1 else color_blen
            ax.vlines(freq_c, y_min, y_max, c, linestyles)
            y_show = y_min + v_txt_offset*(y_max - y_min)
            x_show = freq_c + h_txt_offset*self._freq_per_row
            ax.text(
                x_show, y_show, "\n".join(names),
                rotation="vertical", verticalalignment="top",
                fontsize=fontsize, c=c
            )

    def plot_unknown_lines(self,
                           ident_result: IdentResult,
                           color: str="grey",
                           linestyle: str="-",
                           alpha: float=0.5):
        """Plot unidentified lines.

        Args:
            ident_result: Identification result.
            color: Color of the lines.
            linestyle: Line style.
            alpha: Transparency.
        """
        freqs = ident_result.get_unknown_lines()
        self.vlines(freqs, colors=color, linestyles=linestyle, alpha=alpha)

    def vlines(self, freqs: np.ndarray, **kwargs):
        """Plot vertical lines.

        Args:
            freqs: Frequencies of the lines.
            **kwargs: Keyword arguments passed to ``plt.vlines``.
        """
        for freq_c in freqs:
            idx_ax = self._get_axe_idx(freq_c)
            ax = self.axes[idx_ax]
            y_min, y_max = self.get_ylim(ax)
            ax.vlines(freq_c, y_min, y_max, **kwargs)

    def set_ylim(self, y_min: float, y_max: float, **kwargs):
        """Set the y limits for each plot.

        Args:
            y_min: Minimum y value.
            y_max: Maximum y value.
            **kwargs: Keyword arguments passed to ``plt.set_ylim``.
        """
        for ax in self.axes:
            ax.set_ylim(y_min, y_max, **kwargs)
        self._y_min = y_min
        self._y_max = y_max

    def get_ylim(self, ax):
        """Get the y limits for the given axis.

        Args:
            ax: Axis.
        """
        y_min, y_max = ax.get_ylim()
        if self._y_min is not None:
            y_min = self._y_min
        if self._y_max is not None:
            y_max = self._y_max
        return y_min, y_max