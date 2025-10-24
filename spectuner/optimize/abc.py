from typing import Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from ..slm_factory import SpectralLineModelFactory
from ..sl_model import create_specie_list


__all__ = ["ModifiedArtificialBeeColony"]


@dataclass
class Bee:
    fun: float
    T_tot_data: list
    x_dict: dict
    T_dict: dict


class ModifiedArtificialBeeColony:
    """Modified Artificial Bee Colony.

    This algorithm takes advantage of the fact that the spectra of multiple
    species can be combined linearly and the ABC algorithm only updates one
    dimension at once. Therefore, every time the algorithm searches new
    positions, we only recompute the spectra of the species that have changed,
    which is much faster than recomputing all the spectra.

    Args:
        n_swarm: Number of bees.
        n_cycle_min: Minimum number of iterations.
        n_cycle_max: Maximum number of iterations.
        n_cycle_dim: Dimension multiplier for minimum number of iterations.
        n_stop: Terminate the algorithm when the loss function does not change
            for ``n_stop`` iterations.
        tol_stop: Tolerance for stopping the algorithm.
        gbest_c: Acceleration coefficient towards the glaobl minimum.
        rstate: Random state.
        chunk_size: Chunk size for parallelization.
    """
    def __init__(self,
                 n_swarm: int=64,
                 tour_size: int=5,
                 n_cycle_min: int=100,
                 n_cycle_max: int=1000,
                 n_cycle_dim: int=5,
                 n_stop: int=15,
                 tol_stop: float=1.e-5,
                 gbest_c: float=1.5,
                 rstate: Optional[np.random.RandomState]=None,
                 chunk_size: int=2):
        self._n_swarm = n_swarm
        self._tour_size = tour_size
        self._n_cycle_min = n_cycle_min
        self._n_cycle_max = n_cycle_max
        self._n_cycle_dim = n_cycle_dim
        self._n_stop = n_stop
        self._tol_stop = tol_stop
        self._gbest_c = gbest_c
        if rstate is None:
            self._rstate = np.random.RandomState()
        else:
            self._rstate = rstate
        self._chunk_size = chunk_size

    def run(self,
            slm_factory: SpectralLineModelFactory,
            obs_info: list,
            species: list,
            pos_init: Optional[np.ndarray]=None,
            pool: Optional[mp.Pool]=None) -> dict:
        """Run the optimization.

        Args:
            slm_factory: Spectral line model factory.
            obs_info: Observation information.
            species: List of species.
            pos_init: Initial positions (n_swarm, n_specie, n_dim).
            pool: Multiprocessing pool.

        Returns:
            dict: Optimization result.
        """

        sl_model_dict, loss_fn, bounds, bees = self._init(
            slm_factory=slm_factory,
            obs_info=obs_info,
            species=species,
            pos_init=pos_init
        )
        history = [min(bee.fun for bee in bees)]

        n_stuck = 0
        n_cycle_min = self._n_cycle_min \
            + (len(species)*len(bounds) - 5)*self._n_cycle_dim
        pbar = tqdm(
            range(self._n_cycle_max - 1),
            desc="loss = NaN", total=float("inf")
        )
        for i_cycle in pbar:
            self._step(
                sl_model_dict=sl_model_dict,
                loss_fn=loss_fn,
                bounds=bounds,
                bees=bees,
                pool=pool,
            )
            fun = min(bee.fun for bee in bees)
            pbar.set_description(f"loss = {fun:.3e}")
            history.append(fun)

            if i_cycle > 1:
                rate = (history[-2] - history[-1])/abs(history[-2])
                if rate < self._tol_stop:
                    n_stuck += 1
                else:
                    n_stuck = 0

            if n_stuck >= self._n_stop and i_cycle + 1 >= n_cycle_min:
                break
        pbar.close()

        i_best = np.argmin([bee.fun for bee in bees])
        bee_best = bees[i_best]
        x = np.concatenate([bee_best.x_dict[key] for key in species])
        speice_list_tot = []
        for mol_name in species:
            speice_list_tot.extend(create_specie_list(mol_name))
        freq_data = sl_model_dict[species[0]].freq_data
        res_dict = {
            "x":  x,
            "fun": bee_best.fun,
            "nfev": None,
            "specie": speice_list_tot,
            "freq": freq_data,
            "T_pred": bee_best.T_tot_data,
            "history": np.array(history),
        }
        return res_dict


    def _init(self,
              slm_factory: SpectralLineModelFactory,
              obs_info: list,
              species: list,
              pos_init: Optional[np.ndarray]=None):
        loss_fn = slm_factory.create_peak_mgr(obs_info)

        sl_model_dict = {}
        for mol_name in species:
            specie_list = create_specie_list(mol_name)
            sl_model_dict[mol_name] \
                = slm_factory.create_sl_model(obs_info, specie_list)
        # The bounds of every molecules are the same
        bounds = sl_model_dict[species[0]].param_mgr.derive_bounds()

        bees = []
        for i_bee in range(self._n_swarm):
            x_dict = {}
            T_dict = {}
            T_tot_data = None
            for i_mol, mol_name in enumerate(species):
                sl_model = sl_model_dict[mol_name]
                if pos_init is None:
                    lower, upper = np.array(bounds).T
                    pos = lower + (upper - lower)*self._rstate.rand(len(lower))
                else:
                    pos = pos_init[i_bee, i_mol]
                T_data = sl_model(pos)
                x_dict[mol_name] = pos
                T_dict[mol_name] = T_data
                if T_tot_data is None:
                    T_tot_data = deepcopy(T_data)
                else:
                    for i_segment, T_pred in enumerate(T_data):
                        T_tot_data[i_segment] += T_pred

                if i_bee == 0:
                    sl_model_dict[mol_name] = sl_model

            fun = loss_fn(T_tot_data)[0]
            bees.append(Bee(fun, T_tot_data, x_dict, T_dict))

        return sl_model_dict, loss_fn, bounds, bees


    def _step(self,
              sl_model_dict: dict,
              loss_fn: Callable,
              bounds: np.ndarray,
              bees: list[Bee],
              pool: object):
        bee_inds = list(range(self._n_swarm))
        self._bee_step(
            sl_model_dict=sl_model_dict,
            loss_fn=loss_fn,
            bounds=bounds,
            bees=bees,
            bee_inds=bee_inds,
            pool=pool,
        )

        funs = np.array([bee.fun for bee in bees])
        bee_inds = self._tournament_selection(funs)
        self._bee_step(
            sl_model_dict=sl_model_dict,
            loss_fn=loss_fn,
            bounds=bounds,
            bees=bees,
            bee_inds=bee_inds,
            pool=pool,
        )

    def _bee_step(self,
                  sl_model_dict: dict,
                  loss_fn: Callable,
                  bounds: np.ndarray,
                  bees: list[Bee],
                  bee_inds: list[int],
                  pool: object):
        species = list(sl_model_dict.keys())
        i_best = np.argmin([bee.fun for bee in bees])
        pos_new_list = []
        for i_bee in bee_inds:
            pos_new_list.append(self._search_new_pos(
                bees=bees,
                species=species,
                bounds=bounds,
                i_bee=i_bee,
                i_best=i_best,
            ))
        fun_new_list = self._call_loss_fn(
            sl_model_dict=sl_model_dict,
            loss_fn=loss_fn,
            bees=bees,
            pos_new_list=pos_new_list,
            pool=pool,
        )
        for i_bee in bee_inds:
            mol_name, pos_new = pos_new_list[i_bee]
            fun_new, T_data_new, T_tot_data = fun_new_list[i_bee]
            self._move(
                bee=bees[i_bee],
                mol_name=mol_name,
                pos_new=pos_new,
                fun_new=fun_new,
                T_data_new=T_data_new,
                T_tot_data=T_tot_data,
            )

    def _search_new_pos(self,
                        bees: list[Bee],
                        species: list,
                        bounds: np.ndarray,
                        i_bee: int,
                        i_best: int):
        rstate = self._rstate
        # Select a bee
        while True:
            j_bee = rstate.randint(len(bees))
            if j_bee != i_bee:
                break
        # Select a specie
        mol_name = species[rstate.randint(len(species))]
        # Select a dim
        i_dim = self._rstate.randint(len(bounds))
        # Generate a new position
        gbest_c = self._gbest_c*rstate.rand()
        pos_i = bees[i_bee].x_dict[mol_name][i_dim]
        pos_j = bees[j_bee].x_dict[mol_name][i_dim]
        pos_best_i = bees[i_best].x_dict[mol_name][i_dim]
        pos_new_i = pos_i \
            + (2*rstate.rand() - 1)*(pos_j - pos_i) \
            + gbest_c*(pos_best_i - pos_i)
        lower, upper = bounds[i_dim]
        if pos_new_i < lower:
            pos_new_i = 2*lower - pos_new_i
        if pos_new_i > upper:
            pos_new_i = 2*upper - pos_new_i
        pos_new_i = np.clip(pos_new_i, lower, upper)
        pos_new = np.copy(bees[i_bee].x_dict[mol_name])
        pos_new[i_dim] = pos_new_i
        return mol_name, pos_new

    def _call_loss_fn(self,
                      sl_model_dict: Callable,
                      loss_fn: Callable,
                      bees: list[Bee],
                      pos_new_list: list,
                      pool=None):
        inputs = []
        for bee, (mol_name, pos_new) in zip(bees, pos_new_list):
            inputs.append((
                sl_model_dict[mol_name], loss_fn, bee, mol_name, pos_new
            ))
        if pool is None:
            return list(map(_call_loss_fn, zip(*inputs)))
        return pool.starmap(_call_loss_fn, inputs, chunksize=self._chunk_size)

    def _move(self,
              bee: Bee,
              mol_name: str,
              pos_new: np.ndarray,
              fun_new: float,
              T_data_new: np.ndarray,
              T_tot_data: list[np.ndarray]):
        if fun_new < bee.fun:
            bee.fun = fun_new
            bee.T_tot_data = T_tot_data
            bee.x_dict[mol_name] = pos_new
            bee.T_dict[mol_name] = T_data_new

    def _tournament_selection(self, funs):
        selected = []
        for _ in range(self._n_swarm):
            groups = self._rstate.choice(
                self._n_swarm, self._tour_size, replace=False
            )
            i_best = min(groups, key=lambda i: funs[i])
            selected.append(i_best)
        return selected

def _call_loss_fn(sl_model: Callable,
                  loss_fn: Callable,
                  target: Bee,
                  mol_name: str,
                  pos_new: np.ndarray):
    T_tot_data = deepcopy(target.T_tot_data)
    T_data_prev = target.T_dict[mol_name]
    T_data_new = sl_model(pos_new)
    for i_segment in range(len(T_tot_data)):
        T_tot_data[i_segment] += T_data_new[i_segment] - T_data_prev[i_segment]
    fun_new = loss_fn(T_tot_data)[0]
    return fun_new, T_data_new, T_tot_data