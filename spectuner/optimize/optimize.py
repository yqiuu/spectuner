from __future__ import annotations
import inspect
import multiprocessing as mp
from typing import Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
from copy import deepcopy

import h5py
import numpy as np
from tqdm import tqdm
from swing import ParticleSwarm, ArtificialBeeColony
from scipy.optimize import minimize

from ..peaks import create_spans
from ..slm_factory import jit_fitting_model, SpectralLineModelFactory
from ..ai import predict_single_pixel, InferenceModel
from ..identify import IdentResult


def optimize(engine: Union[SpectralLineModelFactory, InferenceModel],
             obs_info: list,
             specie_list: list,
             config_opt: dict,
             T_base_data: Optional[list]=None,
             x0: Optional[np.ndarray]=None,
             config_inf: dict=None):
    opt = create_optimizer(config_opt)
    if isinstance(engine, SpectralLineModelFactory):
        fitting_model = engine.create_fitting_model(
            obs_info=obs_info,
            specie_list=specie_list,
            T_base_data=T_base_data,
        )
        jit_fitting_model(fitting_model)
        if x0 is None:
            return opt(fitting_model)
        return opt(fitting_model, x0)
    elif isinstance(engine, InferenceModel) is not None:
        return engine.call_single(
            obs_info=obs_info,
            specie_name=specie_list[0]["root"],
            postprocess=opt,
            T_base_data=T_base_data,
            device=config_inf["device"],
        )
    else:
        raise ValueError(f"Unknown engine: {engine}.")


def optimize_all(engine: Union[SpectralLineModelFactory, InferenceModel],
                 obs_info: list,
                 targets: list,
                 config_opt: dict,
                 T_base_data: list=None,
                 trans_counts: dict=None,
                 config_inf: dict=None,
                 pool: mp.Pool=None):
    # Optimize without inference model
    if isinstance(engine, SpectralLineModelFactory):
        results = []
        for specie_list in targets:
            fitting_model = engine.create_fitting_model(
                obs_info=obs_info,
                specie_list=specie_list,
                T_base_data=T_base_data,
            )
            results.append(
                pool.apply_async(_optimize_worker, (fitting_model, config_opt))
            )
        with tqdm(enumerate(targets), total=len(targets)) as pbar:
            for i_res, specie_list in pbar:
                pbar.set_description("Fitting {}".format(
                    join_specie_names(specie_list[0]["species"])))
                results[i_res] = results[i_res].get()
        return results
    elif isinstance(engine, InferenceModel):
        specie_names = []
        numbers = []
        for specie_list in targets:
            name = specie_list[0]["root"]
            specie_names.append(name)
            numbers.append(trans_counts[name])
        opt = create_optimizer(config_opt)
        results = predict_single_pixel(
            inf_model=engine,
            obs_info=obs_info,
            entries=specie_names,
            numbers=numbers,
            postprocess=opt,
            max_diff=config_inf["max_diff"],
            max_batch_size=config_inf["max_batch_size"],
            device=config_inf["device"],
            pool=pool
        )
        for res, specie_list in zip(results, targets):
            res["specie"] = specie_list
        return results
    else:
        raise ValueError(f"Unknown engine: {engine}.")


def _optimize_worker(fitting_model, config_opt):
    opt = create_optimizer(config_opt)
    jit_fitting_model(fitting_model)
    return opt(fitting_model)


def create_pool(n_process, use_mpi):
    if use_mpi:
        from mpi4py.futures import MPIPoolExecutor
        return MPIPoolExecutor(n_process)
    else:
        return Pool(n_process)


def prepare_base_props(fname, config):
    if fname is not None:
        fname = Path(fname)
        fname = fname.with_name(f"identify_{fname.name}")
        with h5py.File(fname) as fp:
            res = IdentResult.load_hdf(fp["combine"])
        T_base_data = res.get_T_pred()
        freqs_exclude = res.get_identified_lines()
        spans_include = create_spans(
            res.get_unknown_lines(), *config["opt"]["bounds"]["v_LSR"]
        )
        exclude_list = derive_exclude_list(res)

        id_offset = 0
        for key in res.specie_data:
            id_offset = max(id_offset, key)
        id_offset += 1
    else:
        T_base_data = None
        freqs_exclude = np.zeros(0)
        spans_include = np.zeros((0, 2))
        exclude_list = []
        id_offset = 0

    return {
        "T_base": T_base_data,
        "freqs_exclude": freqs_exclude,
        "spans_include": spans_include,
        "exclude_list": exclude_list,
        "id_offset": id_offset
    }


def derive_exclude_list(res):
    exclude_set = set()
    for sub_dict in res.specie_data.values():
        for key in sub_dict:
            exclude_set.add(key.split(";")[0])
    return list(exclude_set)


def print_fitting(specie_list):
    print("Fitting: {}.".format(", ".join(specie_list)))


def join_specie_names(species):
    return ", ".join(species)


def random_mutation_by_group(pm, params, bounds, prob=0.4, rstate=None):
    """Perturb the parameters by a group of molecules.

    Args:
        pm (ParameterManager): Parameter Manager.
        params (array): Parameters
        bounds (array): Bounds
        prob (float, optional): Mutation probability. The code will perturb at
            least one group unless the probability is 0. Defaults to 0.4.
        rstate (RandomState, optional): RNG.

    Returns:
        _type_: _description_
    """
    if rstate is None:
        rstate = np.random

    params_mol, params_den, params_misc = pm.split_params(params, need_reshape=False)
    lb_mol, lb_den, _ = pm.split_params(bounds[:, 0], need_reshape=False)
    ub_mol, ub_den, _ = pm.split_params(bounds[:, 1], need_reshape=False)

    params_mol = params_mol.copy()
    params_den = params_den.copy()
    n_replace = int(prob*pm.n_mol)
    if n_replace == 0 and prob > 0:
        n_replace = 1
    id_list = np.random.choice(pm.id_list, n_replace, replace=False)
    for key in id_list:
        inds = pm.get_mol_slice(key)
        params_mol[inds] = rstate.uniform(lb_mol[inds], ub_mol[inds])
        inds = pm.get_den_slice(key)
        if inds is not None:
            params_den[inds] = np.random.uniform(lb_den[inds], ub_den[inds])
    params_new = np.concatenate([params_mol, params_den, params_misc])
    return params_new


def create_optimizer(config_opt: dict) -> Optimizer:
    method = config_opt["method"]
    kwargs = config_opt.get("kwargs_opt", {})
    if method in ("pso", "abc"):
        cls_opt = SwingOptimizer
    else:
        cls_opt = ScipyOptimizer
    sig = inspect.signature(cls_opt)
    default_keys = tuple(k for k, v in sig.parameters.items()
                         if v.default is not inspect.Parameter.empty)
    for key in default_keys:
        if key in config_opt:
            kwargs[key] = config_opt[key]
    return cls_opt(method, **kwargs)


class Optimizer(ABC):
    def __init__(self, n_draw=1):
        self._n_draw = n_draw

    @abstractmethod
    def __call__(self, fitting_model, *args) -> dict:
        """
        This method may have the following input signatures:

          - ``len(args) == 0``: Default mode.
          - ``len(args) == 1``: Possible initial guess is provided.
          - ``len(args) > 1``: Possible initial guess ``(n_draw, D)``
            is provided.
        """
        raise NotImplementedError

    @property
    def n_draw(self):
        return self._n_draw


class SwingOptimizer(Optimizer):
    def __init__(self, method, n_draw=32, n_cycle_min=100, n_cycle_max=1000,
                 n_cycle_dim=5, n_stop=15, tol_stop=1.e-5, n_trail=1,
                 save_history=True, save_all=False, **kwargs):
        super().__init__(n_draw=n_draw)
        self._method = method
        self._n_cycle_min = n_cycle_min
        self._n_cycle_max = n_cycle_max
        self._n_cycle_dim = n_cycle_dim
        self._n_stop = n_stop
        self._tol_stop = tol_stop
        self._n_trail = n_trail
        self._save_history = save_history
        self._save_all = save_all
        self._kwargs = kwargs

    def __call__(self, fitting_model, *args, pool=None) -> dict:
        res_list = []
        for _ in range(self._n_trail):
            res_list.append(self._run_sub(fitting_model, *args, pool=pool))
        res_dict = deepcopy(min(res_list, key=lambda x: x["fun"]))
        if len(res_list) > 1:
            for res in res_list:
                del res["specie"]
            res_dict["trial"] = {f"{idx}": res for idx, res in enumerate(res_list)}
        if self._save_all:
            cost_all = np.concatenate([res["cost_all"] for res in res_list])
            pos_all = np.vstack([res["pos_all"] for res in res_list])
            blob = []
            for res in res_list:
                blob.extend(res["blob"])
            res_dict["cost_all"] = cost_all
            res_dict["pos_all"] = pos_all
            res_dict["blob"] = blob
        return res_dict

    def _run_sub(self, fitting_model, *args, pool=None) -> dict:
        if self._method == "pso":
            cls_opt = ParticleSwarm
        elif self._method == "abc":
            cls_opt = ArtificialBeeColony
        else:
            raise ValueError("Unknown optimizer: {}.".format(self._method))

        blob = self._kwargs.get("blob", False)
        kwargs = deepcopy(self._kwargs)
        if len(args) > 0:
            kwargs["initial_pos"] = self.derive_initial_pos(
                args[0], fitting_model.bounds, kwargs["nswarm"]
            )
        opt = cls_opt(
            fitting_model,
            fitting_model.bounds,
            nswarm=self.n_draw,
            pool=pool,
            blob=blob,
            **self._kwargs
        )
        n_cycle_min = self._n_cycle_min \
            + (len(fitting_model.bounds) - 5)*self._n_cycle_dim
        n_cycle_max = self._n_cycle_max

        def compute_rate(opt):
            history = opt.memo["cost"]
            return abs(history[-2] - history[-1])/abs(history[-2])

        pos_all = []
        cost_all = []
        blob = []
        n_stuck = 0
        for i_cycle in range(n_cycle_max):
            for data in opt.swarm(niter=1, progress_bar=False).values():
                if self._save_all:
                    pos_all.append(data["pos"])
                    cost_all.append(data["cost"])
                    if data["blob"] is not None:
                        blob.extend(data["blob"])

            if i_cycle > 1:
                rate = compute_rate(opt)
                if rate < self._tol_stop:
                    n_stuck += 1
                else:
                    n_stuck = 0

            if n_stuck >= self._n_stop and i_cycle + 1 >= n_cycle_min:
                break

        if len(pos_all) != 0:
            pos_all = np.vstack(pos_all)
            cost_all = np.concatenate(cost_all)

        T_pred_data = fitting_model.sl_model(opt.pos_global_best)

        res_dict = {
            "x":  opt.pos_global_best,
            "fun": opt.cost_global_best,
            "nfev": opt.memo["ncall"][-1],
            "specie": fitting_model.sl_model.specie_list,
            "freq": fitting_model.sl_model.freq_data,
            "T_pred": T_pred_data,
        }
        if self._save_history:
            res_dict["history"] = opt.memo
        if self._save_all:
            res_dict["pos_all"] = pos_all
            res_dict["cost_all"] = cost_all
            res_dict["blob"] = np.asarray(blob)
        return res_dict

    def derive_initial_pos(self, x0, bounds, n_swarm):
        assert n_swarm >= 1

        lb, ub = bounds.T
        initial_pos = lb + (ub - lb)*np.random.rand(n_swarm - 1, 1)
        initial_pos = np.vstack([x0, initial_pos])
        return initial_pos


class UniformOptimizer(Optimizer):
    def __init__(self, n_draw=50):
        super().__init__(n_draw)

    def __call__(self, fitting_model, *args) -> dict:
        if len(args) == 0:
            lower, upper = fitting_model.bounds.T
            samps_ = lower \
                + (upper - lower)*np.random.rand(self._n_compute, len(lower))
        else:
            samps_ = args[0]

        values = tuple(map(fitting_model, samps_[:self._n_compute]))
        l_tot = np.asarray(values)
        idx = np.argmin(l_tot)
        x0 = samps_[idx]
        l_tot_min = l_tot[idx]

        return {
            "x":  x0,
            "fun": l_tot_min,
            "nfev": self._n_compute + 1,
            "specie": fitting_model.sl_model.specie_list,
            "freq": fitting_model.sl_model.freq_data,
            "T_pred": fitting_model.sl_model(x0),
        }


class ScipyOptimizer(Optimizer):
    def __init__(self, method, n_draw=50, jac="2-point"):
        super().__init__(n_draw)
        self._method = method
        self._jac = jac

    def __call__(self, fitting_model, *args) -> dict:
        if len(args) == 1:
            x0 = args[0]
            l_tot_min = fitting_model(x0)
            nfev = 1
        else:
            if len(args) == 0:
                lower, upper = fitting_model.bounds.T
                samps_ = lower \
                    + (upper - lower)*np.random.rand(self.n_draw, len(lower))
            else:
                samps_ = args[0]
            values = tuple(map(fitting_model, samps_))
            l_tot = np.asarray(values)
            x0 = samps_[np.argmin(l_tot)].astype("f8")
            l_tot_min = np.min(l_tot)
            nfev = self.n_draw

        kwargs = {"method": self._method}
        if self._method in ("L-BFGS-B", "TNC", "SLSQP"):
            lower, upper = fitting_model.bounds.T
            h = 1.e-5
            kwargs.update(
                jac=self._jac,
                options={"eps": h*(upper - lower)}
            )
        res = minimize(
            fitting_model,
            x0=x0,
            bounds=fitting_model.bounds,
            **kwargs
        )

        if res.fun < l_tot_min:
            x_best = res.x
            fun = res.fun
        else:
            x_best = x0
            fun = l_tot_min

        nfev += res.nfev + 1

        return {
            "x":  x_best,
            "fun": fun,
            "nfev": nfev,
            "specie": fitting_model.sl_model.specie_list,
            "freq": fitting_model.sl_model.freq_data,
            "T_pred": fitting_model.sl_model(x_best),
            #"success": success
        }