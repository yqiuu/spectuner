from __future__ import annotations
import multiprocessing as mp
from typing import Optional, Union
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from swing import ParticleSwarm, ArtificialBeeColony
from scipy.optimize import minimize, least_squares
from scipy.cluster.vq import kmeans2

from .. import ai
from ..slm_factory import jit_fitting_model, SpectralLineModelFactory
from ..utils import pick_default_kwargs


def optimize(engine: Union[SpectralLineModelFactory, ai.InferenceModel],
             obs_info: list,
             specie_list: list,
             config: dict,
             T_base_data: Optional[list]=None,
             x0: Optional[np.ndarray]=None):
    opt = create_optimizer(config)
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
    elif isinstance(engine, ai.InferenceModel) is not None:
        return engine.call_single(
            obs_info=obs_info,
            specie_name=specie_list[0]["root"],
            postprocess=opt,
            T_base_data=T_base_data,
            device=config["inference"]["device"],
        )
    else:
        raise ValueError(f"Unknown engine: {engine}.")


def optimize_all(engine: Union[SpectralLineModelFactory, ai.InferenceModel],
                 obs_info: list,
                 targets: list,
                 config: dict,
                 T_base_data: list=None,
                 trans_counts: dict=None,
                 pool: mp.Pool=None):
    # Optimize without inference model
    if isinstance(engine, SpectralLineModelFactory):
        results = []
        with tqdm(total=len(targets), desc="Fitting") as pbar:
            callback = lambda _: pbar.update()
            for specie_list in targets:
                fitting_model = engine.create_fitting_model(
                    obs_info=obs_info,
                    specie_list=specie_list,
                    T_base_data=T_base_data,
                )
                if pool is None:
                    res = _optimize_worker(fitting_model, config)
                    pbar.update()
                else:
                    res = pool.apply_async(
                        _optimize_worker,
                        args=(fitting_model, config),
                        callback=callback
                    )
                results.append(res)

            if pool is not None:
                results = [res.get() for res in results]
        return results
    elif isinstance(engine, ai.InferenceModel):
        specie_names = []
        numbers = []
        for specie_list in targets:
            name = specie_list[0]["root"]
            specie_names.append(name)
            numbers.append(trans_counts[name])
        opt = create_optimizer(config)
        config_inf = config["inference"]
        results = ai.predict_single_pixel(
            inf_model=engine,
            obs_info=obs_info,
            entries=specie_names,
            numbers=numbers,
            postprocess=opt,
            max_diff=config_inf["max_diff"],
            max_batch_size=config_inf["batch_size"],
            device=config_inf["device"],
            pool=pool
        )
        for res, specie_list in zip(results, targets):
            res["specie"] = specie_list
        return results
    else:
        raise ValueError(f"Unknown engine: {engine}.")


def _optimize_worker(fitting_model, config):
    opt = create_optimizer(config)
    jit_fitting_model(fitting_model)
    return opt(fitting_model)


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


def create_optimizer(config: dict) -> Optimizer:
    config_opt = config["optimizer"]
    method = config_opt["method"]
    if method == "auto":
        if config["inference"]["ckpt"] is None:
            method = "pso"
        else:
            method = "slsqp"

    kwargs = config_opt.get("kwargs_opt", {})
    if method == "vanilla":
        cls_opt = VanillaOptimizer
    elif method in ("pso", "abc"):
        cls_opt = SwingOptimizer
    elif method in ("trf", "dogbox", "lm"):
        cls_opt = LeastSquares
    else:
        cls_opt = ScipyOptimizer

    kwargs.update(pick_default_kwargs(cls_opt, config_opt))
    # method is modified if it is 'auto', and therefore we need to update method
    kwargs.update(method=method)
    return cls_opt(**kwargs)


class Optimizer(ABC):
    def __init__(self, n_draw=50):
        self._n_draw = n_draw

    def __call__(self, fitting_model, *args) -> dict:
        res = self._optimize(fitting_model, *args)
        res["specie"] = fitting_model.sl_model.specie_list
        res["freq"] = fitting_model.sl_model.freq_data
        res["T_pred"] = fitting_model.sl_model(res["x"])
        return res

    @abstractmethod
    def _optimize(self, fitting_model, *args) -> dict:
        """
        This method may have the following input signatures:

          - ``len(args) == 0``: Default mode.
          - ``len(args) == 1``: Possible initial guess is provided.
          - ``len(args) > 1``: Possible initial guess ``(n_draw, D)``
            is provided.
        """

    @property
    def n_draw(self):
        return self._n_draw


class SwingOptimizer(Optimizer):
    def __init__(self,
                 method: str,
                 n_swarm: int=28,
                 n_cycle_min: int=100,
                 n_cycle_max: int=1000,
                 n_cycle_dim: int=5,
                 n_stop: int=15,
                 tol_stop: float=1.e-5,
                 n_trial: int=1,
                 save_history: bool=True,
                 save_all: bool=False,
                 **kwargs):
        super().__init__(n_draw=n_swarm)
        self._method = method
        self._n_cycle_min = n_cycle_min
        self._n_cycle_max = n_cycle_max
        self._n_cycle_dim = n_cycle_dim
        self._n_stop = n_stop
        self._tol_stop = tol_stop
        self._n_trial = n_trial
        self._save_history = save_history
        self._save_all = save_all
        self._kwargs = kwargs

    def _optimize(self, fitting_model, *args, pool=None) -> dict:
        res_list = []
        for _ in range(self._n_trial):
            res_list.append(self._optimize_sub(fitting_model, *args, pool=pool))
        res_dict = deepcopy(min(res_list, key=lambda x: x["fun"]))
        if len(res_list) > 1:
            res_dict["trial"] = {f"{idx}": res for idx, res in enumerate(res_list)}
            res_dict["nfev"] = sum(res["nfev"] for res in res_list)
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

    def _optimize_sub(self, fitting_model, *args, pool=None) -> dict:
        if self._method == "pso":
            cls_opt = ParticleSwarm
        elif self._method == "abc":
            cls_opt = ArtificialBeeColony
        else:
            raise ValueError("Unknown optimizer: {}.".format(self._method))

        blob = self._kwargs.get("blob", False)
        kwargs = deepcopy(self._kwargs)
        nfev = 0
        if len(args) == 1:
            kwargs["initial_pos"] = self.derive_initial_pos(
                args[0], fitting_model.bounds, self.n_draw
            )
        elif len(args) > 1:
            kwargs["initial_pos"] = args[0]
        opt = cls_opt(
            fitting_model,
            fitting_model.bounds,
            nswarm=self.n_draw,
            pool=pool,
            blob=blob,
            **kwargs
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
            "nfev": opt.memo["ncall"][-1] + nfev,
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
        x0 = np.atleast_2d(x0)
        n_rand = n_swarm - x0.shape[0]
        if n_rand == 0:
            return x0
        elif n_rand > 0:
            lb, ub = bounds.T
            x_rand = lb + (ub - lb)*np.random.rand(n_rand, 1)
            return np.vstack([x0, x_rand])

        raise ValueError("n_rand < 0")


class VanillaOptimizer(Optimizer):
    def __init__(self, n_draw=50):
        super().__init__(n_draw)

    def _optimize(self, fitting_model, *args) -> dict:
        if len(args) == 0:
            lower, upper = fitting_model.bounds.T
            samps_ = lower \
                + (upper - lower)*np.random.rand(self.n_draw, len(lower))
        else:
            samps_ = args[0]

        fun_all = tuple(map(fitting_model, samps_[:self.n_draw]))
        fun_all = np.asarray(fun_all)
        idx = np.argmin(fun_all)
        x0 = samps_[idx]
        fun_min = fun_all[idx]

        return {
            "x":  x0,
            "fun": fun_min,
            "nfev": self.n_draw,
            "x_all": samps_,
            "fun_all": fun_all
        }


class ScipyOptimizer(Optimizer):
    def __init__(self,
                 method: str,
                 n_draw: int=50,
                 jac: str="3-point",
                 n_cluster: int=1,
                 maxiter: int=2000):
        super().__init__(n_draw)
        self._method = method.lower()
        self._jac = jac
        self._n_cluster = n_cluster
        self._maxiter = maxiter

    def _optimize(self, fitting_model, *args) -> dict:
        kwargs = {"method": self._method, "options": {"maxiter": self._maxiter}}
        if self._method in ("l-bfgs-b", "tnc", "slsqp"):
            lower, upper = fitting_model.bounds.T
            kwargs.update(jac=self._jac)

        if len(args) == 1:
            x0 = args[0]
            res = minimize(
                fitting_model,
                x0=x0,
                bounds=fitting_model.bounds,
                **kwargs
            )
            nfev = res.nfev
            x_best = res.x
            fun_best = res.fun
        else:
            if len(args) == 0:
                lower, upper = fitting_model.bounds.T
                samps_ = lower \
                    + (upper - lower)*np.random.rand(self.n_draw, len(lower))
            else:
                samps_ = args[0]
            values = tuple(map(fitting_model, samps_))
            values = np.asarray(values)

            if self._n_cluster > 1 and len(args) > 1:
                clusters = _clustering(samps_, values, self._n_cluster)
                results = []
                for sub in clusters.values():
                    x0, _ = min(sub, key=lambda x: x[1])
                    results.append(minimize(
                        fitting_model,
                        x0=x0,
                        bounds=fitting_model.bounds,
                        **kwargs
                    ))
                res = min(results, key=lambda x: x.fun)
                nfev = sum(res.nfev for res in results)
            else:
                x0 = samps_[np.argmin(values)].astype("f8")
                res = minimize(
                    fitting_model,
                    x0=x0,
                    bounds=fitting_model.bounds,
                    **kwargs
                )
                nfev = len(values) + res.nfev

            idx = np.argmin(values)
            if res.fun < values[idx]:
                x_best = res.x
                fun_best = res.fun
            else:
                x_best = samps_[idx]
                fun_best = values[idx]

        return {
            "x":  x_best,
            "fun": fun_best,
            "nfev": nfev,
        }


def _clustering(samps, values, n_cluster):
    mu = np.mean(samps, axis=0)
    std = np.std(samps, axis=0)
    samps_scaled = (samps - mu)/std
    _, labels = kmeans2(samps_scaled, n_cluster, minit="++")
    clusters = defaultdict(list)
    for i, x, v in zip(labels, samps, values):
        clusters[i].append((x, v))
    return clusters


class LeastSquares(Optimizer):
    def __init__(self,
                 method: str,
                 n_draw: int=50,
                 jac: str="3-point",
                 maxiter: int=2000):
        super().__init__(n_draw)
        self._method = method
        self._jac = jac
        self._maxiter = maxiter

    def _optimize(self, fitting_model, *args) -> dict:
        lower, upper = fitting_model.bounds.T
        kwargs = {
            "method": self._method,
            "jac": self._jac,
            "bounds": (lower, upper),
            "x_scale": "jac",
            "max_nfev": self._maxiter,
        }

        if len(args) == 1:
            x0 = args[0]
            res = least_squares(
                fitting_model,
                x0=x0,
                **kwargs
            )
            nfev = res.nfev
            x_best = res.x
            fun_best = res.cost
        else:
            if len(args) == 0:
                lower, upper = fitting_model.bounds.T
                samps_ = lower \
                    + (upper - lower)*np.random.rand(self.n_draw, len(lower))
            else:
                samps_ = args[0]
            values = np.asarray(tuple(map(fitting_model, samps_)))
            values = .5*np.sum(np.square(values), axis=1)

            idx_min = np.argmin(values)
            x0 = samps_[idx_min].astype("f8")
            res = least_squares(
                fitting_model,
                x0=x0,
                **kwargs
            )
            nfev = len(values) + res.nfev

            if res.cost < values[idx_min]:
                x_best = res.x
                fun_best = res.cost
            else:
                x_best = samps_[idx_min]
                fun_best = values[idx_min]

        return {
            "x":  x_best,
            "fun": fun_best,
            "nfev": nfev,
        }