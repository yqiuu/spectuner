import pytest
import numpy as np
from numpy import testing
import spectuner



@pytest.mark.parametrize("theta_info", [
    {"is_shared": False, "is_log": True, "special": "scaled"},
    {"is_shared": False, "is_log": True, "special": "eta"},
])
def test_derive_recover(theta_info):
    param_info = {
        "theta": theta_info,
        "T_ex": {"is_shared": True, "is_log": False},
        "N_tot": {"is_shared": True, "is_log": True},
        "delta_v": {"is_shared": False, "is_log": True},
        "v_offset": {"is_shared": False, "is_log": False},
    }
    bounds_info = {
        "theta": [-3, -0.00043],
        "T_ex":  [1.0, 1000.0],
        "N_tot": [12.0, 20.0],
        "delta_v": [-0.5, 1.5],
        "v_offset": [-12., 12.],
    }
    for key, bound in bounds_info.items():
        param_info[key]["bound"] = bound
    specie_list = [
        {"id": 0, "root": None, "species": ["A", "B"]},
        {"id": 1, "root": None, "species": ["C", "D", "E"]},
        {"id": 2, "root": None, "species": ["F"]},
    ]
    obs_info = [
        {"beam_info": (1./3600, 1.5/3600)},
        {"beam_info": (5./3600, 1.2/3600)},
    ]

    param_mgr = spectuner.ParameterManager(specie_list, param_info, obs_info)
    rstate = np.random.RandomState(825)
    lb, ub = param_mgr.derive_bounds().T
    params = lb + (ub - lb)*rstate.rand(len(ub))
    params_mol = param_mgr.derive_params(params)
    params_re = param_mgr.recover_params(params_mol)
    testing.assert_allclose(params, params_re)