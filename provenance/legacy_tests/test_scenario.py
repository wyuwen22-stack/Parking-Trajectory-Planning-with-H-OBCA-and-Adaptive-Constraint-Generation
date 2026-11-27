from pathlib import Path

import numpy as np
from scipy.io import loadmat

from traj_py import simple_scenario


def load_reference(name: str) -> dict:
    return loadmat(Path(__file__).parent / "reference" / name, simplify_cells=True)


def assert_numeric_equal(actual, expected, name: str) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    difference = np.max(np.abs(actual - expected))
    print(f'{name}: max_absolute={difference:.16g}')
    assert difference <= 1e-12, f"{name}: max absolute error {difference}"


def test_simple_scenario_matches_matlab_reference() -> None:
    scenario = simple_scenario()
    reference = load_reference("scenario_simple_reference.mat")
    veh = reference["veh"]
    map_data = reference["map"]

    assert_numeric_equal(scenario.veh.L, veh["L"], "veh.L")
    assert_numeric_equal(scenario.veh.ego, veh["ego"], "veh.ego")
    assert_numeric_equal(scenario.veh.Rmin, veh["Rmin"], "veh.Rmin")
    assert_numeric_equal(scenario.resolution, reference["resolution"], "resolution")
    for name in ("XYbounds", "start", "goal"):
        assert_numeric_equal(getattr(scenario, name), map_data[name], f"map.{name}")
    assert_numeric_equal(len(scenario.obs), map_data["nOb"], "map.nOb")
    for name in ("obs", "Aineq", "bineq"):
        actual = getattr(scenario, name)
        expected = map_data[name]
        assert len(actual) == len(expected), f"map.{name}: count differs"
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_numeric_equal(left, right, f"map.{name}[{index}]")

    occupancy = reference["occupancy"]
    assert scenario.occupancy.shape == occupancy.shape == (40, 250)
    mismatches = np.count_nonzero(scenario.occupancy != occupancy)
    assert mismatches == 0, f"occupancy: {mismatches} cells differ"
