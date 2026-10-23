import pytest
import torch

from one_shape_bo.bayesian import Bounds
from one_shape_bo.optimization import OptimizationConfig
from one_shape_bo.runtime import make_generator
from one_shape_bo.sampling import (
    burn_in_feasibility_limits,
    feasible_mask,
    sample_burn_in_feasible,
)


def _constraints(*, htype, bx=1.0, bx_left=None, bx_right=None, smin=5.0):
    del htype
    return Bounds(
        100,
        50,
        3,
        bx,
        1.0,
        0.5,
        smin,
        50.0,
        0.005,
        tri_q_min=0.1,
        bx_left=bx_left,
        bx_right=bx_right,
    )


def test_asymmetric_margins_preserve_legacy_fallback_and_shift_box():
    config = OptimizationConfig(bx=3.0, bx_left=5.0)
    legacy = _constraints(htype=0, bx=3.0)
    asymmetric = _constraints(
        htype=0, bx=3.0, bx_left=5.0, bx_right=2.5
    )

    assert config.resolved_bx_left == 5.0
    assert config.resolved_bx_right == 3.0
    assert legacy.box == (3.0, 97.0, 1.0, 49.0)
    assert asymmetric.box == (5.0, 97.5, 1.0, 49.0)


def test_burn_in_preflight_rejects_impossible_usable_area():
    constraints = _constraints(
        htype=0, bx_left=40.0, bx_right=40.0, smin=5.0
    )

    limits = burn_in_feasibility_limits(constraints, (0, 0, 0))

    assert limits["necessary_conditions_passed"] is False
    assert any("required void area" in item for item in limits["failures"])


@pytest.mark.parametrize("htype", [0, 1])
def test_dedicated_burn_in_sampler_collects_requested_tight_design(htype):
    constraints = _constraints(
        htype=htype, bx_left=5.0, bx_right=2.5, smin=5.0
    )

    designs, diagnostics = sample_burn_in_feasible(
        2,
        3,
        htype,
        constraints,
        50.0,
        generator=make_generator(0),
        return_diagnostics=True,
    )

    assert designs.shape == (2, 21)
    assert feasible_mask(designs, constraints).all()
    assert diagnostics["sampler"] == "burn_in_clearance_aware"
    assert diagnostics["accepted"] == 2
    assert [row["area_strategy"] for row in diagnostics["strata"]] == [
        "near_equal",
        "broad",
    ]
