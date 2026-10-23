import sys
from pathlib import Path

import numpy as np
import torch

from one_shape_bo.physical import (
    cached_fem,
    configured_loads_and_supports,
    generate_geometry,
    true_objective,
)


def test_source_physical_model_matches_legacy_for_mixed_design():
    legacy_root = Path("legacy/One_Shape_Only").resolve()
    sys.path.insert(0, str(legacy_root))
    try:
        from Functions.Physical import Generate_Geometry, True_Objective

        design = torch.tensor(
            [
                0.0, 0.10, 0.15, 0.65, 0.50, 0.52, 0.75,
                1.0, 0.72, 0.55, 0.12, 0.18, 0.30, 0.00,
            ],
            dtype=torch.double,
        )
        current = generate_geometry(30, 15, design, nholes=2)
        legacy = Generate_Geometry(30, 15, design, nholes=2)

        assert np.array_equal(current, legacy)
        assert true_objective(current, 30, 15, 1e-9, 1.0, 3.0) == (
            True_Objective(legacy, 30, 15, 1e-9, 1.0, 3.0)
        )
    finally:
        sys.path.remove(str(legacy_root))


def test_half_beam_adds_left_symmetry_and_custom_support_and_load():
    loads, fixed = configured_loads_and_supports(
        4,
        2,
        beam_model="half",
        support_conditions=((3, 0, "v"),),
        load_conditions=((0, 2, "v", -0.5),),
    )

    assert loads == [(2, 1, -0.5)]
    assert set(fixed) == {0, 2, 4, 19}


def test_full_beam_has_only_explicit_supports_and_sums_nodal_loads():
    _, _, _, _, _, free, load_vector = cached_fem(
        4,
        2,
        beam_model="full",
        support_conditions=((1, 0, "both"), (3, 0, "v")),
        load_conditions=(
            (2, 2, "v", -1.0),
            (2, 2, "v", -0.5),
            (0, 1, "u", 2.0),
        ),
    )

    fixed = set(range(len(load_vector))) - set(free)
    assert fixed == {6, 7, 19}
    assert 0 not in fixed
    assert load_vector[2] == 2.0
    assert load_vector[17] == -1.5


def test_full_beam_custom_conditions_produce_finite_compliance():
    compliance = true_objective(
        np.ones((2, 4)),
        4,
        2,
        1e-9,
        1.0,
        3.0,
        beam_model="full",
        support_conditions=((0, 0, "both"), (4, 0, "v")),
        load_conditions=((2, 2, "v", -1.0),),
    )

    assert np.isfinite(compliance)
    assert compliance > 0
