import sys
from pathlib import Path

import numpy as np
import torch

from one_shape_bo.physical import generate_geometry, true_objective


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
