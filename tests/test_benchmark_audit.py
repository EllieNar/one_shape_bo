import numpy as np
import matplotlib.pyplot as plt

from one_shape_bo.benchmark import audit_reference_image


def test_benchmark_audit_is_read_only_and_reports_binary_metrics(tmp_path):
    image = np.ones((20, 40, 3), dtype=float)
    image[2:18, 3:37] = 0.0
    image[6:14, 12:28] = 1.0
    path = tmp_path / "reference.png"
    plt.imsave(path, image)
    before = path.read_bytes()

    report = audit_reference_image(
        path,
        nelx=10,
        nely=5,
        solid_max=0.9,
        raster_tolerance=0.0,
        logged_continuous_compliance=None,
    )

    assert path.read_bytes() == before
    assert report.grid_shape == (5, 10)
    assert 0.0 < report.solid_fraction < 1.0
    assert report.void_components >= 1
    assert report.recomputed_binary_compliance > 0.0


def test_repository_reference_audit_exposes_non_equivalent_volume():
    path = (
        "outputs/Expected_Output_TRIANGLE/"
        "Three_Hole_Expected_Result_vf05_nx100_ny50_c96_Rasterised.png"
    )

    report = audit_reference_image(path)

    assert report.grid_shape == (50, 100)
    assert report.void_components == 3
    assert report.solid_fraction == 0.5332
    assert not report.volume_feasible
    assert report.recomputed_binary_compliance > 0.0
