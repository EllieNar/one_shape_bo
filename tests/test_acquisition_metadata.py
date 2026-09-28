import torch

from one_shape_bo import bayesian


class SimpleConstraints:
    nholes = 3
    box_bounds = torch.stack(
        (torch.zeros(21, dtype=torch.double), torch.ones(21, dtype=torch.double))
    )
    nonlinear_constraints = []


def test_acquisition_preserves_mixed_tuple_source_without_geometry_output(
    monkeypatch, capsys
):
    captured = {}
    point = torch.tensor(
        [
            0, 0.10, 0.10, 0.55, 0.55, 0.60, 0.60,
            0, 0.30, 0.30, 0.55, 0.55, 0.60, 0.60,
            1, 0.70, 0.70, 0.10, 0.10, 0.25, 0.00,
        ],
        dtype=torch.double,
    )

    def fake_optimize(**kwargs):
        captured["fixed_features"] = kwargs["fixed_features_list"][0]
        return kwargs["batch_initial_conditions"].clone(), torch.tensor(2.0)

    monkeypatch.setattr(bayesian, "optimize_acqf_mixed", fake_optimize)
    monkeypatch.setattr(
        bayesian,
        "geometry_signature",
        lambda design, constraints: b"unique",
    )
    result = bayesian.Acquisition(
        acq_function=object(),
        constraints=SimpleConstraints(),
        initial_conditions=[{"point": point, "source": "global"}],
    )

    assert not result.failed_restarts
    assert len(result.successful_results) == 1
    assert result.successful_results[0]["source"] == "global"
    assert result.successful_results[0]["shape_tuple"] == (0, 0, 1)
    assert captured["fixed_features"] == {
        0: 0.0,
        7: 0.0,
        14: 1.0,
        20: 0.0,
    }
    assert capsys.readouterr().out == ""
