from types import SimpleNamespace

import pytest

from one_shape_bo.gaussian import GPHealthDiagnostics
from one_shape_bo.optimization import (
    GPWarmStartController,
    OptimizationConfig,
    _assess_gp_health,
    _validate_health_configuration,
    _validate_restart_ranking_configuration,
)


class DummyHealthModel:
    def __init__(self, diagnostics, noise_mode="learned", prescribed=None):
        self._diagnostics = diagnostics
        self.noise_mode = noise_mode
        self.prescribed_observation_noise_variance = prescribed

    def health_diagnostics(self, points):
        return self._diagnostics


def health_diagnostics(signal=0.001, noise=1.0, nu=0.001):
    return GPHealthDiagnostics(
        signal_variance=signal,
        observation_noise_variance=noise,
        signal_to_noise_ratio=signal / noise,
        signal_fraction_nu=nu,
        representative_point_count=4,
        posterior_mean_spread_ratio=0.5,
        posterior_std_ratio=0.25,
    )


def test_learned_noise_warning_is_report_only_by_default():
    config = OptimizationConfig()
    result = _assess_gp_health(
        DummyHealthModel(health_diagnostics()),
        SimpleNamespace(),
        config,
        [],
    )

    assert result["health_warning_triggered"] is True
    assert result["health_warning_reasons"] == [
        "signal_fraction_nu_below_threshold"
    ]
    assert config.resolved_warm_start_policy == "always"
    assert result["health_action"] == "warn_only"
    assert result["next_fit_forced_cold"] is False


def test_health_reset_remains_available_for_learned_noise():
    config = OptimizationConfig(warm_start_policy="health_reset")
    result = _assess_gp_health(
        DummyHealthModel(health_diagnostics()),
        SimpleNamespace(),
        config,
        [],
    )

    assert result["health_action"] == "cold_fit_next_iteration"
    assert result["next_fit_forced_cold"] is True


def test_health_cold_start_is_consumed_by_next_fit_only():
    controller = GPWarmStartController()
    previous = object()
    controller.observe_health({"next_fit_forced_cold": True})

    assert controller.model_for_fit(previous) is None
    assert controller.model_for_fit(previous) is previous


def test_prescribed_noise_warning_is_report_only_by_default():
    config = OptimizationConfig(
        observation_noise_variance=1e-8,
        health_signal_variance_min=0.01,
    )
    result = _assess_gp_health(
        DummyHealthModel(
            health_diagnostics(signal=0.001, noise=1e-8, nu=0.99999),
            noise_mode="prescribed",
            prescribed=1e-8,
        ),
        SimpleNamespace(),
        config,
        [],
    )

    assert config.resolved_warm_start_policy == "always"
    assert result["health_warning_reasons"] == [
        "signal_variance_below_threshold"
    ]
    assert result["health_action"] == "warn_only"
    assert result["next_fit_forced_cold"] is False


def test_predictive_warning_waits_for_complete_window():
    config = OptimizationConfig(
        health_nu_min=0.0,
        health_predictive_z_max=2.0,
        health_predictive_window=2,
    )
    model = DummyHealthModel(health_diagnostics(signal=1.0, noise=0.1, nu=0.9))
    history = []

    first = _assess_gp_health(
        model,
        SimpleNamespace(),
        config,
        history,
        withheld_predictive_mean=0.0,
        withheld_predictive_std=1.0,
        withheld_observed_response=3.0,
    )
    second = _assess_gp_health(
        model,
        SimpleNamespace(),
        config,
        history,
        withheld_predictive_mean=0.0,
        withheld_predictive_std=1.0,
        withheld_observed_response=3.0,
    )

    assert first["health_warning_triggered"] is False
    assert second["health_warning_reasons"] == [
        "predictive_rms_z_above_threshold"
    ]
    assert second["predictive_rms_z"] == 3.0


@pytest.mark.parametrize("noise", [0.0, -1.0, float("inf")])
def test_prescribed_noise_must_be_finite_and_positive(noise):
    with pytest.raises(ValueError, match="finite and positive"):
        _validate_health_configuration(
            OptimizationConfig(observation_noise_variance=noise)
        )


def test_restart_ranking_weights_default_and_normalize():
    default = OptimizationConfig()
    scaled = OptimizationConfig(acquisition_weight=55.0, novelty_weight=45.0)

    assert default.resolved_restart_ranking_weights == (0.55, 0.45)
    assert scaled.resolved_restart_ranking_weights == (0.55, 0.45)
    _validate_restart_ranking_configuration(default)
    _validate_restart_ranking_configuration(scaled)


@pytest.mark.parametrize(
    "acquisition,novelty",
    [(0.0, 0.0), (-1.0, 1.0), (float("inf"), 1.0)],
)
def test_restart_ranking_config_rejects_invalid_weights(acquisition, novelty):
    with pytest.raises(ValueError):
        _validate_restart_ranking_configuration(
            OptimizationConfig(
                acquisition_weight=acquisition,
                novelty_weight=novelty,
            )
        )
