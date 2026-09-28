from one_shape_bo.optimization import StagnationRestartController


def test_stagnation_changes_split_without_changing_budget_and_resets():
    controller = StagnationRestartController(
        ng_rst=8,
        nl_rst=8,
        patience=3,
        exploration_fraction=0.75,
    )

    assert controller.allocation() == (8, 8)
    controller.observe(False)
    controller.observe(False)
    assert controller.allocation() == (8, 8)
    controller.observe(False)
    assert controller.allocation() == (12, 4)
    assert sum(controller.allocation()) == 16
    controller.observe(True)
    assert controller.allocation() == (8, 8)
