from gui.measurement_diagnostics_viewer import _format, _ratio_status


def test_measurement_formatting_helpers():
    assert _format(.125, "fraction") == "12.50%"
    assert _ratio_status(1.05) == "matched"
    assert _ratio_status(.5) == "deficit"
    assert _ratio_status(1.5) == "excess"
