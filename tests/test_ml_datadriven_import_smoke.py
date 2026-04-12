import importlib


def test_ml_datadriven_import_smoke() -> None:
    assert importlib.import_module("sciona.atoms.ml.datadriven") is not None
    assert importlib.import_module("sciona.probes.ml.datadriven") is not None
