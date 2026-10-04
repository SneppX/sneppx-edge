import importlib


def test_version_constant():
    mod = importlib.import_module("sneppx_edge")
    assert getattr(mod, "__version__", None) == "0.1.0"
