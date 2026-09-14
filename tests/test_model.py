import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sneppx_edge.model import QuantizedModel, quantize_model  # noqa: E402
from sneppx_edge.runtime import Runtime, RuntimeError  # noqa: E402

W = [[0.5, -1.0, 2.0], [1.0, 1.0, 0.25]]
B = [0.1, -0.2]


def test_forward_linear():
    m = QuantizedModel(W, bias=B)
    out = m.forward([1.0, 2.0, 3.0])
    assert abs(out[0] - (0.5 - 2.0 + 6.0 + 0.1)) < 1e-9
    assert abs(out[1] - (1.0 + 2.0 + 0.75 - 0.2)) < 1e-9


def test_quantize_roundtrip_close():
    m = QuantizedModel(W, bias=B)
    q = m.quantize()
    assert q.dtype == "uint8"
    out_q = q.forward([1.0, 2.0, 3.0])
    out_f = m.forward([1.0, 2.0, 3.0])
    for a, b_ in zip(out_q, out_f):
        assert abs(a - b_) < 0.05


def test_save_load(tmp_path):
    m = QuantizedModel(W, bias=B).quantize()
    p = m.save(tmp_path / "model.json")
    loaded = QuantizedModel.load(p)
    assert loaded.out_features == 2
    assert loaded.dtype == "uint8"
    assert loaded.scale == m.scale


def test_runtime_lifecycle(tmp_path):
    path = QuantizedModel(W, bias=B).save(tmp_path / "m.json")
    rt = Runtime()
    try:
        rt.infer([1.0])
        assert False, "should raise before load"
    except RuntimeError:
        pass
    rt.load(path, quantize=True)
    assert rt.model.dtype == "uint8"
    out = rt.infer([1.0, 2.0, 3.0])
    assert len(out) == 2


def test_quantize_model_helper():
    q = quantize_model(W, bias=B)
    assert q is not None
    assert q.dtype == "uint8"