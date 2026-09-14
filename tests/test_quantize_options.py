import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sneppx_edge.model import QuantizedModel, quantize_model  # noqa: E402
from sneppx_edge.runtime import Runtime  # noqa: E402
from sneppx_edge import cli  # noqa: E402

W = [[0.5, -1.0, 2.0], [1.0, 1.0, 0.25]]
B = [0.1, -0.2]


def test_forward_batch_matches_forward():
    m = QuantizedModel(W, bias=B)
    rows = [[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]]
    single = [m.forward(r) for r in rows]
    assert m.forward_batch(rows) == single


def test_quantize_symmetric_neutral_at_zero():
    m = QuantizedModel([[1.0, -1.0]], bias=None)
    q = m.quantize(symmetric=True)
    assert q.scale > 0
    assert q.zero_point == 128
    assert q.weight[0][1] == 256 - q.weight[0][0]  # symmetric about 128


def test_quantize_per_channel_scheme():
    m = QuantizedModel(W, bias=B)
    q = m.quantize(per_channel=True)
    assert q.scheme == "asymmetric-channel"
    assert isinstance(q.scale, list) and len(q.scale) == 2
    assert isinstance(q.zero_point, list) and len(q.zero_point) == 2
    out_q = q.forward([1.0, 2.0, 3.0])
    out_f = m.forward([1.0, 2.0, 3.0])
    for a, b_ in zip(out_q, out_f):
        assert abs(a - b_) < 0.05


def test_quantize_symmetric_scheme():
    m = QuantizedModel(W, bias=B)
    q = m.quantize(symmetric=True)
    assert q.scheme == "symmetric-tensor"
    assert q.zero_point == 128


def test_quantize_helper_options():
    q = quantize_model(W, bias=B, symmetric=True, per_channel=True)
    assert q.scheme == "symmetric-channel"


def test_scheme_persists(tmp_path):
    p = QuantizedModel(W, bias=B).quantize(per_channel=True).save(tmp_path / "m.json")
    loaded = QuantizedModel.load(p)
    assert loaded.scheme == "asymmetric-channel"
    assert isinstance(loaded.scale, list)


def test_per_channel_save_load_dequant(tmp_path):
    p = QuantizedModel(W, bias=B).quantize(per_channel=True).save(tmp_path / "m.json")
    loaded = QuantizedModel.load(p)
    wr = loaded._dequantized_weights()
    assert len(wr) == 2 and len(wr[0]) == 3
    for a, b_ in zip(loaded.forward([1.0, 2.0, 3.0]),
                     QuantizedModel(W, bias=B).forward([1.0, 2.0, 3.0])):
        assert abs(a - b_) < 0.05


def test_runtime_infer_batch(tmp_path):
    path = QuantizedModel(W, bias=B).save(tmp_path / "m.json")
    rt = Runtime()
    rt.load(path, quantize=True)
    outs = rt.infer_batch([[1.0, 2.0, 3.0], [0.0, 1.0, -1.0]])
    assert len(outs) == 2
    assert all(len(o) == 2 for o in outs)


def test_cli_quantize_symmetric_per_channel(tmp_path):
    sp = QuantizedModel(W, bias=B).save(tmp_path / "float.json")
    out = tmp_path / "q.json"
    rc = cli.main(["quantize", str(sp), str(out), "--symmetric", "--per-channel"])
    assert rc == 0
    loaded = QuantizedModel.load(out)
    assert loaded.scheme == "symmetric-channel"


def test_cli_infer_batch(tmp_path, capsys):
    sp = QuantizedModel(W, bias=B).save(tmp_path / "float.json")
    rc = cli.main(["infer-batch", str(sp),
                   "--rows", "1", "2", "3", "--rows", "0", "1", "-1"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert len(out) == 2 and len(out[0]) == 2


def test_cli_infer_batch_no_rows(tmp_path, capsys):
    sp = QuantizedModel(W, bias=B).save(tmp_path / "float.json")
    rc = cli.main(["infer-batch", str(sp)])
    assert rc == 2
    assert "error: supply" in capsys.readouterr().err