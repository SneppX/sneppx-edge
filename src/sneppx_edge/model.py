"""On-device inference model with uint8 per-tensor quantization (no deps).

Quantizes float32 weight matrices to uint8 (symmetric/asymmetric), stores
the ``scale`` and ``zero_point``, and runs simple ``matmul + bias`` inference
in dequantized precision - enough to validate the full quant→infer→verify
pipeline for a model checkpoint format.
"""

import json
import math
import pathlib


class QuantizedModel:
    """Quantized (or unquantized) linear model for on-device inference.

    ``weight`` is a list-of-lists (rows = out_features, cols = in_features).
    ``bias`` is an optional list of length out_features.
    """

    def __init__(self, weight, bias=None, scale=None, zero_point=None,
                 dtype="float32", scheme=None):
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype
        if scheme is None:
            self.scheme = "uint8" if dtype == "uint8" else "float32"
        else:
            self.scheme = scheme
        self.in_features = len(weight[0]) if weight else 0
        self.out_features = len(weight) if weight else 0

    # -- inference ---------------------------------------------------------

    def forward(self, inputs):
        """Run a single linear pass ``inputs @ weight.T + bias``.

        ``inputs`` is a list of length ``in_features``. Returns a list of
        length ``out_features``.
        """
        w = self._dequantized_weights()
        out = []
        for row in w:
            s = sum(x * ww for x, ww in zip(inputs, row))
            if self.bias is not None:
                s += self.bias[len(out)]
            out.append(s)
        return out

    def forward_batch(self, rows):
        """Run one linear pass per input row.

        ``rows`` is a list of length-N feature lists; returns a list of
        length-N output lists (``inputs @ weight.T + bias`` for each row).
        """
        w = self._dequantized_weights()
        outs = []
        for inputs in rows:
            out = []
            for row in w:
                s = sum(x * ww for x, ww in zip(inputs, row))
                if self.bias is not None:
                    s += self.bias[len(out)]
                out.append(s)
            outs.append(out)
        return outs

    # -- quantization ------------------------------------------------------

    def quantize(self, symmetric=False, per_channel=False):
        """Return a new ``QuantizedModel`` with weights quantized to uint8.

        *symmetric*: scale uses ``max(|min|,|max|)/127`` with the neutral
        value at zero_point 128 (better for activation-style weight ranges
        centered on zero). Default asymmetric keeps the full [0, 255] range.

        *per_channel*: one scale/zero_point per output row instead of a
        single tensor-wide pair (better for outlier rows, slightly larger
        metadata).
        """
        if per_channel:
            scales, zero_points, quantized = [], [], []
            for row in self.weight:
                sc, zp, qrow = _quantize_row(row, symmetric)
                scales.append(sc)
                zero_points.append(zp)
                quantized.append(qrow)
            scheme = ("symmetric-" if symmetric else "asymmetric-") + "channel"
            return QuantizedModel(
                quantized,
                bias=list(self.bias) if self.bias else None,
                scale=scales, zero_point=zero_points,
                dtype="uint8", scheme=scheme,
            )
        scale, zero_point, quantized = _quantize_tensor(self.weight, symmetric)
        scheme = ("symmetric-" if symmetric else "asymmetric-") + "tensor"
        return QuantizedModel(
            quantized,
            bias=list(self.bias) if self.bias else None,
            scale=scale, zero_point=zero_point,
            dtype="uint8", scheme=scheme,
        )

    # -- persistence -------------------------------------------------------

    def save(self, path):
        path = pathlib.Path(path)
        data = {
            "format": "sneppx-edge-v1",
            "dtype": self.dtype,
            "scheme": self.scheme,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "weight": self.weight,
            "bias": self.bias,
            "scale": self.scale,
            "zero_point": self.zero_point,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path):
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        return cls(
            weight=data["weight"],
            bias=data.get("bias"),
            scale=data.get("scale"),
            zero_point=data.get("zero_point"),
            dtype=data.get("dtype", "float32"),
            scheme=data.get("scheme"),
        )

    # -- internals ---------------------------------------------------------

    def _dequantized_weights(self):
        if self.dtype == "uint8" and self.scale is not None:
            if isinstance(self.scale, list):
                # per-channel scheme: scale/zero_point parallel to rows
                return [
                    [(v - zp) * sc for v in row]
                    for row, sc, zp in zip(self.weight, self.scale, self.zero_point)
                ]
            return [[(v - self.zero_point) * self.scale for v in row]
                    for row in self.weight]
        return [list(row) for row in self.weight]

    def __repr__(self):
        return (f"QuantizedModel(out={self.out_features}, in={self.in_features}, "
                f"dtype={self.dtype}, scheme={self.scheme})")


def _quantize_tensor(weight, symmetric):
    wmin = min(min(row) for row in weight)
    wmax = max(max(row) for row in weight)
    if symmetric:
        span = max(abs(wmin), abs(wmax))
        scale = span / 127.0 if span != 0 else 1.0
        zero_point = 128
    else:
        span = wmax - wmin if wmax != wmin else 1.0
        scale = span / 255.0
        zero_point = round(-wmin / scale)
    quantized = [[max(0, min(255, round(v / scale + zero_point))) for v in row]
                 for row in weight]
    return scale, zero_point, quantized


def _quantize_row(row, symmetric):
    wmin, wmax = min(row), max(row)
    if symmetric:
        span = max(abs(wmin), abs(wmax))
        scale = span / 127.0 if span != 0 else 1.0
        zero_point = 128
    else:
        span = wmax - wmin if wmax != wmin else 1.0
        scale = span / 255.0
        zero_point = round(-wmin / scale)
    qrow = [max(0, min(255, round(v / scale + zero_point))) for v in row]
    return scale, zero_point, qrow


def quantize_model(weight, bias=None, symmetric=False, per_channel=False):
    """Convenience function: float model in, quantized out."""
    return QuantizedModel(weight, bias=bias).quantize(
        symmetric=symmetric, per_channel=per_channel)