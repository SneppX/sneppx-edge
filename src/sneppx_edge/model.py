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

    def __init__(self, weight, bias=None, scale=None, zero_point=None, dtype="float32"):
        self.weight = weight
        self.bias = bias
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype
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

    # -- quantization ------------------------------------------------------

    def quantize(self):
        """Return a new ``QuantizedModel`` with weights quantized to uint8."""
        wmin = min(min(row) for row in self.weight)
        wmax = max(max(row) for row in self.weight)
        span = wmax - wmin if wmax != wmin else 1.0
        scale = span / 255.0
        zero_point = round(-wmin / scale)
        zero_point = max(0, min(255, zero_point))
        quantized = []
        for row in self.weight:
            quantized.append([max(0, min(255, round(v / scale + zero_point))) for v in row])
        m = QuantizedModel(quantized, bias=list(self.bias) if self.bias else None,
                           scale=scale, zero_point=zero_point, dtype="uint8")
        return m

    # -- persistence -------------------------------------------------------

    def save(self, path):
        path = pathlib.Path(path)
        data = {
            "format": "sneppx-edge-v1",
            "dtype": self.dtype,
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
        )

    # -- internals ---------------------------------------------------------

    def _dequantized_weights(self):
        if self.dtype == "uint8" and self.scale is not None:
            return [[(v - self.zero_point) * self.scale for v in row]
                    for row in self.weight]
        return [list(row) for row in self.weight]

    def __repr__(self):
        return (f"QuantizedModel(out={self.out_features}, in={self.in_features}, "
                f"dtype={self.dtype}, quantized={self.dtype == 'uint8'})")


def quantize_model(weight, bias=None):
    """Convenience function: float model in, quantized out."""
    return QuantizedModel(weight, bias=bias).quantize()