"""Minimal inference runtime facade for quantized models.

Provides load → quantize (optional) → infer in a single call.
"""

from sneppx_edge.model import QuantizedModel


class RuntimeError(Exception):
    """Runtime-level error."""


class Runtime:
    """Load and infer with a :class:`QuantizedModel`."""

    def __init__(self, backend="cpu"):
        self.backend = backend
        self._model = None

    def load(self, path, quantize=False, symmetric=False, per_channel=False):
        """Load a model; optionally quantize on load."""
        self._model = QuantizedModel.load(path)
        if quantize:
            self._model = self._model.quantize(
                symmetric=symmetric, per_channel=per_channel)
        return self._model

    def infer(self, inputs):
        """Run a single forward pass (batch of one)."""
        if self._model is None:
            raise RuntimeError("no model loaded; call load() first")
        return self._model.forward(inputs)

    def infer_batch(self, rows):
        """Run one forward pass per input row (batched)."""
        if self._model is None:
            raise RuntimeError("no model loaded; call load() first")
        return self._model.forward_batch(rows)

    @property
    def model(self):
        return self._model