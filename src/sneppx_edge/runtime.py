class Runtime:
    """Runtime facade (skeleton)."""

    def __init__(self, backend="cpu"):
        self.backend = backend

    def load(self, path):
        from sneppx_edge.model import QuantizedModel
        return QuantizedModel(path)

    def infer(self, model, inputs):
        return model.forward(inputs)
