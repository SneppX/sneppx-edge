import argparse
import json
import sys

from sneppx_edge.model import QuantizedModel
from sneppx_edge.runtime import Runtime, RuntimeError


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sneppx-edge", description="on-device inference CLI")
    parser.add_argument("--version", action="version", version="sneppx-edge 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("quantize", help="quantize a float model checkpoint")
    q.add_argument("model", help="path to the float checkpoint")
    q.add_argument("out", help="path to write the uint8 checkpoint")

    inf = sub.add_parser("infer", help="run one forward pass")
    inf.add_argument("model", help="path to the checkpoint")
    inf.add_argument("inputs", nargs="+", type=float, help="input features as floats")
    inf.add_argument("--quantize", action="store_true", help="quantize on load")

    args = parser.parse_args(argv)

    if args.command == "quantize":
        qm = QuantizedModel.load(args.model).quantize()
        qm.save(args.out)
        print(f"quantized -> {args.out}")
        return 0

    if args.command == "infer":
        rt = Runtime()
        try:
            rt.load(args.model, quantize=args.quantize)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        try:
            out = rt.infer(list(args.inputs))
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(out))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())