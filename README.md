# SNEPPX Edge - On-Device Inference SDK

Quantized, small-footprint runtime for running SneppX models on phones and
embedded targets (Vulkan/Metal/ROCm backends).

> Status: functional core (uint8 quantize + infer)

## Layout
- `src/sneppx_edge/runtime.py` - runtime loader / inference facade
- `src/sneppx_edge/model.py` - quantized model core
- `src/sneppx_edge/cli.py` - `quantize` / `infer` / `infer-batch` CLI
- `tests/` - 17 tests

## Quantization schemes
| Flag | Scheme | Notes |
|------|--------|-------|
| (default) | asymmetric tensor | single scale/zero_point for all weights |
| `--symmetric` | symmetric tensor | zero-centered, zero_point 128 |
| `--per-channel` | asymmetric channel | per-row scale/zero_point, best residual |
| `--symmetric --per-channel` | symmetric channel | both |

Batched inference is available via `infer-batch` / `Runtime.infer_batch`.

## Roadmap
- [x] uint8 quantize (`--symmetric`, `--per-channel`) + batch inference
- [ ] quantized 4-bit load/infer
- [ ] Vulkan/Metal backend adapters
- [ ] per-device licensing (commercial tier)

## License
MIT core - commercial license required for proprietary/Edge use.
