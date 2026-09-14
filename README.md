# SNEPPX Edge - On-Device Inference SDK

Quantized, small-footprint runtime for running SneppX models on phones and
embedded targets (Vulkan/Metal/ROCm backends).

> Status: skeleton (WIP)

## Layout
- `src/sneppx_edge/runtime.py` - runtime loader skeleton
- `src/sneppx_edge/model.py` - quantized model stub
- `tests/` - smoke tests

## Roadmap
- [ ] quantized 4-bit load/infer
- [ ] Vulkan/Metal backend adapters
- [ ] per-device licensing (commercial tier)

## License
MIT core - commercial license required for proprietary/Edge use.
