# Local model assets

Large model files are intentionally excluded from Git. This directory receives:

- `sensevoice_time100.onnx`
- `sensevoice_time100_fp.rknn`
- `tokens.json`

The ONNX input contract is:

- `speech`: float32, `[1, 96, 560]`
- `speech_lengths`: float32, `[1]`, valid acoustic-frame count from 1 through 96

The output contract is `logits`: float32, `[1, 100, 25055]`.
