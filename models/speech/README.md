# Speech runtime assets

The deployment scripts place the following non-Git runtime files here:

```text
models/speech/sensevoice/sensevoice_time100_fp.rknn
models/speech/sensevoice/config.yaml
models/speech/sensevoice/am.mvn
models/speech/sensevoice/tokens.json
models/speech/tts/zh_CN-huayan-medium.onnx
models/speech/tts/zh_CN-huayan-medium.onnx.json
```

`models/manifest.sha256` records their checksums. Large models are shipped in
the box image or a release artifact, not ordinary Git history.
