DONE Model downloader
DONE Parse the model into embed_tokens tensor, layers (list of 24) norm (final RMSNorm)
- Wire up the math to verify its correct (some basic shi like silu(x @ gate.T) bla bla)

- Implement a model parser from safetensors to raw weight tensor in PT
- The prefill pass, store KV
- Write the autoregressive decode loop, append K and V to the KV cache
- Get it to generate coherent sentences. This is our naive baseline

