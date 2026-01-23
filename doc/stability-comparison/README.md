# Comparison of numerical stability


We've compared the numerical stability of an embedding model to get an impression how big the impact is when changing inference runtimes, parameters, quantization, and hardware backends.

## Chosen model

We used embeddinggemma:300m.

For Q4_0 we used:

url = "https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/6661a6504c30d8304af13455cb4a5d4f5bc6011f/embeddinggemma-300m-Q4_0.gguf?download=true"
hash = "edc6015cb15694c27be7d1d33f1bc015db9a358ff51ed524628c027504907ba9"

For F32 we used:

url = "https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-F32.gguf"
hash = "a3125072128fc76d1c1d8d19f7b095c7e3bfbf00594dcf8a8bd3bcb334935d57"

The request was:

curl http://127.0.0.1:8000/models/embeddinggemma:300m/proxy/v1/embeddings   -H "Content-Type: application/json"   -d '{
    "input": "The food was delicious and the waiter...",
    "model": "embeddinggemma",
    "encoding_format": "float"
  }'


## Results

## Euclidean Distance

|                      |         1-m4-cpu-F32 |        2-m4-cpu-Q4_0 |       3-m4-metal-F32 |      4-m4-metal-Q4_0 |     5-W7900-rocm-F32 | 6-W7900-rocm_mmq-F32 |
| -------------------- | -------------------: | -------------------: | -------------------: | -------------------: | -------------------: | -------------------: |
| 1-m4-cpu-F32         |                      |               0.3958 |               0.0007 |               0.3951 |               0.0006 |               0.0006 |
| 2-m4-cpu-Q4_0        |                      |                      |               0.3957 |               0.0118 |               0.3957 |               0.3957 |
| 3-m4-metal-F32       |                      |                      |                      |               0.3950 |               0.0005 |               0.0005 |
| 4-m4-metal-Q4_0      |                      |                      |                      |                      |               0.3950 |               0.3950 |
| 5-W7900-rocm-F32     |                      |                      |                      |                      |                      |               0.0000 |
| 6-W7900-rocm_mmq-F32 |                      |                      |                      |                      |                      |                      |

## Cosine Similarity

|                      |         1-m4-cpu-F32 |        2-m4-cpu-Q4_0 |       3-m4-metal-F32 |      4-m4-metal-Q4_0 |     5-W7900-rocm-F32 | 6-W7900-rocm_mmq-F32 |
| -------------------- | -------------------: | -------------------: | -------------------: | -------------------: | -------------------: | -------------------: |
| 1-m4-cpu-F32         |                      |               0.9217 |               1.0000 |               0.9220 |               1.0000 |               1.0000 |
| 2-m4-cpu-Q4_0        |                      |                      |               0.9217 |               0.9999 |               0.9217 |               0.9217 |
| 3-m4-metal-F32       |                      |                      |                      |               0.9220 |               1.0000 |               1.0000 |
| 4-m4-metal-Q4_0      |                      |                      |                      |                      |               0.9220 |               0.9220 |
| 5-W7900-rocm-F32     |                      |                      |                      |                      |                      |               1.0000 |
| 6-W7900-rocm_mmq-F32 |                      |                      |                      |                      |                      |                      |
