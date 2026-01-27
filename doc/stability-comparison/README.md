# Comparison of numerical stability of embedding models when moving from Ollama to llama-cpp

When switching inference engines from Ollama to llama-cpp-server we had to switch our
source of model files from the Ollama hub to huggingface.co. We were unsure whether
the results - specifically for the embedding models - would be sufficiently compatible
to our previous environment, so we launched a little investigation into the numerical
stability of one of the embedding models in the hope to extract general insights.

The parameters we looked at included: change of runtime, change of inference parameters of the runtime, quantization and hardware backends.

All tests were performed using `embeddinggemma-300m` - one of the embedding models from our initial selection in the Flying Circus AI platform.

# Summary

Overall stability was quite high. We saw the biggest deviation in model outputs when switching quantization - which corresponds with the intuitive expectations around quantization.

We also found that the GGUF variants of this specific model available on hugging face were missing the "dense modules" and thus resulted in vastly different output. At the
same time there were slight differences in the GGUF file formats between llama-cpp and ollama that didn't allow us to directly consume the model files for embeddinggemma from Ollama in llama-cpp.

As a result we created a custom model ([flyingcircusio/embeddinggemma-300m-GGUF-with-dense-modules](https://huggingface.co/flyingcircusio/embeddinggemma-300m-GGUF-with-dense-modules)) derived from the original embeddinggemma that
provides numerical stability when moving from Ollama to llama-cpp.

# Chosen model

We arbitrarily chose one of the embedding models we provide on our platform for in-depth analysis and the choice fell on [google/embeddinggemma-300m](https://huggingface.co/google/embeddinggemma-300m).

One of the parameters we wanted to test against was quantization.

The baseline in Ollama uses BF16:


As a high precision model in llama-cpp we used this F32 variant:


As a quantized model in llama-cpp we used this Q4_0 variant:

https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/6661a6504c30d8304af13455cb4a5d4f5bc6011f/embeddinggemma-300m-Q4_0.gguf (SHA256  edc6015cb15694c27be7d1d33f1bc015db9a358ff51ed524628c027504907ba9)

For F32 we used:

https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/resolve/main/embeddinggemma-300M-F32.gguf"
hash = "a3125072128fc76d1c1d8d19f7b095c7e3bfbf00594dcf8a8bd3bcb334935d57"

Finally, we provide a custom quantization to resolve the issues we encountered:


# Request

As input to the model we used the example sentence from the original's model card:

  $ curl http://127.0.0.1:8000/models/embeddinggemma:300m/proxy/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
      "input": "The food was delicious and the waiter...",
      "model": "embeddinggemma",
      "encoding_format": "float"
    }'

# Environment and parameter variations

The following parameters should help when trying to reproduce our results:

* `llama-cpp`: version: 6981 (647b960)
* `ollama`: 0.12.11 (rocm/cpu/metal)
* Hardware: Apple M2, Apple M4, AMD Radeon Pro W7900
* Drivers and arguments:
  * CPU-only (using `-ngl 0`)
  * rocm
  * rocm (with mmq CMAKE flag)
  * metal
  * metal (with dense modules)

# Results

The raw results are available in this folder as JSON files. The results have been summarized automatically using the `compare.py` script in this folder.

<!-- BEGIN comparison -->
## Euclidean Distance

|                          | 0-W7900-ollama_rocm-BF16 |             1-M4-cpu-F32 |            2-M4-cpu-Q4_0 |           3-M4-metal-F32 |          4-M4-metal-Q4_0 |        5-M2-metal_st-F32 |         6-W7900-rocm-F32 |        7-W7900-rocm-Q4_0 |     8-W7900-rocm_mmq-F32 |    9-W7900-rocm_mmq-Q4_0 |
| ------------------------ | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| 0-W7900-ollama_rocm-BF16 |                          |                   1.4196 |                   1.4167 |                   1.4196 |                   1.4169 |                   0.0077 |                   1.4196 |                   1.4173 |                   1.4196 |                   1.4173 |
| 1-M4-cpu-F32             |                          |                          |                   0.3958 |                   **0.0007** |                   0.3951 |                   **1.4196** |                   **0.0006** |                   0.3986 |                   **0.0006** |                   0.3986 |
| 2-M4-cpu-Q4_0            |                          |                          |                          |                   0.3957 |                   **0.0118** |                   1.4165 |                   0.3957 |                   **0.0390** |                   0.3957 |                   **0.0390** |
| 3-M4-metal-F32           |                          |                          |                          |                          |                   0.3950 |                   **1.4196** |                   **0.0005** |                   0.3986 |                   **0.0005** |                   0.3986 |
| 4-M4-metal-Q4_0          |                          |                          |                          |                          |                          |                   1.4167 |                   0.3950 |                   **0.0356** |                   0.3950 |                   **0.0356** |
| 5-M2-metal_st-F32        |                          |                          |                          |                          |                          |                          |                   **1.4196** |                   1.4171 |                   **1.4196** |                   1.4171 |
| 6-W7900-rocm-F32         |                          |                          |                          |                          |                          |                          |                          |                   0.3986 |                   **0.0000** |                   0.3986 |
| 7-W7900-rocm-Q4_0        |                          |                          |                          |                          |                          |                          |                          |                          |                   0.3986 |                   **0.0000** |
| 8-W7900-rocm_mmq-F32     |                          |                          |                          |                          |                          |                          |                          |                          |                          |                   0.3986 |

## Cosine Similarity

|                          | 0-W7900-ollama_rocm-BF16 |             1-M4-cpu-F32 |            2-M4-cpu-Q4_0 |           3-M4-metal-F32 |          4-M4-metal-Q4_0 |        5-M2-metal_st-F32 |         6-W7900-rocm-F32 |        7-W7900-rocm-Q4_0 |     8-W7900-rocm_mmq-F32 |    9-W7900-rocm_mmq-Q4_0 |
| ------------------------ | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| 0-W7900-ollama_rocm-BF16 |                          |                  -0.0077 |                  -0.0035 |                  -0.0077 |                  -0.0038 |                   1.0000 |                  -0.0077 |                  -0.0043 |                  -0.0077 |                  -0.0043 |
| 1-M4-cpu-F32             |                          |                          |                   0.9217 |                   **1.0000** |                   0.9220 |                  **-0.0076** |                   **1.0000** |                   0.9205 |                   **1.0000** |                   0.9205 |
| 2-M4-cpu-Q4_0            |                          |                          |                          |                   0.9217 |                   **0.9999** |                  -0.0032 |                   0.9217 |                   **0.9992** |                   0.9217 |                   **0.9992** |
| 3-M4-metal-F32           |                          |                          |                          |                          |                   0.9220 |                  **-0.0076** |                   **1.0000** |                   0.9206 |                   **1.0000** |                   0.9206 |
| 4-M4-metal-Q4_0          |                          |                          |                          |                          |                          |                  -0.0035 |                   0.9220 |                   **0.9994** |                   0.9220 |                   **0.9994** |
| 5-M2-metal_st-F32        |                          |                          |                          |                          |                          |                          |                  **-0.0076** |                  -0.0040 |                  **-0.0076** |                  -0.0040 |
| 6-W7900-rocm-F32         |                          |                          |                          |                          |                          |                          |                          |                   0.9206 |                   **1.0000** |                   0.9206 |
| 7-W7900-rocm-Q4_0        |                          |                          |                          |                          |                          |                          |                          |                          |                   0.9206 |                   **1.0000** |
| 8-W7900-rocm_mmq-F32     |                          |                          |                          |                          |                          |                          |                          |                          |                          |                   0.9206 |

## Angle

|                          | 0-W7900-ollama_rocm-BF16 |             1-M4-cpu-F32 |            2-M4-cpu-Q4_0 |           3-M4-metal-F32 |          4-M4-metal-Q4_0 |        5-M2-metal_st-F32 |         6-W7900-rocm-F32 |        7-W7900-rocm-Q4_0 |     8-W7900-rocm_mmq-F32 |    9-W7900-rocm_mmq-Q4_0 |
| ------------------------ | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: | -----------------------: |
| 0-W7900-ollama_rocm-BF16 |                          |                    90.4° |                    90.2° |                    90.4° |                    90.2° |                     0.4° |                    90.4° |                    90.2° |                    90.4° |                    90.2° |
| 1-M4-cpu-F32             |                          |                          |                    22.8° |                     **0.0°** |                    22.8° |                    **90.4°** |                     **0.0°** |                    23.0° |                     **0.0°** |                    23.0° |
| 2-M4-cpu-Q4_0            |                          |                          |                          |                    22.8° |                     **0.7°** |                    90.2° |                    22.8° |                     **2.2°** |                    22.8° |                     **2.2°** |
| 3-M4-metal-F32           |                          |                          |                          |                          |                    22.8° |                    **90.4°** |                     **0.0°** |                    23.0° |                     **0.0°** |                    23.0° |
| 4-M4-metal-Q4_0          |                          |                          |                          |                          |                          |                    90.2° |                    22.8° |                     **2.0°** |                    22.8° |                     **2.0°** |
| 5-M2-metal_st-F32        |                          |                          |                          |                          |                          |                          |                    **90.4°** |                    90.2° |                    **90.4°** |                    90.2° |
| 6-W7900-rocm-F32         |                          |                          |                          |                          |                          |                          |                          |                    23.0° |                     **0.0°** |                    23.0° |
| 7-W7900-rocm-Q4_0        |                          |                          |                          |                          |                          |                          |                          |                          |                    23.0° |                     **0.0°** |
| 8-W7900-rocm_mmq-F32     |                          |                          |                          |                          |                          |                          |                          |                          |                          |                    23.0° |

<!-- END comparison -->

# Findings and conclusions

1. Stability within quantisations over different inference engines, hardware and drivers is very high for high precision models and acceptably high for quantized models. Switching to lower quantizations has a certain degree of variation that might still be acceptable but could also warrant re-indexing depending on your use case.

2. Switching from ollama to llama-cpp showed stark deviations due to unobvious differences in their approach implementing the gemma model:

  * Ollama apparently chose to use a single code base to handle the Gemma LLM and embedding models so the model file was not reusable by llama-cpp.
  * The available GGUF conversions on Hugging Face did not include the dense models whereas the Ollama GGUF did.

  Even though our AI offering so far has been provided as a "pilot" we always strive to
provide operational continuity and thus decided to provide a custom GGUF conversion
that matches the BF16 precision (which gemma is originally trained in anyway) and includes
the dense modules to match the Ollama model we provided thus far. The model (together with instructions for reproducing it) is publicly available on huggingface as [flyingcircusio/embeddinggemma-300m](https://huggingface.co/flyingcircusio/embeddinggemma-300m-GGUF-with-dense-modules)

3. We are introducing continuous monitoring for numberical stability of the embedding models that we provide that will ensure that changes in the environment or model do not accidentally distort the results our customers are getting while at the same time supporting operational flexibility to improve things over time.
