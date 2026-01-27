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

| Model | Quantization | URL | HASH/ID |
| - | - | - | - |
| Baseline Ollama | BF16 |  https://ollama.com/library/embeddinggemma:300m | 85462619ee72 (ID) |
| llama-cpp unsloth/embeddinggemma-300m-GGUF | F32 | https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/blob/6661a6504c30d8304af13455cb4a5d4f5bc6011f/embeddinggemma-300M-F32.gguf | a3125072128fc76d1c1d8d19f7b095c7e3bfbf00594dcf8a8bd3bcb334935d57 (SHA256) |
| llama-cpp unsloth/embeddinggemma-300m-GGUF | Q4_0 | https://huggingface.co/unsloth/embeddinggemma-300m-GGUF/blob/6661a6504c30d8304af13455cb4a5d4f5bc6011f/embeddinggemma-300m-Q4_0.gguf | edc6015cb15694c27be7d1d33f1bc015db9a358ff51ed524628c027504907ba9 (SHA256)
| llama-cpp cduk/embeddinggemma-300m-GGUF-with-dense-modules | F32 | https://huggingface.co/cduk/embeddinggemma-300m-GGUF-with-dense-modules/blob/6c4b6b0b86f1917506d8047478e350e51ac65fe8/embeddinggemma-300M-F32.gguf | 695c5960fa074ccdc7993d9a9d215a20bee20c03f8eba23111caa6de0d0c6991 (SHA256) |
| llama-cpp flyingcircusio/embeddinggemma-300m-GGUF-with-dense-modules | BF16 | https://huggingface.co/flyingcircusio/embeddinggemma-300m-GGUF-with-dense-modules/blob/4e7ac746a96002b1837ae0099f9253d89e14e603/embeddinggemma-300M-BF16-with-dense.gguf | 4610577e176f925b20e1981212e2d03741e0b48b2051e78e73da5865f0552671 |

# Request

As input to the model we used the example sentence from the original's model card:

```
$ curl http://127.0.0.1:8000/models/embeddinggemma:300m/proxy/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "The food was delicious and the waiter...",
    "model": "embeddinggemma",
    "encoding_format": "float"
  }'
```

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

|                           |                 01-M4-cpu-F32 |                02-M4-cpu-Q4_0 |               03-M4-metal-F32 |              04-M4-metal-Q4_0 |         05-M2-metal_dense-F32 |             06-W7900-rocm-F32 |            07-W7900-rocm-Q4_0 |         08-W7900-rocm_mmq-F32 |        09-W7900-rocm_mmq-Q4_0 |      10-W7900-rocm_dense-BF16 |
| ------------------------- | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: |
| 00-W7900-ollama_rocm-BF16 |                        1.4196 |                        1.4167 |                        1.4196 |                        1.4169 |                        0.0077 |                        1.4196 |                        1.4173 |                        1.4196 |                        1.4173 |                    **0.0067** |
| 01-M4-cpu-F32             |                               |                        0.3958 |                    **0.0007** |                        0.3951 |                    **1.4196** |                    **0.0006** |                        0.3986 |                    **0.0006** |                        0.3986 |                        1.4198 |
| 02-M4-cpu-Q4_0            |                               |                               |                        0.3957 |                    **0.0118** |                        1.4165 |                        0.3957 |                    **0.0390** |                        0.3957 |                    **0.0390** |                        1.4168 |
| 03-M4-metal-F32           |                               |                               |                               |                        0.3950 |                    **1.4196** |                    **0.0005** |                        0.3986 |                    **0.0005** |                        0.3986 |                        1.4198 |
| 04-M4-metal-Q4_0          |                               |                               |                               |                               |                        1.4167 |                        0.3950 |                    **0.0356** |                        0.3950 |                    **0.0356** |                        1.4169 |
| 05-M2-metal_dense-F32     |                               |                               |                               |                               |                               |                    **1.4196** |                        1.4171 |                    **1.4196** |                        1.4171 |                        0.0066 |
| 06-W7900-rocm-F32         |                               |                               |                               |                               |                               |                               |                        0.3986 |                    **0.0000** |                        0.3986 |                        1.4198 |
| 07-W7900-rocm-Q4_0        |                               |                               |                               |                               |                               |                               |                               |                        0.3986 |                    **0.0000** |                        1.4173 |
| 08-W7900-rocm_mmq-F32     |                               |                               |                               |                               |                               |                               |                               |                               |                        0.3986 |                        1.4198 |
| 09-W7900-rocm_mmq-Q4_0    |                               |                               |                               |                               |                               |                               |                               |                               |                               |                        1.4173 |

## Cosine Similarity

|                           |                 01-M4-cpu-F32 |                02-M4-cpu-Q4_0 |               03-M4-metal-F32 |              04-M4-metal-Q4_0 |         05-M2-metal_dense-F32 |             06-W7900-rocm-F32 |            07-W7900-rocm-Q4_0 |         08-W7900-rocm_mmq-F32 |        09-W7900-rocm_mmq-Q4_0 |      10-W7900-rocm_dense-BF16 |
| ------------------------- | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: |
| 00-W7900-ollama_rocm-BF16 |                       -0.0077 |                       -0.0035 |                       -0.0077 |                       -0.0038 |                        1.0000 |                       -0.0077 |                       -0.0043 |                       -0.0077 |                       -0.0043 |                    **1.0000** |
| 01-M4-cpu-F32             |                               |                        0.9217 |                    **1.0000** |                        0.9220 |                   **-0.0076** |                    **1.0000** |                        0.9205 |                    **1.0000** |                        0.9205 |                       -0.0079 |
| 02-M4-cpu-Q4_0            |                               |                               |                        0.9217 |                    **0.9999** |                       -0.0032 |                        0.9217 |                    **0.9992** |                        0.9217 |                    **0.9992** |                       -0.0036 |
| 03-M4-metal-F32           |                               |                               |                               |                        0.9220 |                   **-0.0076** |                    **1.0000** |                        0.9206 |                    **1.0000** |                        0.9206 |                       -0.0079 |
| 04-M4-metal-Q4_0          |                               |                               |                               |                               |                       -0.0035 |                        0.9220 |                    **0.9994** |                        0.9220 |                    **0.9994** |                       -0.0039 |
| 05-M2-metal_dense-F32     |                               |                               |                               |                               |                               |                   **-0.0076** |                       -0.0040 |                   **-0.0076** |                       -0.0040 |                        1.0000 |
| 06-W7900-rocm-F32         |                               |                               |                               |                               |                               |                               |                        0.9206 |                    **1.0000** |                        0.9206 |                       -0.0079 |
| 07-W7900-rocm-Q4_0        |                               |                               |                               |                               |                               |                               |                               |                        0.9206 |                    **1.0000** |                       -0.0044 |
| 08-W7900-rocm_mmq-F32     |                               |                               |                               |                               |                               |                               |                               |                               |                        0.9206 |                       -0.0079 |
| 09-W7900-rocm_mmq-Q4_0    |                               |                               |                               |                               |                               |                               |                               |                               |                               |                       -0.0044 |

## Angle

|                           |                 01-M4-cpu-F32 |                02-M4-cpu-Q4_0 |               03-M4-metal-F32 |              04-M4-metal-Q4_0 |         05-M2-metal_dense-F32 |             06-W7900-rocm-F32 |            07-W7900-rocm-Q4_0 |         08-W7900-rocm_mmq-F32 |        09-W7900-rocm_mmq-Q4_0 |      10-W7900-rocm_dense-BF16 |
| ------------------------- | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: | ----------------------------: |
| 00-W7900-ollama_rocm-BF16 |                         90.4° |                         90.2° |                         90.4° |                         90.2° |                          0.4° |                         90.4° |                         90.2° |                         90.4° |                         90.2° |                      **0.4°** |
| 01-M4-cpu-F32             |                               |                         22.8° |                      **0.0°** |                         22.8° |                     **90.4°** |                      **0.0°** |                         23.0° |                      **0.0°** |                         23.0° |                         90.5° |
| 02-M4-cpu-Q4_0            |                               |                               |                         22.8° |                      **0.7°** |                         90.2° |                         22.8° |                      **2.2°** |                         22.8° |                      **2.2°** |                         90.2° |
| 03-M4-metal-F32           |                               |                               |                               |                         22.8° |                     **90.4°** |                      **0.0°** |                         23.0° |                      **0.0°** |                         23.0° |                         90.5° |
| 04-M4-metal-Q4_0          |                               |                               |                               |                               |                         90.2° |                         22.8° |                      **2.0°** |                         22.8° |                      **2.0°** |                         90.2° |
| 05-M2-metal_dense-F32     |                               |                               |                               |                               |                               |                     **90.4°** |                         90.2° |                     **90.4°** |                         90.2° |                          0.4° |
| 06-W7900-rocm-F32         |                               |                               |                               |                               |                               |                               |                         23.0° |                      **0.0°** |                         23.0° |                         90.5° |
| 07-W7900-rocm-Q4_0        |                               |                               |                               |                               |                               |                               |                               |                         23.0° |                      **0.0°** |                         90.3° |
| 08-W7900-rocm_mmq-F32     |                               |                               |                               |                               |                               |                               |                               |                               |                         23.0° |                         90.5° |
| 09-W7900-rocm_mmq-Q4_0    |                               |                               |                               |                               |                               |                               |                               |                               |                               |                         90.3° |

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
