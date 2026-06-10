# A100 Raw Results Table

All values are directly from the JSON output files without any post-processing.

## Metric Definitions

| System | Field | Definition |
|--------|-------|------------|
| TGX | `latency_ms_per_token` | `run_time_ms / (step.max() + 1)` where step.max()+1 = prompt_len + gen_len = 1088. Does NOT divide by batch_size. |
| PyTorch | `latency_ms_per_token` | `run_time_ms / (cur_pos - prompt_len)` = `run_time_ms / gen_len`. Does NOT divide by batch_size. |
| vLLM | `avg_iter_seconds` | Wall-clock time (s) for the entire batch (all bs requests) to complete prefill + decode. |
| vLLM | `latency_ms_per_token` | `avg_iter_seconds * 1000 / ((input_len + output_len) * batch_size)` = `avg_s * 1000 / (1088 * bs)`. Divides by batch_size. |
| SGLang | `total_iter_seconds` | Wall-clock time (s) for the entire batch to complete prefill + decode. |
| SGLang | `latency_ms_per_token` | `total_iter_seconds * 1000 / ((input_len + output_len) * batch_size)` = `total_s * 1000 / (1088 * bs)`. Divides by batch_size. |

## Raw Data

### Qwen3-0.6B

| BS | TGX ms/tok | PyTorch ms/tok | vLLM avg_iter_s | vLLM ms/tok | SGLang total_iter_s | SGLang ms/tok |
|-----|-----------|---------------|----------------|------------|--------------------|--------------| 
| 1 | 2.561 | 23.889 | 2.3232 | 2.1353 | 2.669 | 2.4531 |
| 2 | 2.590 | 23.042 | 2.7907 | 1.2825 | 3.109 | 1.4288 |
| 4 | 2.647 | 23.368 | 2.8823 | 0.6623 | 3.155 | 0.7250 |
| 8 | 2.939 | 23.292 | 3.0719 | 0.3529 | 3.245 | 0.3728 |
| 16 | 7.427 | 23.106 | 3.5145 | 0.2019 | 3.516 | 0.2020 |

### Qwen3-1.7B

| BS | TGX ms/tok | PyTorch ms/tok | vLLM avg_iter_s | vLLM ms/tok | SGLang total_iter_s | SGLang ms/tok |
|-----|-----------|---------------|----------------|------------|--------------------|--------------| 
| 1 | 3.818 | 23.270 | 3.9246 | 3.6072 | 4.291 | 3.9439 |
| 2 | 3.856 | 23.253 | 3.9789 | 1.8285 | 4.301 | 1.9766 |
| 4 | 3.954 | 22.478 | 4.0637 | 0.9338 | 4.455 | 1.0237 |
| 8 | 4.362 | 23.690 | 4.2842 | 0.4922 | 4.524 | 0.5198 |
| 16 | 10.211 | 22.887 | 4.7867 | 0.2750 | 4.881 | 0.2804 |

### Qwen3-8B

| BS | TGX ms/tok | PyTorch ms/tok | vLLM avg_iter_s | vLLM ms/tok | SGLang total_iter_s | SGLang ms/tok |
|-----|-----------|---------------|----------------|------------|--------------------|--------------| 
| 1 | 11.592 | 30.590 | 11.2894 | 10.3763 | 11.473 | 10.5450 |
| 2 | 11.716 | 32.295 | 11.3443 | 5.2134 | 11.552 | 5.3088 |
| 4 | 11.986 | 31.347 | 11.4815 | 2.6382 | 11.749 | 2.6997 |
| 8 | 13.161 | 30.957 | 11.7921 | 1.3548 | 12.102 | 1.3904 |
| 16 | 29.464 | 31.141 | 12.5040 | 0.7183 | 12.739 | 0.7318 |
