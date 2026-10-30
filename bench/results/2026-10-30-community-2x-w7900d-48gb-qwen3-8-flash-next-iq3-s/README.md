# Community benchmark on 2x AMD Radeon PRO W7900D 48 GB (gfx1100)

Measured on 2026-10-30 by 01xjw. Flash-Next IQ3_S on two Radeon PRO W7900D 48 GB cards per engine (layer split,
8 serving slots, MTP), with a second identical engine on two more cards of the same host.

This report has three parts:
- the `tools/strata_bench.py` suite on one engine, without and with the gfx1100 hipBLASLt table;
- a concurrency sweep over both engines;
- an accuracy comparison against the unquantized model.

Main limitations:
- one host;
- the engine is 0.1.43.1 with one local commit, older than `main`;
- 65,536-token context, so there is no 128K row.

The full data (the per-arm JSON and the serving A/B) and a 0.1.54 comparison on the same host are in
[01xjw/Strata](https://github.com/01xjw/Strata/tree/main/bench/results/2026-10-30-community-2x-w7900d-48gb-qwen3-8-flash-next-iq3-s).

## Hardware and software

- **GPU and VRAM:** 2x AMD Radeon PRO W7900D 48 GB per engine (gfx1100, PCI ID `1002:744b`). The host has eight such
  cards; the two engines used four, and the other four were not used by this test.
- **CPU and RAM:** AMD EPYC 9334 32-Core Processor (64 cores, 128 threads); installed RAM 503.6 GiB. Storage and PCIe
  link: not recorded.
- **OS:** Linux 6.8.0-79-generic; amdgpu driver 6.16.13; ROCm 7.2.1 (container).
- **Strata:** source build of `a0116b27` (0.1.43.1, #2038) plus one local engine commit, not upstream. It runs
  `--pipeline-windows 1` beside `--batch` and copies a new request's checkpoint and K/V into its slot on the device.
  The diff is in the branch linked above (`engine-0.1.43.1-local.diff`).
- **Background workloads and power limits:** the engine served no other traffic during the suite; power limits not
  recorded.

## Model and configuration

- **Model:** ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO, IQ3_S (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`),
  Strata pack `qwen3.8-flash-next-iq3_s`, MTP drafter (`mtp/rt`).
- **Vision encoder:** off. **Expert profile:** the shipped `data/expert-profile.bin`.
- **Engine settings:** context 65,536 tokens; `--kv int8`; expert cache `auto` (24,576 slots: every expert in VRAM);
  `--prefill auto`; `--mmap-experts`.
- **Environment:** `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`.
- **MTP and sampling:** `--spec 4 --spec-min-p 0.5`; thinking off for the suite; temperature 0 everywhere.

Server config of one engine (paths shortened; the second engine is identical on GPU 2,3):

```text
a.json: backend hip, gpu [0, 1], gpu_order as_given, parallel 8, args:
  --pack <data>/packs/qwen3.8-flash-next-iq3_s --native <IQ3_S shard 1> --ple-gguf <IQ3_S shard 2>
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --max-context 65536 --kv int8 --pool-workers 24 --vram-reserve-mib 2048
  --mtp <data>/mtp/rt --pipeline-windows 1
```

## Method

1. **Bench suite:** `python tools/strata_bench.py --url http://127.0.0.1:8000 --long --clients 8`, run twice.
   - Run 1 (`results.json`) was against the engine as it had been serving, without the table.
   - Run 2 (`results-hipblaslt-table.json`) was a fresh engine process with the same config, plus
     `STRATA_HIPBLASLT_TUNING=tools/hip/gfx1100-hipblaslt-100202.txt` and `STRATA_HIPBLASLT_VERBOSE=1`.
   - Five fixed greedy prompts (three short, a 4K and a 16K word list) gave byte-identical answers in both runs.
2. **Concurrency sweep** (2026-10-22, same build and settings, no table):
   - Tool: `tools/serve_load.py` against both engines at the same time, each engine with its own client.
   - Requests: about 56 prompt tokens with a unique nonce, 128 new tokens, temperature 0; 3 rounds per level.
3. **Accuracy** (2026-10-27, same build and settings, no table):
   - Strata side: both engines behind one load-balancing proxy, 16 requests in flight.
   - Baseline: the unquantized model (`fp8_e4m3` KV cache) on SGLang 0.5.18 with 4x AMD Instinct MI300X (tp 4, ep 4),
     8 requests in flight.
   - Both sides used the same samples, prompts, request parameters and grading code (IFEval with the
     lm-evaluation-harness 0.4 checks).
   - Statistics: paired 95% intervals from a 10,000-sample bootstrap; McNemar exact test.

## Results

### Bench suite, one engine (2 cards)

| Configuration | Table | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | off | 46 (43-51) | 0 | 100 | 9 | not measured | 76.2 (73.8-107.4) | 0.21 (0.20-0.23) |
| short chat | on | 46 (43-51) | 0 | 100 | 9 | not measured | 79.3 (69.7-104.1) | 0.22 (0.20-0.23) |
| 4K prompt, fresh | off | 4019 | 0 | 24 cap | 3 | 766.5 (766.2-766.7) | not measured | 5.27 (5.27-5.27) |
| 4K prompt, fresh | on | 4019 | 0 | 24 cap | 3 | 1421.1 (1414.5-1425.5) | not measured | 2.85 (2.84-2.87) |
| 16K prompt, fresh | off | 16058 | 0 | 24 cap | 3 | 1035.4 (1029.9-1036.1) | not measured | 15.54 (15.53-15.62) |
| 16K prompt, fresh | on | 16058 | 0 | 24 cap | 3 | 1947.2 (1940.9-1952.3) | not measured | 8.28 (8.26-8.30) |
| 32K prompt, fresh | off | 32049 | 0 | 24 cap | 3 | 1214.9 (1211.7-1218.6) | not measured | 26.42 (26.34-26.49) |
| 32K prompt, fresh | on | 32049 | 0 | 24 cap | 3 | 2252.4 (2237.5-2253.1) | not measured | 14.27 (14.27-14.36) |
| agent session, follow-up turns | off | 14441 (14412-14496) at the last turn | cache hit 68.7% | up to 64 | 3 | not measured | 79.4 (53.2-103.9) | 4.33 (2.55-5.01) |
| agent session, follow-up turns | on | 12942 (12925-12959) at the last turn | cache hit 69.3% | up to 64 | 3 | not measured | 82.4 (73.3-92.5) | 2.39 (1.67-2.85) |

8 clients at once on this engine:

- Table off: total 144.7 tok/s (143.7-145.0), first token p50 1.38 s / p95 2.48 s, latency p95 7.11 s, 0 errors.
- Table on: total 139.7 tok/s (138.4-140.9), first token p50 1.54 s / p95 2.71 s, latency p95 7.38 s, 0 errors.

The 4K / 16K / 32K rows cap the answer at 24 tokens, so decode at those depths was not measured. Speeds are the
engine's own timings per request (in the JSON files); first-token times are measured by the client.

### Concurrency sweep, both engines (4 cards)

| Clients per engine | Errors | Total tok/s, engine 1 / engine 2 | TTFT p50 (s) | Latency p95 (s) |
| ---: | ---: | --- | --- | --- |
| 1 | 0 | 49.4 / 54.1 | 0.32 / 0.28 | 11.86 / 10.10 |
| 8 | 0 | 126.1 / 130.5 | 1.48 / 1.47 | 8.53 / 8.31 |
| 32 | 0 | 126.8 / 130.2 | 13.35 / 13.06 | 34.93 / 34.20 |
| 64 | 0 | 126.5 / 130.7 | 29.41 / 31.23 | 64.72 / 63.72 |

- Both engines together: 103.5 / 256.6 / 257.0 / 257.2 tok/s at 2 / 16 / 64 / 128 clients.
- Each engine has 8 slots, so above 8 clients per engine the extra requests queue.
- The 1-client rows include a one-off stall on the first request after a restart (latency p95 10-12 s for 128
  tokens).
- Only the summary of this sweep was kept.

### Accuracy against the unquantized model

| Benchmark | n | IQ3_S, 4x W7900D | Unquantized, 4x MI300X | Difference (points) | 95% CI | McNemar p |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| GSM8K (thinking off, `max_tokens` 1536) | 1319 | 96.74% | 96.89% | -0.15 | [-0.83, +0.53] | 0.82 |
| IFEval prompt-level strict (`max_tokens` 2048) | 541 | 85.95% | 85.95% | 0.00 | - | - |
| MMLU-Pro (14 categories x 30, `max_tokens` 2048) | 420 | 79.29% | 79.76% | -0.48 | [-3.33, +2.38] | 0.87 |
| GSM8K, thinking on (first 200, `max_tokens` 8192) | 200 | 97.50% | 97.50% | 0.00 | [-2.0, +2.0] | 1.00 |

Every interval contains 0: at these sample sizes, no accuracy difference was detected. The other IFEval levels and
the truncation rates are in `accuracy-summary.json`.

## Correctness and limitations

- **Correctness checks:**
  - The accuracy comparison checks the model on this hardware.
  - The five probe answers are byte-identical with and without the table.
- **Accuracy method limits:**
  - MMLU-Pro is a 420-question subset (interval about +/-2.9 points).
  - Truncated answers (about 9.5% on MMLU-Pro, on both sides) count as wrong.
  - The two accuracy servers ran at different concurrency on different hardware, so their speeds are not reported.
- **Bench suite limits:**
  - The agent sessions are generated per run, so their prompt lengths differ between the two runs.
  - Engine logs are not included; each request's engine timings are in the JSON files.
- **Coverage:** one host; one engine build with a local commit; power and clocks not recorded.

## Files

| File | What it holds |
| --- | --- |
| `results.json` | `tools/strata_bench.py` per-run data and machine details, without the table |
| `results-hipblaslt-table.json` | the same with the gfx1100 table |
| `accuracy-summary.json` | accuracy, truncation, output-token and paired-comparison aggregates for both servers |
