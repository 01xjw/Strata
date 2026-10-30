# Community benchmark on 2x AMD Radeon PRO W7900D 48 GB (gfx1100)

Measured from 2026-10-22 to 2026-10-30 by 01xjw. Flash-Next IQ3_S served by Strata on two Radeon PRO W7900D 48 GB
cards per engine (layer split, 8 serving slots, MTP), with a second identical engine on two more cards of the same host.

- **Accuracy:** against the unquantized model on GSM8K, IFEval and a MMLU-Pro subset, every paired 95% interval
  contains 0. GSM8K 96.74% vs 96.89%, IFEval prompt-level strict 85.95% vs 85.95%, MMLU-Pro 79.29% vs 79.76%.
- **Throughput:**
  - One engine (2 cards): short-chat decode 77-80 tok/s; 8 clients at once 140-145 tok/s total.
  - Prompt 1,421 / 1,947 / 2,252 tok/s at 4K / 16K / 32K with the gfx1100 hipBLASLt table.
  - Two engines (4 cards): 257 tok/s total from 16 clients up.
- **What it took on RDNA3** (each change leaves the answers byte-identical):
  - the gfx1100 hipBLASLt table: prompt +84-86%;
  - glibc huge pages, then `--pipeline-windows 1` beside `--batch`: 64 clients 87-94 -> 134-139 tok/s total;
  - on 0.1.54, one engine fix: the new Linux file-tier defaults stay off when every expert is in VRAM. Decode
    57.3 -> 80.4 tok/s.
- **Main limitations:**
  - one host;
  - accuracy was measured on the 0.1.43.1 build; for 0.1.54 the evidence is that the answers are identical with and
    without the changes;
  - 0.1.54 reads long prompts 21-30% slower than 0.1.43.1 here, for reasons not yet found;
  - 65,536-token context, so there is no 128K row.

## Hardware and software

- **GPU and VRAM:** 2x AMD Radeon PRO W7900D 48 GB per engine (gfx1100, PCI ID `1002:744b`, subsystem `1002:0e0c`,
  VBIOS `113-D7070910-100`); `rocm-smi` and the bench report them as "AMD Radeon". The host has eight such cards; the
  two engines used four (GPU 0,1 and GPU 2,3), and the other four were not used by this test.
- **CPU and RAM:** AMD EPYC 9334 32-Core Processor (64 cores, 128 threads); installed RAM 503.6 GiB. Storage and PCIe
  link: not recorded.
- **OS:** Linux 6.8.0-79-generic; amdgpu driver 6.16.13; ROCm 7.2.1 (container).
- **Strata, two builds:**
  - **0.1.43.1 (deployed):** source build of `a0116b27` (0.1.43.1, #2038) plus one local engine commit (`e5d86ee0`).
    The commit runs `--pipeline-windows 1` beside `--batch` and copies a new request's checkpoint and K/V into its slot
    on the device. Its engine part is in `engine-0.1.43.1-local.diff` (5 files, +196 / -27).
  - **0.1.54:** source build of `main` `870aedd9` plus the commits of branch
    [`rdna3/w7900d`](https://github.com/01xjw/Strata/tree/rdna3/w7900d) (next section). Built with the upstream
    defaults, which include `STRATA_MMQ_KQUANTS=ON` (0.1.43.1 was built with it off).
- **Background workloads and power limits:** the engine under test served no other traffic during a run; power limits
  not recorded.

## Model and configuration

- **Model:** ISTA-DASLab Qwen3.8-Flash-Next GSQ-RCO, IQ3_S
  (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`, `...-00002-of-00002.gguf`), Strata pack
  `qwen3.8-flash-next-iq3_s`, MTP drafter (`mtp/rt`). Repository revision: not recorded. The expert blobs mix IQ3_XXS,
  IQ3_S, IQ2_S and IQ4_XS gate/up with IQ4_NL or Q2_0 down projections (9 of 48 layers use Q2_0).
- **Vision encoder:** off. **Expert profile:** the shipped `data/expert-profile.bin`.
- **Engine settings:** context 65,536 tokens; `--kv int8`; expert cache `auto` (the server reports 24,576 slots,
  47,962 MiB: every expert of the model is in VRAM); `--prefill auto`; `--mmap-experts`.
- **Environment:** `GLIBC_TUNABLES=glibc.malloc.hugetlb=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1`.
- **MTP:** `--spec 4 --spec-min-p 0.5`.
- **Sampling:** thinking off for the bench suite, on where a benchmark below says so; temperature 0 everywhere.

Server config of one engine (paths shortened; the second engine is identical on GPU 2,3):

```text
python -m serve.server --engine strata --config a.json --host 0.0.0.0 --port 8000

a.json: backend hip, gpu [0, 1], gpu_order as_given, parallel 8, args:
  --pack <data>/packs/qwen3.8-flash-next-iq3_s
  --native <data>/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
  --ple-gguf <data>/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
  --mmap-experts --expert-profile data/expert-profile.bin --expert-cache auto --prefill auto
  --spec 4 --spec-min-p 0.5 --max-context 65536 --kv int8 --pool-workers 24 --vram-reserve-mib 2048
  --mtp <data>/mtp/rt --pipeline-windows 1
```

## RDNA3 adaptation: what changed and what each change did

All numbers below are from this host. "Answers" compares the generated text with the arm before the change.

| Change | Where | Effect on 2x W7900D | Answers |
| --- | --- | --- | --- |
| gfx1100 hipBLASLt table | Setting: `STRATA_HIPBLASLT_TUNING=tools/hip/gfx1100-hipblaslt-100202.txt`. The table ships upstream; setup writes it into its configs, but a hand-written config does not have it | 0.1.43.1: prompt +85 / +86 / +84% at 4K / 16K / 32K, 32K first token 26.2 -> 14.3 s. 0.1.54: +112 / +73 / +106% (B0 -> B1). On 0.1.43.1, decode and 8 clients unchanged | 5/5 probes identical, on both builds |
| glibc huge pages | Setting: `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` | Each slot checkpoint allocates about 58 MB of host memory; with huge pages the long prompt-read stalls (sometimes over 1 s) go away. 64 clients 87-94 -> 121 tok/s total, latency p95 93 -> 71 s | 210/210 identical |
| `--pipeline-windows 1` beside `--batch` | Commit `47f366b0` (`src/program/generate.cpp`, `docs/MULTI_GPU.md`). `main` turns pw1 off under `--batch`; on a two-stage split it now stays on, and `--batch-mtp` turns off with it | With huge pages: 8 / 32 / 64 clients 121.6 / 122.1 / 120.8 -> 140.6 / 142.2 / 133.7 tok/s, latency p95 at 64 clients 71 -> 61 s. A 56-token prompt reads in about 270 ms instead of 420 ms | 315/315 identical |
| File-tier defaults off when every expert is in VRAM | Commit `735a40ec` (`src/program/generate.cpp` +30 / -2, `src/core/expert_source.cpp` +1) | 0.1.54 turns on io prefetch, its decode gate and the routing-aware prefetch for an `experts.bin` pack on Linux, even with all 24,576 experts in VRAM. That costs decode: 39.0 (no table) and 57.3 (table) tok/s against 77-79 on 0.1.43.1. With the fix: 80.4 tok/s, agent turns 57.3 -> 84.0; same as setting the switches off by hand (79.6). `STRATA_IO_PREFETCH=1` and `STRATA_LOOKAHEAD=1` still turn them on | 5/5 probes identical |
| Slot copy on the device | 0.1.43.1: the local commit. 0.1.54: upstream's opt-in `STRATA_SLOT_COPY_DIRECT=1 STRATA_SLOT_COPY_BATCHSYNC=1` | Not measured on its own | - |
| RDNA3 tooling | Commit `0ecc5d2a`: `gfx_arch_is_rdna3_dgpu()` with a test; `arch_defaults_test` asserts that RDNA3 gets no gfx12 defaults and no switch that changes answers; `docs/RDNA3_VALIDATION.md`; a two-instance service helper; a model-free validation script; `tools/ab_engine.py --long-words --save-text-dir` | - | - |

Tried on this host and not kept, all on 0.1.43.1 (2026-10-22):

- **8-row prompt windows:** no faster (432 vs 407-420 ms), and 3 of 315 answers differed.
- **Moving the checkpoint into the slot instead of copying it:** 64 clients about 20% slower without huge pages, and
  only about 11 ms saved with them.
- **The gfx1151 decode switches:** answers identical, but 64 clients gave 103 vs 121 tok/s.
- **The RDNA4 GDN switches and `--batch-groups 1`:** slower.
- **`--short-read 0`:** prompt read 806 ms, and the answers change.

## Method

1. **Bench suite** (2026-10-30): `python tools/strata_bench.py --url http://127.0.0.1:8000 --long --clients 8`.
   - The first run was against one engine that had been serving for two days, so model loading was not timed. Its data
     is `results.json`.
   - One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation
     cache cannot help the fresh-prompt tests.
   - Short chat: the three questions of `tools/ab_engine.py`, 100 new tokens, 3 runs.
   - 4K / 16K / 32K: generated word lists, 24 new tokens, 3 fresh prompts each.
   - Agent session: system prompt, four tools, 5 follow-up turns of about 1,500 words each, 64 new tokens per turn,
     3 sessions.
   - Concurrency: 8 requests sent at once.
   - Speeds are the engine's own timings; first-token times are measured by the client over HTTP.
2. **Engine comparison** (2026-10-30, `runs-2026-10-30/`):
   - Seven arms on the same two cards, each a fresh engine process with the config above.
   - Each arm ran the same suite, then `scripts/probe.py`: five fixed greedy prompts without a per-run tag
     (three short, a 4K and a 16K word list), with the sha256 of each answer.
   - The 8-client test was then rerun on its own (`scripts/clients8.py`, 5 rounds; median of rounds 2-5), alternating
     C1, B3, C1, B3 on fresh starts.
   - In the arms table, "slot copy" is `STRATA_SLOT_COPY_DIRECT=1 STRATA_SLOT_COPY_BATCHSYNC=1`, and "table" is the
     gfx1100 hipBLASLt table.

   | Arm | Build | Settings beyond the config |
   | --- | --- | --- |
   | A0 | 0.1.43.1 + local commit | none: the engine as deployed |
   | A1 | same | table (and `STRATA_HIPBLASLT_VERBOSE=1`) |
   | B0 | 0.1.54 + `47f366b0` + `0ecc5d2a` | `STRATA_HIPBLASLT_TABLE=off`, slot copy |
   | B1 | same | table (and `STRATA_HIPBLASLT_VERBOSE=1`), slot copy |
   | B2 | same | table, slot copy, `STRATA_LOOKAHEAD=0` |
   | B3 | same | table, slot copy, `STRATA_IO_PREFETCH=0 STRATA_COLD_OVERLAP=0` |
   | C1 | same + the fix | table, slot copy |

   C1 ran the fix's first revision. `735a40ec` adds one line to it, which clears the decode gate's flag when io
   prefetch is turned off. On this config the flag is not read again once io prefetch is off: `begin_layer` returns
   early when the source is not staged. That line was built and unit-tested, not rerun on the GPUs.
3. **Serving A/B** (2026-10-22, `serving-2026-10-22/`):
   - Setup: one engine on two other cards of the same host, 0.1.43.1 lineage.
   - Load: `tools/serve_load.py`, streaming chat requests of about 56 prompt tokens with a unique nonce, 128 new
     tokens, temperature 0. At each level C requests were sent together, 3 rounds per level.
   - Total tok/s is generated tokens over the wall time.
   - Answers were compared across arms with the same nonce.
4. **Concurrency sweep, both engines** (2026-10-22): the same load against both deployed engines at the same time,
   each engine with its own client. Only the summary was kept.
5. **Accuracy** (2026-10-27, the 0.1.43.1 build and the settings above):
   - Strata side: both engines behind one load-balancing proxy, 16 requests in flight.
   - Baseline: the unquantized model (no quantization, `fp8_e4m3` KV cache) on SGLang 0.5.18 with 4x AMD Instinct
     MI300X (tp 4, ep 4), 8 requests in flight.
   - Both sides got the same sample file, prompts and request parameters (temperature 0, per-task `max_tokens` and
     thinking), and were graded by the same code (IFEval with the lm-evaluation-harness 0.4 checks).
   - Statistics: paired 95% intervals from a 10,000-sample bootstrap; McNemar exact test. Summary:
     `accuracy-summary.json`.

## Results

### One engine (2 cards), by build

| | 0.1.43.1, deployed (`results.json` / A0) | 0.1.43.1 + table (A1) | 0.1.54 + this branch + table (C1) |
| --- | --- | --- | --- |
| Short-chat decode, tok/s | 76.2 / 77.1 | 79.3 | 80.4 |
| Agent follow-up decode, tok/s | 79.4 / 86.2 | 82.4 | 84.0 |
| Prompt 4K / 16K / 32K, tok/s | 766.5 / 1,035.4 / 1,214.9; 770.1 / 1,044.3 / 1,227.3 | 1,421.1 / 1,947.2 / 2,252.4 | 1,339.8 / 1,630.7 / 1,580.1 |
| 32K first token, s | 26.42 / 26.15 | 14.27 | 20.32 |
| 8 clients, total tok/s | 144.7 / 139.9 | 139.7 | 83.9 in the suite; 114.3 and 126.5 rerun alone |
| Probe answers | reference | identical to A0 | identical to stock 0.1.54 (B0-B3) |

The first column has two runs of the same engine on the same day: the bench suite above and arm A0, a few hours apart.
They agree within 1-4%, except agent decode (+9%).

### Engine comparison, all arms

| Arm | Short-chat decode, median (range) | Agent decode | Prompt 4K / 16K / 32K | 32K first token, s | 8 clients total, in the suite |
| --- | --- | ---: | --- | ---: | ---: |
| A0 | 77.1 (69.2-107.3) | 86.2 | 770.1 / 1,044.3 / 1,227.3 | 26.15 | 139.9 |
| A1 | 79.3 (69.7-104.1) | 82.4 | 1,421.1 / 1,947.2 / 2,252.4 | 14.27 | 139.7 |
| B0 | 39.0 (26.0-61.6) | 63.5 | 610.0 / 838.2 / 848.1 | 37.83 | 141.9 |
| B1 | 57.3 (36.7-105.3) | 57.3 | 1,293.9 / 1,453.3 / 1,746.7 | 18.39 | 137.0 |
| B2 | 70.2 (49.8-104.3) | 72.8 | 1,208.3 / 1,753.0 / 1,621.3 | 19.81 | 82.5 |
| B3 | 79.6 (62.6-104.6) | 75.0 | 1,320.1 / 1,581.1 / 1,775.9 | 18.09 | 139.6 |
| C1 | 80.4 (50.3-108.5) | 84.0 | 1,339.8 / 1,630.7 / 1,580.1 | 20.32 | 83.9 |

- **Errors:** 0 in every arm.
- **Agent cache hit:** 69.3-69.5% in every arm.
- **8 clients, rerun alone** (median of rounds 2-5, in this order): C1 114.3, B3 112.9, C1 126.5, B3 114.4. On 0.1.54
  the 8-client total moves between about 80 and 140 tok/s from one server start to the next, with or without the fix.
- **Probe sha256 prefixes:**
  - A0 and A1: `7df3c44307ef 0773e5957e65 3c867b2e39a4 00a172d7f157 c7e5807018fa`.
  - B0 to C1: `97c7fabc00bd adedf0c976a2 f1a02e63c79f 99fc335ad7f8 c7e5807018fa`.

  Each build gives the same five answers in every arm. The two builds differ on four of the five prompts; that is
  upstream's change between 0.1.43.1 and 0.1.54, not one of these commits.

### Serving A/B, one engine (2 cards), 0.1.43.1 lineage

| Arm | File | 8 / 32 / 64 clients, total tok/s | Latency p95 at 64 clients, s | Answers |
| --- | --- | --- | ---: | --- |
| v3: 8 slots, MTP, slot copy on the device | `v3.json`, `v3-repeat1.json`, `v3-repeat2.json` | 95.3 / 89.3 / 87.2; 64 clients again 91.7, 94.0 | 93.9 | 315/315 |
| v3 + `--pipeline-windows 1`, no huge pages | `v3-pw1.json` | 101.3 / 75.1 / 83.0 | 117.6 | - |
| v3 + huge pages | `v3-hugepages.json` | 121.6 / 122.1 / 120.8 | 70.7 | 210/210 |
| v3 + huge pages + pw1 + checkpoint move | `v3-hugepages-pw1-ckptmove.json` | 136.9 / 140.2 / 139.1 | 58.9 | 210/210 |
| **v3 + huge pages + pw1 (deployed)** | `v3-hugepages-pw1.json` | **140.6 / 142.2 / 133.7** | **61.3** | 315/315 |

- Different arms vary by +/-15-30%, while repeats of one arm agree.
- Without huge pages, pw1 is lost in that noise.
- Each engine has 8 slots, so above 8 clients the extra requests queue and the latency is mostly waiting time.

### Concurrency sweep, both engines (4 cards, deployed build, no table)

| Clients per engine | Requests per engine | Errors | Total tok/s, engine 1 / engine 2 | Per-request tok/s p50 | TTFT p50 (s) | Latency p95 (s) |
| ---: | ---: | ---: | --- | --- | --- | --- |
| 1 | 3 | 0 | 49.4 / 54.1 | 55.6 / 61.3 | 0.32 / 0.28 | 11.86 / 10.10 |
| 8 | 24 | 0 | 126.1 / 130.5 | 19.0 / 19.9 | 1.48 / 1.47 | 8.53 / 8.31 |
| 32 | 96 | 0 | 126.8 / 130.2 | 17.7 / 18.1 | 13.35 / 13.06 | 34.93 / 34.20 |
| 64 | 192 | 0 | 126.5 / 130.7 | 18.3 / 18.4 | 29.41 / 31.23 | 64.72 / 63.72 |

- Both engines together: 103.5 / 256.6 / 257.0 / 257.2 tok/s at 2 / 16 / 64 / 128 clients.
- Before the huge pages and pw1 (same day, same load), the two engines gave 89.9 / 90.9 tok/s at 8 clients and
  89.9 / 90.2 at 64; the 315 answers were byte-identical between the two versions.
- The 1-client rows include a one-off stall on the first request after a restart (latency p95 10-12 s for 128
  tokens).

### Accuracy against the unquantized model

| Benchmark | Metric | n | IQ3_S, 4x W7900D | Unquantized, 4x MI300X | Difference (points) | 95% CI | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| GSM8K (thinking off, `max_tokens` 1536) | accuracy | 1319 | 96.74% | 96.89% | -0.15 | [-0.83, +0.53] | 0.82 |
| IFEval (`max_tokens` 2048) | prompt-level strict | 541 | 85.95% | 85.95% | 0.00 | - | - |
| | instruction-level strict | | 90.41% | 90.53% | -0.12 | | |
| | prompt-level loose | | 90.02% | 88.17% | +1.85 | | |
| | instruction-level loose | | 93.41% | 92.33% | +1.08 | | |
| MMLU-Pro (14 categories x 30, `max_tokens` 2048) | accuracy | 420 | 79.29% | 79.76% | -0.48 | [-3.33, +2.38] | 0.87 |
| GSM8K, thinking on (first 200, `max_tokens` 8192) | accuracy | 200 | 97.50% | 97.50% | 0.00 | [-2.0, +2.0] | 1.00 |

- Every interval contains 0: at these sample sizes, no accuracy difference between IQ3_S on Strata and the unquantized
  model was detected.
- Truncation (`finish_reason` length) was similar on both sides: GSM8K 0.68% / 0.76%, IFEval 0.74% / 1.85%, MMLU-Pro
  9.52% / 9.76%, GSM8K thinking 0% / 0%.
- Errors and retries: 0 on both sides.

### Other reports of the same model in this directory

Each row is the submitter's own medians; machines, engine versions and settings differ, so compare with care. This
report is the only one here with 48 GB cards, an 8-client serving run and an accuracy comparison against the
unquantized model.

| Folder | Hardware | Strata | Prompt tok/s | Decode tok/s |
| --- | --- | --- | --- | --- |
| this one | 2x Radeon PRO W7900D 48 GB (gfx1100), layer split, 8 slots, MTP | 0.1.43.1 + table | 1,421 / 2,252 (4K / 32K) | 79.3 short chat; 140 total at 8 clients |
| [2026-10-06-community-rx-7900-xtx](../2026-10-06-community-rx-7900-xtx/) | RX 7900 XTX 24 GB (gfx1100), Windows, ready-made AMD engine | 0.1.40 | 1,612 (38K) | 77.5 short; 79.7 at 38K |
| [2026-10-04-community-rx7900xtx-hipblaslt-100500](../2026-10-04-community-rx7900xtx-hipblaslt-100500/) | RX 7900 XTX 24 GB (gfx1100), gfx1100 table | 0.1.38 lineage | 1,687 (128K) | 62.3 |
| [2026-10-09-community-4x-rx-7900-xt](../2026-10-09-community-4x-rx-7900-xt/) | 4x RX 7900 XT 20 GB (gfx1100), layer split, five prompt opt-ins | 0.1.41 | 2,052 / 3,804 (4K / 32K) | 60.0 / 52.7 |
| [2026-10-08-community-r9700-linux](../2026-10-08-community-r9700-linux/) | Radeon AI PRO R9700 32 GB (gfx1201) | 0.1.40.3 | 768 / 751 (4K / 32K) | 96.4 / 95.6 |
| [2026-10-08-community-2x-titan-rtx-0.1.40.3](../2026-10-08-community-2x-titan-rtx-0.1.40.3/) | 2x TITAN RTX 24 GB | 0.1.40.3 | 869 / 1,549 (4K / 32K) | 70.9 / 77.3 |
| [2026-10-03-community-rtx4090-iq3s-140k-code](../2026-10-03-community-rtx4090-iq3s-140k-code/) | RTX 4090 | 0.1.38 | 2,059 / 2,948 (4K / 32K) | 77.5 / 97.2 |
| [2026-10-02-community-rtx-5090-iq3s](../2026-10-02-community-rtx-5090-iq3s/) | RTX 5090 | 0.1.33 | 2,473 / 4,168 (4K / 32K) | 141.8 / 142.2 |

## Correctness and limitations

- **Correctness checks:**
  - The accuracy part is the correctness check of the model on this hardware.
  - The probe hashes and the serving A/B answers show that each change above leaves the output byte-identical.
- **Accuracy was measured on the 0.1.43.1 build.** On 0.1.54, the five probe answers are identical with and without
  these commits, but differ from 0.1.43.1 on four prompts (upstream's change); the accuracy run was not repeated on
  0.1.54.
- **Accuracy method limits:**
  - MMLU-Pro is a 420-question subset (interval about +/-2.9 points).
  - Truncated MMLU-Pro answers count as wrong on both sides (about 9.5%), which lowers both absolute scores equally.
  - Only the answer text was graded, not the reasoning.
  - Per-question records are not included; `accuracy-summary.json` has every aggregate and the paired counts.
  - The two accuracy servers ran at different concurrency (16 and 8) on different hardware, so their speeds are not a
    hardware comparison and are not reported here.
- **0.1.54 is slower on long prompts here than 0.1.43.1:** 32K 1,580-1,776 vs 2,252 tok/s with the table. Not
  investigated yet. One known difference is the `STRATA_MMQ_KQUANTS=ON` build default.
- **0.1.54 decode varies more:**
  - the lowest short-chat decode is 26-63 tok/s across arms, against 69 on 0.1.43.1;
  - the 8-client total varies between server starts (see above).
- **Not measured on their own:**
  - pw1 on 0.1.54: every 0.1.54 arm has it on;
  - the device slot copy.
- **Prompt opt-ins not tried:** the 4x RX 7900 XT report's five prompt opt-ins (`STRATA_SPLIT_OWN`,
  `STRATA_STAGE_TRIM`, `STRATA_PF_FUSED`, `STRATA_PF_GEMM`, `STRATA_PF_SWITCH_MIN_T`) were not tried on this host.
- **Coverage gaps:**
  - The engine is tuned for serving (8 slots, `--pipeline-windows 1`), not for one stream.
  - The 4K / 16K / 32K rows cap the answer at 24 tokens, so decode at those depths was not measured.
  - No 128K row: the configured context is 65,536.
  - One host; power and clocks were not recorded.

## Files

| File | What it holds |
| --- | --- |
| `results.json` | `tools/strata_bench.py` per-run data and machine details, first bench-suite run (0.1.43.1) |
| `runs-2026-10-30/A0.json` ... `C1.json` | the same suite for each arm of the engine comparison (storage details removed) |
| `runs-2026-10-30/probes.json` | probe answers' sha256, token counts and timings per arm |
| `runs-2026-10-30/clients8-repeat.json` | the alternating 8-client reruns, every round |
| `serving-2026-10-22/*.json` | serving A/B summaries per arm and client level (`tools/serve_load.py` fields) |
| `accuracy-summary.json` | accuracy, truncation, output-token and paired-comparison aggregates for both servers |
| `engine-0.1.43.1-local.diff` | the engine part of the local commit in the 0.1.43.1 build (`a0116b27..e5d86ee0`, `src/` and `include/`) |
| `scripts/probe.py`, `scripts/clients8.py` | the probe and the 8-client rerun (`python scripts/probe.py URL`, `python scripts/clients8.py URL tools 5`; key from `STRATA_KEY`) |
