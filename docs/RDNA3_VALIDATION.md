# RDNA3 validation (gfx1100 / gfx1101 / gfx1102)

This is the test checklist for promoting an existing gfx11 path to an architecture default.  It does not claim that an
untested switch is faster or quality-neutral.  Follow the repository rule: a new feature stays opt-in, and a default
changes only where a same-day, same-card measurement shows that it is faster.

The prepared Kubernetes pool is one 8-GPU gfx1100 pod on one test node.  It covers one-, two-, four- and
eight-card behavior on gfx1100.  It does **not** validate gfx1101 or gfx1102.

## Connect to the pool

Set `NS` and `POD` to the pool's namespace and pod, and `STRATA_SRC` to the local checkout.

```sh
kubectl get pod -n "$NS" -o wide
kubectl exec -n "$NS" "$POD" -- rocm-smi --showproductname
kubectl exec -n "$NS" -it "$POD" -- bash
```

The pod uses:

- `/data` - the node-local NVMe directory for model shards, packs, MTP data and the PLE table;
- `/workspace` - temporary source/build space.  Copy code and results between it and the development machine.

Copy the current checkout into the pod (run locally):

```sh
kubectl exec -n "$NS" "$POD" -- \
  bash -lc 'rm -rf /workspace/Strata && mkdir -p /workspace/Strata'
tar -C "$STRATA_SRC" \
  --exclude=.git --exclude='build*' --exclude='.venv' -cf - . |
kubectl exec -i -n "$NS" "$POD" -- \
  tar -xf - -C /workspace/Strata
```

Copy results back:

```sh
mkdir -p "$STRATA_SRC/bench/results/rdna3-cluster"
kubectl cp \
  "$NS/$POD:/workspace/Strata/bench/results/rdna3-cluster/." \
  "$STRATA_SRC/bench/results/rdna3-cluster/"
```

## Model-free build and tests

Inside the pod:

```sh
cd /workspace/Strata
export ROCM_PATH=/opt/rocm
export STRATA_HIP_ARCH=gfx1100
bash tools/hip/rdna3_validate.sh all 2>&1 | tee /workspace/rdna3-model-free.log
```

If the cluster cannot fetch the pinned llama.cpp revision, put that checkout on the node and set:

```sh
export STRATA_GGML_DIR=/data/src/llama.cpp
```

Useful narrower commands:

```sh
bash tools/hip/rdna3_validate.sh configure
bash tools/hip/rdna3_validate.sh build
bash tools/hip/rdna3_validate.sh host-tests
bash tools/hip/rdna3_validate.sh hip-tests
```

The minimum gfx1100 set expected to run includes:

```sh
ctest --test-dir build-rdna3 --output-on-failure \
  -R '^(hip_intrinsics|hip_router_fast|hip_gdn_rec_head|hip_prefill_wmma_gemm_parity|hip_prompt_attn_wmma|hip_prefill_native_batch|hip_native_qsa_score)$'
```

`hip_prefill_hcd_exact_parity` and `hip_prefill_hipblaslt_gemm` can skip when no matching tuning table is selected.
A return code of 77 is a registered skip, not a pass.

## Confirm one-, two-, four- and eight-GPU visibility

The pod reserves all eight devices.  Restrict each process instead of creating more pods:

```sh
for visible in 0 0,1 0,1,2,3 0,1,2,3,4,5,6,7; do
  echo "HIP_VISIBLE_DEVICES=${visible}"
  HIP_VISIBLE_DEVICES="${visible}" ./build-rdna3/strata-device --list-devices
done
```

Run model-free tests on one card unless the test explicitly exercises a layer split:

```sh
HIP_VISIBLE_DEVICES=0 bash tools/hip/rdna3_validate.sh hip-tests
```

2026-10-13, the gfx1100 pod: GPU 0/1 were left to an existing service.  The unused devices were checked
with `HIP_VISIBLE_DEVICES=2`, `2,3` and `2,3,4,5`.  In each case `strata-device --list-devices` reported the expected
gfx1100 cards, `strata-device --selftest` passed, and the model-free HIP set
`hip_intrinsics|hip_native_qsa_score|hip_prefill_native_batch|hip_gdn_rec_head|hip_prefill_wmma_gemm_parity|hip_router_fast|hip_prompt_attn_wmma`
passed 7/7.  Raw log: `bench/results/rdna3-cluster/rdna3-1-2-4.log`; summary:
`bench/results/rdna3-cluster/rdna3-1-2-4-summary.md`.  This is a readiness check, not evidence for enabling an
architecture default.

## Prepare a server config

The model stays under `/data`, for example:

```text
/data/models/qwen-flash/
  model-00001-of-00002.gguf
  model-00002-of-00002.gguf
  pack/
  mtp/
```

Start from the config written by `setup.sh`, or create a config whose important fields are:

```json
{
  "exe": "/workspace/Strata/build-rdna3/strata",
  "backend": "hip",
  "gpu": [0],
  "cwd": "/workspace/Strata",
  "model_name": "rdna3-validation",
  "tokenizer": "/data/models/qwen-flash/pack/tokenizer",
  "log": "/workspace/Strata/bench/results/rdna3-cluster/engine.log",
  "args": [
    "--pack", "/data/models/qwen-flash/pack",
    "--native", "/data/models/qwen-flash/model-00001-of-00002.gguf",
    "--ple-gguf", "/data/models/qwen-flash/model-00002-of-00002.gguf",
    "--mmap-experts",
    "--expert-profile", "data/expert-profile.bin",
    "--expert-cache", "auto",
    "--prefill", "auto",
    "--spec", "4",
    "--spec-min-p", "0.5",
    "--mtp", "/data/models/qwen-flash/mtp",
    "--max-context", "65536",
    "--kv", "int8",
    "--adapt-every", "0",
    "--pcie-frac", "0",
    "--vram-reserve-mib", "1024"
  ]
}
```

Replace the paths and CPU worker count for the actual pack.  `--mmap-experts` requires the pack's `experts.bin`.
Keep `--adapt-every 0 --pcie-frac 0` for byte-comparison arms; enable production behavior in a separate speed run.

Start the API:

```sh
cd /workspace/Strata
.venv/bin/python -m serve.server --engine strata \
  --config /data/config/strata-rdna3.json --port 8080
```

From the development machine:

```sh
kubectl port-forward -n "$NS" "pod/$POD" 8080:8080
curl -fsS http://127.0.0.1:8080/v1/models
```

## TODO: exact RDNA3 architecture defaults

Test every switch alone, not as one bundle.  Each arm needs fixed prompts, greedy sampling, 4K/32K/64K inputs and at
least two model formats.  Save token ids as well as timing.

2026-10-13, the gfx1100 pod: a restricted first-pass A/B was completed for `STRATA_GDN_HEAD`,
`STRATA_GDN_PP`, `STRATA_GDN_CONVL2`, `STRATA_GDN_NOY`, `STRATA_CVEC_FUSE`, `STRATA_QFUSE_HC`,
`STRATA_FUSE_WAIT` and `STRATA_FUSE_HITS`.  Each switch was tested alone with A as disabled/baseline and B as the
enabled candidate.  All eight runs had 60 rows and zero full-text SHA mismatches, but token-id capture failed with
the available tokenizer library, only the IQ2_XS model format was covered, and speed results were mixed.  No RDNA3
architecture default is promoted from that run.  See `bench/results/rdna3-cluster/rdna3-switch-ab-summary.md`.

Use `--save-text-dir` with `tools/ab_engine.py` so the full response text and its SHA256 are kept in addition to the
per-request timing JSONL.  Full text identity is still not a replacement for token-id capture, but it prevents a run
from being reduced to the 80-character preview.

Prompt candidates:

```sh
python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  A:STRATA_GDN_HEAD=0 B:STRATA_GDN_HEAD=1 --rounds 5 --log /workspace/gdn-head.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  A:STRATA_GDN_PP=0 B:STRATA_GDN_PP=2 --rounds 5 --log /workspace/gdn-pp.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  A:STRATA_GDN_CONVL2=0 B:STRATA_GDN_CONVL2=1 --rounds 5 --log /workspace/gdn-convl2.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  A:STRATA_GDN_NOY=0 B:STRATA_GDN_NOY=1 --rounds 5 --log /workspace/gdn-noy.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  A:STRATA_CVEC_FUSE=0 B:STRATA_CVEC_FUSE=1 --rounds 5 --log /workspace/cvec-fuse.jsonl
```

Verify/decode fusion candidates:

```sh
python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  A:STRATA_QFUSE_HC=0 B:STRATA_QFUSE_HC=1 --rounds 5 --log /workspace/qfuse-hc.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  A:STRATA_FUSE_WAIT=0 B:STRATA_FUSE_WAIT=1 --rounds 5 --log /workspace/fuse-wait.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  A:STRATA_FUSE_HITS=0 B:STRATA_FUSE_HITS=1 --rounds 5 --log /workspace/fuse-hits.jsonl
```

Do not add `STRATA_SH_STREAM` or `STRATA_HCD_EXACT` to an RDNA3 default table without a separate result:

- shared-stream results already disagree across discrete HIP platforms;
- HCD exact depends on a matching hipBLASLt solution, not only an architecture name.

## TODO: prefill chunk

Use the same model, prompt order and restarted server for each arm:

```sh
python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  C8:STRATA_PREFILL_AUTO_MAX=8192 \
  C16:STRATA_PREFILL_AUTO_MAX=16384 \
  C32:STRATA_PREFILL_AUTO_MAX=32768 \
  --rounds 5 --long-words 32000 --log /workspace/prefill-chunks.jsonl
```

Also use the dedicated fresh/follow-up probe:

```sh
python3 tools/hip/bench_prefill.py \
  --model rdna3-validation --url http://127.0.0.1:8080 \
  --engine-log /workspace/Strata/bench/results/rdna3-cluster/engine.log \
  --label chunk-8192 --output /workspace/chunk-8192.json
```

Record the engine's lend-coverage line, `wait copy`, RAM availability and disk reads.  A faster warmed result with
incomplete RAM coverage is not enough to change bare `--prefill auto`.

## TODO: paths that change floating-point results or routing

Keep these opt-in until a quality report exists:

```sh
python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  BASE:STRATA_HIP_WMMA=0 WMMA:STRATA_HIP_WMMA=1 \
  --rounds 5 --log /workspace/prompt-attn-wmma.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  BASE:STRATA_SELECT_WMMA=0 SELECT:STRATA_SELECT_WMMA=1 \
  --rounds 5 --log /workspace/select-wmma.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  MMQ:STRATA_PF_FUSED=0 FUSED:STRATA_PF_FUSED=1 \
  --rounds 5 --log /workspace/fused-iq.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  BLAS:STRATA_WMMA_GEMM=0 WMMA:STRATA_WMMA_GEMM=1 \
  --rounds 5 --log /workspace/dense-wmma.jsonl

python3 tools/ab_engine.py /data/config/strata-rdna3.json \
  FULL:STRATA_ROUTE_TAIL_SKIP=0 SKIP7:STRATA_ROUTE_TAIL_SKIP=7 \
  --rounds 5 --log /workspace/tail-skip.jsonl
```

For each arm record first-token logits/KL, top-1 agreement, full greedy token ids, long-context retrieval and repeated
starts.  Text looking reasonable is not a quality gate.

## TODO: hipBLASLt

Find the installed version and select the matching gfx1100 table:

```sh
grep -E 'HIPBLASLT_VERSION_(MAJOR|MINOR|PATCH)' \
  "${ROCM_PATH}/include/hipblaslt/hipblaslt-version.h"
export STRATA_HIPBLASLT_TUNING="$PWD/tools/hip/gfx1100-hipblaslt-100202.txt"  # example only
export STRATA_HIPBLASLT_VERBOSE=1
ctest --test-dir build-rdna3 --output-on-failure \
  -R '^(hip_prefill_hipblaslt_gemm|hip_prefill_hcd_exact_parity)$'
```

The version in the filename/header must match the installed library.  gfx1100/1101/1102 are accepted as one RDNA3
family by the loader, but each target card must still run the per-solution support gate and an end-to-end comparison.

To recalibrate actual model shapes:

```sh
cmake --build build-rdna3 --target tune_hipblaslt -j4
./build-rdna3/tune_hipblaslt --help
```

Store a new table as `tools/hip/<arch>-hipblaslt-<version>.txt` only after checking `fallbacks=0` for covered shapes
and comparing prompt speed with plain hipBLAS.

## TODO: layer split on 2 / 4 / 8 gfx1100 cards

Create config copies with:

```json
{"backend": "hip", "gpu": [0, 1], "layer_split": "auto"}
```

then `[0,1,2,3]` and `[0,1,2,3,4,5,6,7]`.  Run the standard benchmark:

```sh
.venv/bin/python tools/strata_bench.py \
  --config /data/config/strata-rdna3-2gpu.json --runs 3 --long \
  --out bench/results/rdna3-cluster
```

Record each stage's VRAM, expert-cache coverage and probed link.  Compare against the best single-card run; a working
layer split is not automatically faster.  Explicit `--pipeline-windows 2` is a separate A/B and is not a default on
gfx1100.

## Bare-metal two-instance IQ3_S concurrency run

For an 8-card gfx1100 host without Kubernetes or Docker, use the helper below to run two independent two-card IQ3_S
services.  It assumes the two IQ3_S GGUF shards are already present under
`/root/strata-data/models/Qwen3.8-flash-Next/IQ3_S`, builds the HIP engine when needed, creates the native pack with
`experts.bin`, writes one config for GPUs `0,1` and one for `2,3`, then starts ports `8090` and `8091`.

```sh
cd /root/Strata
chmod +x tools/hip/iq3s_two_2gpu_instances.sh
PIPELINE_WINDOWS=1 CONTEXT=65536 tools/hip/iq3s_two_2gpu_instances.sh all
```

Useful follow-up commands:

```sh
tools/hip/iq3s_two_2gpu_instances.sh status
LOAD_CLIENTS=1,2,4,8 LOAD_ROUNDS=3 tools/hip/iq3s_two_2gpu_instances.sh load
LONG_CLIENTS=2,4,8 LONG_PROMPT_TOKENS=32768,64000 tools/hip/iq3s_two_2gpu_instances.sh long-load
tools/hip/iq3s_two_2gpu_instances.sh stop
```

Results are written under `/root/strata-data/runs/iq3s-two-2gpu/results`.  The helper's defaults are the measured
service settings: 8 slots (`PARALLEL=8`, two pipelined groups of 4), the MTP draft head when `$DATA_ROOT/mtp/rt`
holds it, and `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` for the servers (`HUGEPAGE_MALLOC=0` drops it).
`PIPELINE_WINDOWS=1` overlaps the two cards on each admitted request's prompt read.  One instance on GPUs 6,7, short
prompts (~56 tokens), 128-token answers, totals in tok/s at 8 / 32 / 64 clients, answers identical byte for byte:

| settings | c8 | c32 | c64 | c64 p95 |
| --- | ---: | ---: | ---: | ---: |
| 8 slots + MTP | 95 | 89 | 87-94 | 93 s |
| + hugetlb | 122 | 122 | 121 | 71 s |
| + hugetlb + `PIPELINE_WINDOWS=1` | 141 | 142 | 134 | 61 s |

Each admission saves a conversation checkpoint (~58 MB of GDN state per stage) into fresh host memory and copies it
into the slot; on 4 KiB pages those faults cost ~25 ms per block and the runs showed prompt reads stalling for a
second or more.  The hugetlb tunable needs THP in `madvise` or `always` mode.  The helper also sets
`STRATA_SLOT_COPY_DIRECT=1 STRATA_SLOT_COPY_BATCHSYNC=1` (`SLOT_COPY_DIRECT=0` drops them): admissions then copy the
slot buffer to buffer on each stage's GPU instead of through host memory, which removed ~140 ms per admission on
2x gfx1100 (engine 0.1.43.1).  This is a service concurrency test, not evidence for changing RDNA3 architecture
defaults.

## Promotion checklist

An exact switch can enter `arch_defaults.cpp` only when:

- its kernel parity test passes;
- 4K/32K/64K token ids and required intermediate buffers are byte-identical;
- it is faster than the same-day baseline outside measurement noise;
- it passes at least two model formats and repeated starts;
- it does not add an OOM, timeout, NaN or memory peak regression.

A bit-changing switch remains opt-in unless its quality threshold was defined before the run and the saved data meets
that threshold.  Add gfx1101/gfx1102 defaults only after tests on those physical architectures; the 8×gfx1100 pool
cannot supply that evidence.
