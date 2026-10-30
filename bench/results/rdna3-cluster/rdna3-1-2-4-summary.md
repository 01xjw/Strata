# RDNA3 gfx1100 1/2/4-card model-free validation

Date: 2026-10-13

Raw log: `bench/results/rdna3-cluster/rdna3-1-2-4.log`

## Scope

This run used the unused gfx1100 devices in the validation pod and avoided the two occupied devices:

- 1 card: `HIP_VISIBLE_DEVICES=2`
- 2 cards: `HIP_VISIBLE_DEVICES=2,3`
- 4 cards: `HIP_VISIBLE_DEVICES=2,3,4,5`

It validates device visibility, `strata-device --selftest`, and the model-free HIP parity/smoke set. It is not an
end-to-end model A/B and is not sufficient by itself to promote architecture defaults.

## Results

Each scenario listed the expected gfx1100 devices, passed `strata-device --selftest`, and passed the same 7-test HIP set:

- `hip_intrinsics`
- `hip_native_qsa_score`
- `hip_prefill_native_batch`
- `hip_gdn_rec_head`
- `hip_prefill_wmma_gemm_parity`
- `hip_router_fast`
- `hip_prompt_attn_wmma`

Summary:

- 1 card: 7/7 passed, 0 failed
- 2 cards: 7/7 passed, 0 failed
- 4 cards: 7/7 passed, 0 failed

## Default-table status

This result supports the model-free gfx1100 readiness check, including the GDN recurrence parity path used by
`STRATA_GDN_HEAD` and `STRATA_GDN_PP`.

It does not prove any of the eight default candidates are ready to enter `arch_defaults.cpp` or a widened verify gate.
The remaining required evidence is:

- per-switch A/B with full output identity for 4K, 32K and 64K prompts;
- at least two model formats;
- repeated starts with same-day baseline timing;
- no OOM, timeout, NaN or memory-peak regression.
