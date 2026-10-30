# RDNA3 gfx1100 8-switch restricted A/B summary

Date: 2026-10-13. Platform: one 8-GPU gfx1100 Kubernetes pod on a test node. Existing 2-card service on GPU0/1 was left running; switch A/B runs used an unused gfx1100 GPU through a second service in the pod.

## Verdict

- All 8 switches completed 5 rounds over 6 requests each (`60` rows per switch: 2 arms x 5 rounds x 6 requests).
- Full response text SHA256 matched for every paired A/B request (`0` mismatches for all switches).
- No switch is promoted to the RDNA3 architecture default table from this run. The timing is mixed, token-id capture failed because the local `tokenizers` library could not load the pack tokenizer, and the run covers only one model format (`IQ2_XS`) on gfx1100.
- A = baseline/disabled. B = candidate/enabled. Switches were tested one at a time, not cumulatively.

## Timing summary

Values are medians in tok/s. `Δ` is B vs A; positive is faster for the enabled candidate.

### `GDN_HEAD` (`gdn-head-full.jsonl`)

- Rows: `60`; text SHA mismatches: `0`.
- Read: No stable speed win; several chat/long prompt cases regressed and decode was flat/slightly lower.

| request | A prompt | B prompt | prompt Δ | A decode | B decode | decode Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat0` | 91.0 | 91.8 | +0.9% | 71.4 | 69.8 | -2.2% |
| `chat1` | 87.8 | 68.8 | -21.6% | 57.9 | 58.1 | +0.3% |
| `chat2` | 93.3 | 91.5 | -1.9% | 67.8 | 65.0 | -4.1% |
| `long4000` | 578.5 | 509.1 | -12.0% | 59.4 | 59.7 | +0.5% |
| `long32000` | 676.9 | 627.7 | -7.3% | 59.0 | 58.6 | -0.7% |
| `long56000` | 480.3 | 626.8 | +30.5% | 59.2 | 57.5 | -2.9% |

### `GDN_PP` (`gdn-pp-full.jsonl`)

- Rows: `60`; text SHA mismatches: `0`.
- Read: Prompt gains at long4000/long32000, but chat and long56000 were mixed.

| request | A prompt | B prompt | prompt Δ | A decode | B decode | decode Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat0` | 97.6 | 91.9 | -5.8% | 72.8 | 70.3 | -3.4% |
| `chat1` | 89.1 | 88.2 | -1.0% | 57.1 | 57.7 | +1.1% |
| `chat2` | 94.9 | 93.5 | -1.5% | 67.2 | 63.9 | -4.9% |
| `long4000` | 518.9 | 602.9 | +16.2% | 60.6 | 60.1 | -0.8% |
| `long32000` | 551.2 | 699.7 | +26.9% | 58.4 | 58.7 | +0.5% |
| `long56000` | 531.1 | 478.2 | -10.0% | 58.0 | 59.3 | +2.2% |

### `GDN_CONVL2` (`gdn-convl2-full.jsonl`)

- Rows: `60`; text SHA mismatches: `0`.
- Read: Strong long4000 prompt gain, but long32000/long56000 prompts and long56000 decode regressed.

| request | A prompt | B prompt | prompt Δ | A decode | B decode | decode Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat0` | 129.0 | 128.7 | -0.2% | 80.6 | 80.6 | +0.0% |
| `chat1` | 113.7 | 89.3 | -21.5% | 66.4 | 64.4 | -3.0% |
| `chat2` | 127.5 | 126.1 | -1.1% | 68.0 | 74.7 | +9.9% |
| `long4000` | 440.1 | 622.8 | +41.5% | 68.9 | 68.8 | -0.1% |
| `long32000` | 641.6 | 570.0 | -11.2% | 65.6 | 64.2 | -2.1% |
| `long56000` | 621.8 | 550.6 | -11.5% | 66.8 | 59.8 | -10.5% |

### `GDN_NOY` (`gdn-noy-full.jsonl`)

- Rows: `60`; text SHA mismatches: `0`.
- Read: Mostly neutral decode; prompt gains/losses were mixed by request length.

| request | A prompt | B prompt | prompt Δ | A decode | B decode | decode Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat0` | 132.3 | 129.5 | -2.1% | 80.7 | 80.6 | -0.1% |
| `chat1` | 113.7 | 111.4 | -2.0% | 66.5 | 66.5 | +0.0% |
| `chat2` | 130.7 | 126.0 | -3.6% | 73.9 | 75.0 | +1.5% |
| `long4000` | 597.2 | 676.6 | +13.3% | 68.8 | 68.7 | -0.1% |
| `long32000` | 662.4 | 644.7 | -2.7% | 67.3 | 67.3 | +0.0% |
| `long56000` | 670.8 | 587.5 | -12.4% | 66.8 | 66.8 | +0.0% |

### `CVEC_FUSE` (`cvec-fuse-full.jsonl`)

- Rows: `60`; text SHA mismatches: `0`.
- Read: Mixed prompt results and a severe long32000 decode regression/outlier, so not a default candidate.

| request | A prompt | B prompt | prompt Δ | A decode | B decode | decode Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat0` | 131.8 | 131.8 | +0.0% | 80.7 | 80.7 | +0.0% |
| `chat1` | 113.4 | 110.7 | -2.4% | 66.6 | 66.5 | -0.2% |
| `chat2` | 129.9 | 127.2 | -2.1% | 74.9 | 74.8 | -0.1% |
| `long4000` | 673.5 | 610.7 | -9.3% | 68.9 | 68.8 | -0.1% |
| `long32000` | 620.7 | 629.5 | +1.4% | 67.3 | 23.1 | -65.7% |
| `long56000` | 644.9 | 693.7 | +7.6% | 66.8 | 66.8 | +0.0% |

### `QFUSE_HC` (`qfuse-hc-full.jsonl`)

- Rows: `60`; text SHA mismatches: `0`.
- Read: Decode improved slightly across requests, but prompt speed was mixed and evidence is still incomplete.

| request | A prompt | B prompt | prompt Δ | A decode | B decode | decode Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat0` | 132.0 | 132.3 | +0.2% | 80.8 | 81.9 | +1.4% |
| `chat1` | 113.0 | 112.9 | -0.1% | 66.5 | 67.5 | +1.5% |
| `chat2` | 130.5 | 129.8 | -0.5% | 74.7 | 76.1 | +1.9% |
| `long4000` | 659.4 | 626.5 | -5.0% | 68.8 | 69.8 | +1.5% |
| `long32000` | 690.5 | 653.7 | -5.3% | 67.3 | 68.3 | +1.5% |
| `long56000` | 623.5 | 628.3 | +0.8% | 66.9 | 67.9 | +1.5% |

### `FUSE_WAIT` (`fuse-wait-full.jsonl`)

- Rows: `60`; text SHA mismatches: `0`.
- Read: Decode roughly flat/slightly up, but long4000/long32000 prompts regressed.

| request | A prompt | B prompt | prompt Δ | A decode | B decode | decode Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat0` | 129.3 | 131.2 | +1.5% | 80.7 | 80.7 | +0.0% |
| `chat1` | 121.6 | 111.1 | -8.6% | 66.4 | 66.5 | +0.2% |
| `chat2` | 129.0 | 128.4 | -0.5% | 73.3 | 74.7 | +1.9% |
| `long4000` | 657.9 | 565.5 | -14.0% | 68.8 | 68.9 | +0.1% |
| `long32000` | 640.3 | 537.8 | -16.0% | 67.1 | 67.3 | +0.3% |
| `long56000` | 617.8 | 629.5 | +1.9% | 66.8 | 66.8 | +0.0% |

### `FUSE_HITS` (`fuse-hits-full.jsonl`)

- Rows: `60`; text SHA mismatches: `0`.
- Read: Decode mostly flat, prompt speed mixed across lengths.

| request | A prompt | B prompt | prompt Δ | A decode | B decode | decode Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat0` | 132.1 | 129.8 | -1.7% | 80.8 | 80.6 | -0.2% |
| `chat1` | 112.7 | 114.9 | +2.0% | 66.5 | 66.5 | +0.0% |
| `chat2` | 129.8 | 127.4 | -1.8% | 75.1 | 76.7 | +2.1% |
| `long4000` | 687.3 | 585.4 | -14.8% | 68.8 | 68.8 | +0.0% |
| `long32000` | 628.8 | 628.3 | -0.1% | 67.3 | 67.3 | +0.0% |
| `long56000` | 609.7 | 714.1 | +17.1% | 66.9 | 66.8 | -0.1% |

## Limitations

- Token-id capture is missing. `tools/ab_engine.py` attempted to load `/data/packs/iq2_xs/tokenizer/tokenizer.json`, but `tokenizers` rejected it with `data did not match any variant of untagged enum ModelUntagged at line 2 column 16`; the JSONL therefore records full text and text SHA256, not canonical token ids.
- This is one model format (`IQ2_XS`) on physical `gfx1100`; it does not validate `gfx1101` or `gfx1102`, and it does not satisfy the two-model-format gate in `docs/RDNA3_VALIDATION.md`.
- The generated `long56000` input did not behave as a true 56K/64K token case for every run because tokenizer/text expansion capped the observed prompt length around the mid-20K range in this harness.
- A final cumulative run is still required before enabling any group of switches, even if an individual switch later passes token-id and two-model validation.

## Files

- `gdn-head-full.jsonl`
- `gdn-pp-full.jsonl`
- `gdn-convl2-full.jsonl`
- `gdn-noy-full.jsonl`
- `cvec-fuse-full.jsonl`
- `qfuse-hc-full.jsonl`
- `fuse-wait-full.jsonl`
- `fuse-hits-full.jsonl`
