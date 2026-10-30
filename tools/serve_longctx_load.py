#!/usr/bin/env python3
"""Concurrent long-context load generator for a running Strata server.

This is the long-prompt counterpart of tools/serve_load.py.  It builds fixed
32K/64K-style prompts from the repository's synthetic word-list prompt,
starts C clients at once, and records queueing, TTFT, latency, prompt speed and
decode speed.  Each request gets a unique nonce so prompt cache reuse cannot
hide the cost of reading the long context.

Example:

    python3 tools/serve_longctx_load.py http://127.0.0.1:8080 \
      --clients 32,64 --prompt-tokens 32768,64000 --rounds 1 \
      --json bench/results/longctx-load.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ab_engine import long_prompt  # noqa: E402
from strata_bench import chat, digest, http_json  # noqa: E402


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def stat(values: list[float | int | None]) -> dict | None:
    xs = [x for x in values if isinstance(x, (int, float))]
    if not xs:
        return None
    return {"median": round(statistics.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3), "n": len(xs)}


def parse_ints(value: str) -> list[int]:
    out = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise SystemExit("empty integer list")
    return out


def calibrate(base: str, key: str, think: bool) -> tuple[float, float]:
    """Estimate prompt tokens per generated word using two small requests."""
    pts = []
    nonce = str(int(time.time()))
    for words in (200, 800):
        prompt = f"[longctx-load-cal-{nonce}-{words}]\n" + long_prompt(words)
        pts.append(digest(chat(base, key, [{"role": "user", "content": prompt}], 1, think)).get("prompt_tokens"))
    slope = (pts[1] - pts[0]) / 600 if all(pts) and pts[1] > pts[0] else None
    tokens_per_word = slope or 1.1
    overhead = max(0.0, (pts[0] or 0) - 200 * tokens_per_word) if slope else 40.0
    return tokens_per_word, overhead


def make_prompt(target_tokens: int, req_id: int, nonce: str, tokens_per_word: float, overhead: float) -> str:
    words = max(50, int((target_tokens - overhead) / tokens_per_word))
    return f"[longctx-load {nonce} target={target_tokens} req={req_id}]\n" + long_prompt(words)


def one_request(base: str, key: str, model: str, think: bool, prompt: str, max_tokens: int,
                timeout: float, start: threading.Event, out: dict, req_id: int) -> None:
    start.wait()
    t0 = time.perf_counter()
    try:
        res = chat(base, key, [{"role": "user", "content": prompt}], max_tokens, think, timeout=timeout)
        row = digest(res)
        row["text_len"] = len(res.get("text") or "")
        row["model"] = model
    except Exception as exc:  # noqa: BLE001
        row = {"ok": False, "err": repr(exc), "total_s": round(time.perf_counter() - t0, 3)}
    out[req_id] = row


def summarize(rows: list[dict], wall: float) -> dict:
    good = [r for r in rows if r.get("ok")]
    firsts = [r.get("ttft_s") for r in good if isinstance(r.get("ttft_s"), (int, float))]
    lats = [r.get("total_s") for r in rows if isinstance(r.get("total_s"), (int, float))]
    gen = sum((r.get("generated_tokens") or 0) for r in good)
    return {
        "requests": len(rows),
        "ok": len(good),
        "errors": len(rows) - len(good),
        "wall_s": round(wall, 3),
        "total_generated_tokens": gen,
        "total_decode_tps": round(gen / max(wall, 1e-6), 3),
        "first_p50": round(pct(firsts, 0.5), 3),
        "first_p95": round(pct(firsts, 0.95), 3),
        "lat_p50": round(pct(lats, 0.5), 3),
        "lat_p95": round(pct(lats, 0.95), 3),
        "prompt_tokens": stat([r.get("prompt_tokens") for r in good]),
        "prompt_tps": stat([r.get("prompt_tps") for r in good]),
        "decode_tps": stat([r.get("decode_tps") for r in good]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url")
    ap.add_argument("--clients", default="32,64", help="comma-separated concurrent request counts")
    ap.add_argument("--prompt-tokens", default="32768,64000", help="comma-separated target prompt token counts")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--model", default="strata")
    ap.add_argument("--key", default=os.environ.get("STRATA_KEY", ""))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--nonce", default=str(int(time.time())))
    ap.add_argument("--save", help="write every response digest under this directory")
    ap.add_argument("--json", help="write machine-readable results")
    a = ap.parse_args()

    base = a.url.rstrip("/")
    clients = parse_ints(a.clients)
    targets = parse_ints(a.prompt_tokens)
    try:
        ctx = int((http_json(base, "/metrics", a.key).get("engine") or {}).get("max_context") or 0)
    except Exception:  # noqa: BLE001
        ctx = 0
    tokens_per_word, overhead = calibrate(base, a.key, a.think)
    print(f"calibrated: {tokens_per_word:.3f} tokens/word, overhead {overhead:.1f}, context {ctx or 'unknown'}", flush=True)

    all_rows = []
    for target in targets:
        if ctx and target + a.max_tokens + 256 > ctx:
            print(f"target={target}: skipped, context is {ctx}", flush=True)
            continue
        for c in clients:
            for rd in range(a.rounds):
                start = threading.Event()
                res: dict[int, dict] = {}
                threads = []
                for i in range(c):
                    prompt = make_prompt(target, i, f"{a.nonce}-{target}-{c}-{rd}", tokens_per_word, overhead)
                    t = threading.Thread(target=one_request,
                                         args=(base, a.key, a.model, a.think, prompt, a.max_tokens,
                                               a.timeout, start, res, i))
                    t.start()
                    threads.append(t)
                t0 = time.perf_counter()
                start.set()
                for t in threads:
                    t.join()
                wall = time.perf_counter() - t0
                rows = [res[i] for i in sorted(res)]
                summary = summarize(rows, wall)
                entry = {"prompt_target_tokens": target, "clients": c, "round": rd,
                         "summary": summary, "requests": rows}
                all_rows.append(entry)
                if a.save:
                    d = Path(a.save) / f"t{target}" / f"c{c}" / f"r{rd}"
                    d.mkdir(parents=True, exist_ok=True)
                    (d / "requests.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
                print(
                    f"target={target} clients={c} round={rd}: "
                    f"ok={summary['ok']}/{summary['requests']} errors={summary['errors']} "
                    f"wall={summary['wall_s']:.1f}s first_p50/p95={summary['first_p50']:.2f}/{summary['first_p95']:.2f}s "
                    f"lat_p50/p95={summary['lat_p50']:.2f}/{summary['lat_p95']:.2f}s",
                    flush=True,
                )

    if a.json:
        Path(a.json).write_text(json.dumps(all_rows, indent=1), encoding="utf-8")
    return 0 if all((r["summary"]["errors"] == 0 for r in all_rows)) else 1


if __name__ == "__main__":
    sys.exit(main())
