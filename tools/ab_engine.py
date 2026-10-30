"""A/B two engine builds or environments on one config: restart the server per arm, alternate arms over rounds.

    python tools/ab_engine.py strata-q2_0.json A:exe=/path/a B:exe=/path/b,STRATA_EXP_MODE=0 [--rounds 3]

An arm is NAME:key=value,...; `exe` replaces the config's engine, `xargs` appends ';'-separated engine arguments,
every other key is set in the engine's environment.  Each round starts the server once per arm (the arms' order alternates), warms it up with one short
answer, then sends the same requests: greedy, fixed prompts, a short chat set and one long prompt.  Prints the
server's own timings (llama.cpp's `timings`: decode and prompt tokens per second) per request and the per-arm
medians.  Decode numbers move run to run (the adaptive expert tier, the OS), so compare medians over rounds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)


def long_prompt(words: int) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def post(port: int, prompt: str, max_tokens: int) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                         "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def long_word_counts(value: str) -> list[int]:
    out = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError("--long-words entries must be positive")
        out.append(n)
    return out or [6000]


def load_tokenizer(config: dict):
    tok = config.get("tokenizer")
    if not tok:
        return None
    path = Path(tok)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        return None, "python tokenizers package is not installed"
    try:
        return Tokenizer.from_file(str(path)), None
    except Exception as exc:  # noqa: BLE001 - tokenizers versions reject some shipped tokenizer JSONs
        return None, f"could not load {path}: {exc}"


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1800) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    post(port, "Say hi.", 8)
                    return
        except Exception:
            pass
        time.sleep(3)
    raise TimeoutError("server not ready")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--long-words", default="6000",
                    help="comma-separated long prompt word counts, e.g. 4000,32000,64000")
    ap.add_argument("--log", default="ab_engine.jsonl")
    ap.add_argument("--save-text-dir", default=None,
                    help="write every full response body and token ids there, and include SHA256/path in the JSONL")
    a = ap.parse_args()
    base = json.loads(Path(a.config).read_text())
    tokenizer, tokenizer_error = load_tokenizer(base)
    if tokenizer_error:
        print(f"warning: token ids will not be saved: {tokenizer_error}", file=sys.stderr, flush=True)
    text_dir = Path(a.save_text_dir) if a.save_text_dir else None
    if text_dir is not None:
        text_dir.mkdir(parents=True, exist_ok=True)
    arms = []
    for s in a.arms:
        name, _, kv = s.partition(":")
        env = dict(x.split("=", 1) for x in kv.split(",") if x)
        arms.append((name, env))
    reqs = [(f"chat{i}", p) for i, p in enumerate(PROMPTS)]
    reqs += [(f"long{words}", long_prompt(words)) for words in long_word_counts(a.long_words)]
    out = open(a.log, "a")
    res: dict = {n: {} for n, _ in arms}
    for rnd in range(a.rounds):
        order = arms if rnd % 2 == 0 else arms[::-1]
        for name, env in order:
            cfg = dict(base)
            e = dict(env)
            if "exe" in e:
                cfg["exe"] = e.pop("exe")
            if "xargs" in e:                       # extra engine arguments, ';'-separated
                cfg["args"] = list(base["args"]) + e.pop("xargs").split(";")
            cfg["port"] = a.port
            cfg["log"] = str(Path(tempfile.gettempdir()) / f"ab_{name}_r{rnd}.log")
            cpath = Path(tempfile.gettempdir()) / f"ab_{name}.json"
            cpath.write_text(json.dumps(cfg, indent=1))
            penv = dict(os.environ, **e)
            proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                                     str(cpath), "--port", str(a.port)], cwd=str(ROOT), env=penv,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                t0 = time.time()
                wait_ready(a.port, proc)
                print(f"round {rnd} {name}: ready in {time.time() - t0:.0f} s", flush=True)
                for tag, p in reqs:
                    try:
                        r = post(a.port, p, a.max_new)
                    except Exception as exc:  # noqa: BLE001 - keep the failed arm's evidence and move on
                        row = {"round": rnd, "arm": name, "req": tag, "error": repr(exc)}
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        res[name].setdefault(tag, []).append(row)
                        print(f"  {tag:6s} ERROR {exc!r}", flush=True)
                        break
                    t = r.get("timings", {})
                    text = r["choices"][0]["message"].get("content", "")
                    text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    text_path = None
                    token_ids = tokenizer.encode(text, add_special_tokens=False).ids if tokenizer is not None else None
                    token_ids_path = None
                    token_ids_sha = None
                    if token_ids is not None:
                        token_blob = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                        token_ids_sha = hashlib.sha256(token_blob).hexdigest()
                    if text_dir is not None:
                        text_path = text_dir / f"round{rnd:02d}-{name}-{tag}.txt"
                        text_path.write_text(text, encoding="utf-8")
                        if token_ids is not None:
                            token_ids_path = text_dir / f"round{rnd:02d}-{name}-{tag}.tokens.json"
                            token_ids_path.write_text(json.dumps(token_ids, separators=(",", ":")), encoding="utf-8")
                    row = {"round": rnd, "arm": name, "req": tag, "decode": t.get("predicted_per_second"),
                           "prompt": t.get("prompt_per_second"), "prompt_n": t.get("prompt_n"),
                           "n": r.get("usage", {}).get("completion_tokens"),
                           "text": text[:80], "text_sha256": text_sha,
                           **({"text_path": str(text_path)} if text_path is not None else {}),
                           **({"token_ids_n": len(token_ids), "token_ids_sha256": token_ids_sha}
                              if token_ids is not None else {}),
                           **({"token_ids_path": str(token_ids_path)} if token_ids_path is not None else {}),
                           **({"token_ids_error": tokenizer_error} if token_ids is None and tokenizer_error else {})}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    res[name].setdefault(tag, []).append(row)
                    print(f"  {tag:6s} decode {row['decode']} tok/s  prompt {row['prompt']} tok/s ({row['prompt_n']})  "
                          f"n {row['n']}", flush=True)
            finally:
                proc.terminate()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                time.sleep(5)
    print("\nmedians over rounds:")
    for tag, _ in reqs:
        cells = []
        for name, _ in arms:
            rows = res[name].get(tag, [])
            d = statistics.median([x.get("decode") for x in rows if x.get("decode")]) if rows else float("nan")
            pr = statistics.median([x.get("prompt") for x in rows if x.get("prompt")]) if rows else float("nan")
            cells.append(f"{name}: decode {d:6.1f}  prompt {pr:7.1f}")
        print(f"  {tag:6s} | " + " | ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
