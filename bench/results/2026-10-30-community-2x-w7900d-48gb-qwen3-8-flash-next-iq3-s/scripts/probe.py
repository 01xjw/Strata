#!/usr/bin/env python3
"""Fixed greedy prompts against a Strata server: prints JSON with each answer's sha256, token counts and timings.

Same request shape as tools/strata_bench.py (temperature 0, thinking off), but the prompts carry no per-run tag, so
two engines that compute the same tokens give the same hashes.  Usage: probe.py URL  (key from STRATA_KEY)."""
import hashlib
import json
import os
import sys
import time
import urllib.request

URL = sys.argv[1].rstrip("/")
KEY = os.environ.get("STRATA_KEY", "")
WORDS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet", "kilo", "lima",
         "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
         "xray", "yankee", "zulu"]


def word_list(n: int, seed: int) -> str:
    out, x = [], seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(WORDS[x % len(WORDS)])
    return " ".join(out)


PROMPTS = [
    ("short1", "Explain in three sentences why the sky is blue."),
    ("short2", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("short3", "List five prime numbers greater than 100 and explain how you checked one of them."),
    ("long4k", "Here is a list of words:\n" + word_list(2600, 7) +
     "\n\nHow many times does the word 'zulu' appear in the first 40 words? Then describe the list in one sentence."),
    ("long16k", "Here is a list of words:\n" + word_list(10500, 11) +
     "\n\nWhat are the last five words of the list? Answer briefly."),
]


def ask(prompt: str) -> dict:
    body = {"model": "strata", "stream": False, "max_tokens": 160, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + KEY} if KEY else {})})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.loads(r.read())
    msg = (j.get("choices") or [{}])[0].get("message") or {}
    text = (msg.get("content") or "") + "\x00" + (msg.get("reasoning_content") or "")
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "chars": len(text),
            "usage": j.get("usage"), "timings": j.get("timings"), "wall_s": round(time.perf_counter() - t0, 3)}


out = {}
for name, prompt in PROMPTS:
    try:
        out[name] = ask(prompt)
    except Exception as e:  # noqa: BLE001
        out[name] = {"error": str(e)[:300]}
print(json.dumps(out, indent=1))
