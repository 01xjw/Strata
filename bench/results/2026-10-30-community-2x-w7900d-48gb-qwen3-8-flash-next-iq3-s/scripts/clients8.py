"""The strata_bench.py --clients 8 test alone: 8 requests at once (serve_load.one, 400 words, 128 new tokens), N rounds.

usage: clients8.py URL TOOLS_DIR ROUNDS   (key from STRATA_KEY); prints one JSON object
"""
import json
import os
import secrets
import sys
import threading
import time

url, tools, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, tools)
import serve_load  # noqa: E402

key = os.environ["STRATA_KEY"]
nonce = secrets.token_hex(4)
rows = []
for rd in range(rounds):
    res: dict = {}
    th = []
    t0 = time.perf_counter()
    for i in range(8):
        t = threading.Thread(target=serve_load.one, args=(url, key, "strata", i, 400, 128, res, {}, False, f" b-8-{rd}-{nonce}"))
        t.start()
        th.append(t)
    for t in th:
        t.join()
    wall = time.perf_counter() - t0
    toks = sum(r["tokens"] for r in res.values())
    firsts = sorted(r["first"] for r in res.values() if r["first"] is not None)
    rows.append({"round": rd, "wall_s": round(wall, 2), "tokens": toks, "total_tps": round(toks / wall, 2),
                 "first_p50_s": round(firsts[len(firsts) // 2], 3) if firsts else None,
                 "errors": sum(1 for r in res.values() if r["err"])})
print(json.dumps({"rounds": rows, "loadavg": open("/proc/loadavg").read().split()[:3]}))
