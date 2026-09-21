"""Judge prompt files with a local Ollama model.

Usage: judge_ollama.py <model> <jobs_dir> <out_dir> [max_tokens]

Sequential over the local HTTP API (Ollama serializes GPU work anyway),
resumable: existing non-empty outputs are skipped.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MODEL, JOBS, OUT = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 12
URL = "http://localhost:11434/api/generate"

OUT.mkdir(parents=True, exist_ok=True)
jobs = sorted(JOBS.glob("*.txt"))
todo = [j for j in jobs if not (OUT / j.name).exists() or not (OUT / j.name).read_text().strip()]
print(f"{len(jobs)} jobs, {len(todo)} to run with {MODEL}", flush=True)

def ask(prompt: str) -> str | None:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": MAX_TOKENS},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["response"].strip()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == 2:
                print(f"  failed: {e}", flush=True)
                return None
            time.sleep(2 * (attempt + 1))
    return None

t0 = time.time()
for i, j in enumerate(todo, 1):
    resp = ask(j.read_text())
    if resp is not None:
        (OUT / j.name).write_text(resp)
    if i % 25 == 0 or i == len(todo):
        el = time.time() - t0
        print(f"{i}/{len(todo)}  {el/i:.1f}s/job  eta {(len(todo)-i)*el/i/60:.0f}min", flush=True)

done = sum(1 for j in jobs if (OUT / j.name).exists() and (OUT / j.name).read_text().strip())
print(f"done: {done}/{len(jobs)}")
