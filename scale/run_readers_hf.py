#!/usr/bin/env python3
"""Fallback reader runner on plain transformers (no vLLM).

Same inputs and outputs as run_readers.py. SciFact prompts are scored, not
generated: the log-probability of each of the three labels as the assistant's
reply is compared and the argmax written, which is parse-free and equivalent to
structured decoding over the label set. QA prompts are generated greedily with
a short cap. Batched, left-padded, resumable. Multi-GPU via device_map=auto.

Usage:
  python run_readers_hf.py --model microsoft/phi-4 --jobs-root $PROJ/jobs --out-root $PROJ/out [--batch 16] [--limit 30]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

LABELS = ["SUPPORT", "CONTRADICT", "NOINFO"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dirs", nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--qa-max-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    reader = args.model.split("/")[-1].lower().replace("-instruct", "").replace("instruct", "")
    jobs_root, out_root = Path(args.jobs_root), Path(args.out_root)
    dirs = args.dirs or sorted(d.name for d in jobs_root.iterdir() if (d / "jobs").is_dir())
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map="auto")
    model.eval()
    dev = next(model.parameters()).device
    print(f"[hf] {args.model} loaded in {time.time()-t0:.0f}s on {torch.cuda.device_count()} GPU(s); dirs={dirs}", flush=True)

    def chat(prompt: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)

    label_ids = [tok(lab, add_special_tokens=False)["input_ids"] for lab in LABELS]

    @torch.no_grad()
    def score_labels(prompts: list[str]) -> list[str]:
        out = []
        for p in prompts:
            base = tok(chat(p), return_tensors="pt", add_special_tokens=False).to(dev)
            n = base["input_ids"].shape[1]
            scores = []
            for ids in label_ids:
                full = torch.cat([base["input_ids"], torch.tensor([ids], device=dev)], dim=1)
                lp = torch.log_softmax(model(input_ids=full).logits[0, n - 1:-1].float(), dim=-1)
                scores.append(sum(lp[i, t].item() for i, t in enumerate(ids)))
            out.append(LABELS[int(torch.tensor(scores).argmax())])
        return out

    @torch.no_grad()
    def generate(prompts: list[str]) -> list[str]:
        enc = tok([chat(p) for p in prompts], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        gen = model.generate(**enc, max_new_tokens=args.qa_max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]

    for d in dirs:
        meta = json.load((jobs_root / d / "meta.json").open())
        out_dir = out_root / f"out_{reader}_{d}"; out_dir.mkdir(parents=True, exist_ok=True)
        prompts = sorted((jobs_root / d / "jobs").glob("*.txt"))
        if args.limit:
            prompts = prompts[: args.limit]
        todo = [p for p in prompts if not (out_dir / p.name).exists() or not (out_dir / p.name).read_text().strip()]
        print(f"[hf] {d}: {len(prompts)} prompts, {len(todo)} to run ({meta['endpoint']})", flush=True)
        t1 = time.time()
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            texts = [p.read_text() for p in chunk]
            answers = score_labels(texts) if meta["endpoint"] == "scifact" else generate(texts)
            for p, a in zip(chunk, answers):
                (out_dir / p.name).write_text(a)
            if (i // args.batch) % 10 == 0:
                print(f"[hf] {d}: {min(i+args.batch, len(todo))}/{len(todo)} ({(i+len(chunk))/(time.time()-t1):.1f}/s)", flush=True)
        print(f"[hf] {d}: done in {time.time()-t1:.0f}s -> {out_dir}", flush=True)
    print(f"[hf] all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
