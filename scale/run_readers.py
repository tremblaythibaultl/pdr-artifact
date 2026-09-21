#!/usr/bin/env python3
"""Run one open-weight reader over every prompt set with vLLM, offline batch.

For each jobs directory given, every <hash>.txt prompt is sent through the
model's chat template. SciFact prompts get structured decoding constrained to
the three labels (no parse failures by construction); QA prompts get a short
free-form answer. Outputs land in <out_root>/out_<reader>_<jobsdir>/<hash>.txt,
skipping prompts already answered, so a killed job resumes.

Usage:
  python run_readers.py --model microsoft/phi-4 --jobs-root $PROJ/jobs --out-root $PROJ/out \
      --tp 1 [--dirs scifact_k5_bge_clip99 ...] [--limit 50]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

LABELS = ["SUPPORT", "CONTRADICT", "NOINFO"]


def structured_choice(labels):
    """Structured decoding restricted to a set of strings, across vLLM API versions."""
    from vllm import SamplingParams
    try:  # vLLM >= 0.10
        from vllm.sampling_params import StructuredOutputsParams
        return SamplingParams(temperature=0.0, max_tokens=8, structured_outputs=StructuredOutputsParams(choice=labels))
    except (ImportError, TypeError):  # older API
        from vllm.sampling_params import GuidedDecodingParams
        return SamplingParams(temperature=0.0, max_tokens=8, guided_decoding=GuidedDecodingParams(choice=labels))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--dirs", nargs="*", default=None, help="jobs subdirs; default: all")
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--qa-max-tokens", type=int, default=24)
    ap.add_argument("--limit", type=int, default=None, help="smoke test: prompts per dir")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--quantization", default=None, help="e.g. awq for a quantized checkpoint")
    ap.add_argument("--enforce-eager", action="store_true", help="skip CUDA graph capture (slower, more robust)")
    ap.add_argument("--custom-all-reduce", action="store_true",
                    help="use vLLM's custom all-reduce for TP>1 (off by default: it crashed with an illegal memory access on the cluster we used)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    reader = args.model.split("/")[-1].lower().replace("-instruct", "").replace("instruct", "")
    jobs_root, out_root = Path(args.jobs_root), Path(args.out_root)
    dirs = args.dirs or sorted(d.name for d in jobs_root.iterdir() if (d / "jobs").is_dir())
    print(f"[readers] {args.model} tp={args.tp} dirs={dirs}", flush=True)

    kw = dict(model=args.model, tensor_parallel_size=args.tp, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, dtype=args.dtype, trust_remote_code=False,
              enforce_eager=args.enforce_eager, disable_custom_all_reduce=not args.custom_all_reduce)
    if args.quantization:
        kw["quantization"] = args.quantization
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[readers] model loaded in {time.time()-t0:.0f}s", flush=True)
    sp_label = structured_choice(LABELS)
    sp_qa = SamplingParams(temperature=0.0, max_tokens=args.qa_max_tokens)

    for d in dirs:
        meta = json.load((jobs_root / d / "meta.json").open())
        out_dir = out_root / f"out_{reader}_{d}"; out_dir.mkdir(parents=True, exist_ok=True)
        prompts = sorted((jobs_root / d / "jobs").glob("*.txt"))
        if args.limit:
            prompts = prompts[: args.limit]
        todo = [p for p in prompts if not (out_dir / p.name).exists() or not (out_dir / p.name).read_text().strip()]
        print(f"[readers] {d}: {len(prompts)} prompts, {len(todo)} to run ({meta['endpoint']})", flush=True)
        if not todo:
            continue
        # drop prompts that cannot fit the context (rare: SciFact k=5 with five long abstracts)
        tk = llm.get_tokenizer()
        budget = args.max_model_len - (8 if meta["endpoint"] == "scifact" else args.qa_max_tokens) - 64
        keep, dropped = [], 0
        for p in todo:
            if len(tk(p.read_text(), add_special_tokens=False)["input_ids"]) <= budget:
                keep.append(p)
            else:
                dropped += 1
        if dropped:
            print(f"[readers] {d}: skipping {dropped} prompts longer than the {args.max_model_len}-token context", flush=True)
        todo = keep
        convs = [[{"role": "user", "content": p.read_text()}] for p in todo]
        sp = sp_label if meta["endpoint"] == "scifact" else sp_qa
        t1 = time.time()
        outs = llm.chat(convs, sp, use_tqdm=False)
        for p, o in zip(todo, outs):
            (out_dir / p.name).write_text(o.outputs[0].text.strip())
        rate = len(todo) / (time.time() - t1)
        print(f"[readers] {d}: done {len(todo)} in {time.time()-t1:.0f}s ({rate:.1f}/s) -> {out_dir}", flush=True)
        if meta["endpoint"] == "scifact":
            from collections import Counter
            print("[readers] label counts:", Counter(o.outputs[0].text.strip() for o in outs), flush=True)
        else:
            print("[readers] sample answers:", [o.outputs[0].text.strip()[:40] for o in outs[:5]], flush=True)
    print(f"[readers] all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
