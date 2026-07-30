"""For each teacher (state_messages, a_teacher) pair, sample N continuations from π_θ under free
generation (temperature=1.0, top_p=1.0, max_new_tokens=256). Classify each continuation by its
action type and whether it matches the teacher's exact action string. Emit empirical probabilities:

    p_emit_any_action      = fraction of samples with a valid <action>...</action>
    p_emit_click_option    = fraction of samples with click[<non-nav, non-ASIN>]
    p_emit_click_buy_now   = fraction with click[buy now]
    p_emit_click_teacher   = fraction with exact teacher click inner string
    p_emit_teacher_option  = fraction where click inner equals teacher's click inner (ignoring case/whitespace)

Also keep a histogram of actions produced so we can see the mode for each state under each model.

Run per-model (one model loaded at a time). Output JSONL: one row per sample.
"""
import argparse
import json
import os
import re
import random
import time
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
ASIN_RE = re.compile(r"^b0[0-9a-z]{8}$", re.IGNORECASE)
NON_OPTION_CLICKS = {
    "buy now", "back to search", "< prev", "next >", "prev", "next",
    "description", "features", "reviews", "attributes", "search"
}


def parse_action(text: str):
    m = ACTION_RE.search(text)
    if not m:
        return None, None
    raw = m.group(1).strip()
    a = raw.lower()
    if a.startswith("click[") and a.endswith("]"):
        inner = a[len("click["):-1].strip()
        if inner == "buy now":
            return "buy_now", inner
        if ASIN_RE.match(inner):
            return "click_asin", inner
        if inner in NON_OPTION_CLICKS:
            return "nav_click", inner
        return "click_option", inner
    if a.startswith("search[") and a.endswith("]"):
        return "search", a[len("search["):-1].strip()
    return "other", raw


def load_model(model_path, device):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def sample_n(model, tok, device, prompt, n, max_new_tokens=256, temperature=1.0):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", None)
    # Expand to batch of n
    input_ids = input_ids.expand(n, -1).contiguous()
    attn = attn.expand(n, -1).contiguous() if attn is not None else None
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        return_dict_in_generate=True,
    )
    seqs = out.sequences[:, input_ids.shape[1]:]
    texts = tok.batch_decode(seqs, skip_special_tokens=True)
    return texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=60)
    p.add_argument("--n_samples_per_state", type=int, default=16)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    with open(args.samples) as f:
        samples = [json.loads(l) for l in f]

    # Deduplicate by (task_id, rollout_id, action) — teacher repeats same state-action in training
    seen = set(); dedup = []
    for s in samples:
        key = (s["task_id"], s["rollout_id"], s["action"])
        if key in seen: continue
        seen.add(key); dedup.append(s)
    print(f"Dedup: {len(samples)} -> {len(dedup)}")
    if args.max_samples > 0:
        dedup = dedup[:args.max_samples]
    samples = dedup

    model, tok = load_model(args.model, args.device)

    fout = open(args.output, "w")
    t0 = time.time()
    for i, s in enumerate(samples):
        prompt = tok.apply_chat_template(s["state_messages"], tokenize=False, add_generation_prompt=True)
        try:
            texts = sample_n(model, tok, args.device, prompt, args.n_samples_per_state,
                             max_new_tokens=256, temperature=args.temperature)
        except Exception as e:
            print(f"Error {i}: {e}")
            continue
        # Teacher action inner string (normalised)
        teacher_action = s["action"]  # e.g. 'click[155- yellow]'
        t_m = re.match(r"click\[(.*)\]", teacher_action.lower())
        teacher_inner = t_m.group(1).strip() if t_m else ""

        action_types = []
        inners = []
        teacher_matches = 0
        for t in texts:
            atype, inner = parse_action(t)
            action_types.append(atype)
            inners.append(inner)
            if atype == "click_option" and inner == teacher_inner:
                teacher_matches += 1

        from collections import Counter
        at_counts = Counter(action_types)
        inner_counts = Counter([(a, i) for (a, i) in zip(action_types, inners)])

        rec = {
            "tag": args.tag,
            "rollout_id": s["rollout_id"],
            "task_id": s["task_id"],
            "action": teacher_action,
            "teacher_inner": teacher_inner,
            "page": s["page"],
            "n": args.n_samples_per_state,
            "p_any_action":      (at_counts.get("click_option",0)+at_counts.get("buy_now",0)+at_counts.get("click_asin",0)+at_counts.get("nav_click",0)+at_counts.get("search",0)+at_counts.get("other",0)) / args.n_samples_per_state,
            "p_click_option":    at_counts.get("click_option", 0) / args.n_samples_per_state,
            "p_buy_now":         at_counts.get("buy_now", 0) / args.n_samples_per_state,
            "p_click_asin":      at_counts.get("click_asin", 0) / args.n_samples_per_state,
            "p_nav_click":       at_counts.get("nav_click", 0) / args.n_samples_per_state,
            "p_search":          at_counts.get("search", 0) / args.n_samples_per_state,
            "p_no_action":       at_counts.get(None, 0) / args.n_samples_per_state,
            "p_teacher_action":  teacher_matches / args.n_samples_per_state,
            "action_types":      list(at_counts.items()),
            "top_inners":        list(inner_counts.most_common(5)),
            "example_first":     texts[0][:400],
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if (i+1) % 5 == 0:
            dt = time.time() - t0
            print(f"[{args.tag}] {i+1}/{len(samples)} done ({dt:.1f}s, {dt/(i+1):.1f}s/sample)")
    fout.close()
    print(f"DONE: {args.output}")


if __name__ == "__main__":
    main()
