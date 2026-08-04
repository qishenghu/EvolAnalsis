"""Aggregate rebuttal runs into the tables we will paste into the response.

Reads validation logs directly (not the queue's markdown, which can lag), groups
runs into the reviewer-facing comparisons, and emits mean/std where we have
replicates.

Usage:
  PYTHONPATH=. python scripts/aggregate_rebuttal_results.py
  PYTHONPATH=. python scripts/aggregate_rebuttal_results.py --out NeurIPS_2026_Latex/data/rebuttal_tables.md
"""
import argparse
import glob
import json
import os
import statistics as stats

# Paper reference points (single seed 2026) — sources cited in the report.
PAPER = {
    "1.5B-AF DUET": 47.5, "1.5B-AF SFT+GRPO": 30.0, "1.5B-AF CHORD": 27.0,
    "1.5B-AF -SC": 31.0, "1.5B-AF -DR3": 47.5,
    "1.5B-WS DUET": 36.0, "1.5B-WS SFT+GRPO": 18.5, "1.5B-WS CHORD": 11.5,
    "1.5B-WS -SC": 1.0, "1.5B-WS -DR3": 9.5,
}

# Historical runs already on disk that belong in the multi-seed story.
KNOWN = {
    # name -> (group, seed)
    "alfworld_qwen1.5b_duet_v39c_postfix": ("1.5B-AF DUET", 2026),
    "webshop_qwen1.5b_duet_swC_02_pk03_v10_floor06": ("1.5B-WS DUET", 2026),
    "webshop_qwen1.5b_sft_rl": ("1.5B-WS SFT+GRPO", 2026),
    "alfworld_qwen1.5b_sft_rl": ("1.5B-AF SFT+GRPO", 2026),
    "alfworld_qwen1.5b_duet_minus_dr3": ("1.5B-AF DUET -DR3", 2026),
    "alfworld_qwen1.5b_duet_minus_dr3_seed2025": ("1.5B-AF DUET -DR3", 2025),
    "alfworld_qwen1.5b_duet_minus_dr3_seed2027": ("1.5B-AF DUET -DR3", 2027),
    "webshop_qwen1.5b_duet_a100_seed2025": ("1.5B-WS DUET", 2025),
    "webshop_qwen1.5b_duet_a100_seed2027": ("1.5B-WS DUET", 2027),
    "webshop_qwen1.5b_sft_rl_a100_seed2025": ("1.5B-WS SFT+GRPO", 2025),
    "webshop_qwen1.5b_sft_rl_a100_seed2027": ("1.5B-WS SFT+GRPO", 2027),
    "alfworld_qwen1.5b_duet_h200_seed2026": ("1.5B-AF DUET", 2026),
    "alfworld_qwen1.5b_duet_h200_seed2025": ("1.5B-AF DUET", 2025),
    "alfworld_qwen1.5b_duet_h200_seed2027": ("1.5B-AF DUET", 2027),
    "alfworld_qwen1.5b_sft_rl_h200_seed2025": ("1.5B-AF SFT+GRPO", 2025),
    "alfworld_qwen1.5b_sft_rl_h200_seed2027": ("1.5B-AF SFT+GRPO", 2027),
    # generalization / sensitivity runs (single seed each, reported standalone)
    "alfworld_qwen1.5b_duet_a100_obsnoise_hash": ("SC: 30% obs noise, exact match", 2026),
    "alfworld_qwen1.5b_duet_a100_obsnoise_soft": ("SC: 30% obs noise, soft match", 2026),
    "alfworld_qwen1.5b_duet_a100_soft_clean": ("SC: clean obs, soft match", 2026),
    "alfworld_qwen1.5b_duet_h200_cache10": ("Teacher cache 10%", 2026),
    "alfworld_qwen1.5b_duet_h200_cache1": ("Teacher cache 1%", 2026),
    "alfworld_qwen1.5b_duet_h200_ntch2": ("Teacher mix 2/group", 2026),
    "alfworld_qwen1.5b_duet_a100_teacher14b": ("Teacher = Qwen2.5-14B", 2026),
    "alfworld_qwen1.5b_duet_a100_teacher32b": ("Teacher = Qwen2.5-32B", 2026),
}


def read_val(path):
    n = strict = lenient = 0
    rw = 0.0
    with open(path) as f:
        for line in f:
            try:
                x = json.loads(line)
            except Exception:
                continue
            n += 1
            s = float(x.get("score", x.get("reward", 0.0)))
            rw += s
            strict += s >= 1.0
            lenient += s >= 0.9
    if n == 0:
        return None
    return {"n": n, "strict": strict / n * 100, "lenient": lenient / n * 100, "reward": rw / n}


def collect():
    """One record per run. Prefers val@100; falls back to val@50 for the SFT+GRPO
    baseline, whose RL phase ends at step 50 (its 50 SFT steps + 50 GRPO steps make
    the same 100-step optimization budget as the other methods)."""
    out = []
    seen = set()
    for path in sorted(glob.glob("experiments/*/*/validation_log/*.jsonl")):
        name = path.split("/")[2]
        step = os.path.basename(path).split(".")[0]
        if step not in ("50", "100"):
            continue
        has100 = os.path.exists(path.replace(f"/{step}.jsonl", "/100.jsonl"))
        if step == "50" and has100:
            continue          # prefer val@100 when both exist
        if name in seen:
            continue
        r = read_val(path)
        if not r:
            continue
        seen.add(name)
        r["at_step"] = int(step)
        group, seed = KNOWN.get(name, (None, None))
        out.append({"name": name, "group": group, "seed": seed, **r})
    return sorted(out, key=lambda d: (d["group"] or "zz", d["seed"] or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="NeurIPS_2026_Latex/data/rebuttal_tables.md")
    args = ap.parse_args()

    runs = collect()
    known = [r for r in runs if r["group"]]
    groups = {}
    for r in known:
        groups.setdefault(r["group"], []).append(r)

    lines = ["# Rebuttal result tables (auto-generated)", "",
             "Metric: strict success rate (score >= 1.0) on the 200-task validation split,",
             "read from `experiments/*/*/validation_log/`. val@100 is used where it exists;",
             "the SFT+GRPO baseline reports val@50 because its RL phase ends at step 50",
             "(50 SFT + 50 GRPO steps = the same 100-step optimization budget as the others).",
             "Paper reference values are shown in parentheses where they differ.", "",
             "## Multi-seed groups", "",
             "WebShop's score is continuous and strict success requires an exact match on every",
             "requested attribute, so mean reward is reported alongside: it is far less sensitive to",
             "where a run sits in its phase transition. ALFWorld's reward is binary, so the two",
             "columns coincide there.", "",
             "| setting | seeds | strict SR per seed | mean | std | mean reward per seed |",
             "|---|---|---|---|---|---|"]
    for g, rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        vals = [r["strict"] for r in rs]
        seeds = ",".join(str(r["seed"]) for r in rs)
        m = stats.mean(vals)
        sd = stats.stdev(vals) if len(vals) > 1 else 0.0
        per = " / ".join(f"{v:.1f}%" for v in vals)
        rew = " / ".join(f"{r['reward']:.3f}" for r in rs)
        lines.append(f"| {g} | {seeds} | {per} | **{m:.1f}%** | {sd:.1f} | {rew} |")

    lines += ["", "## Single-run settings", "",
              "| setting | strict SR | lenient SR | mean reward | n |", "|---|---|---|---|---|"]
    for g, rs in sorted(groups.items()):
        if len(rs) >= 2:
            continue
        r = rs[0]
        ref = PAPER.get(g)
        ref_s = f" (paper: {ref}%)" if ref else ""
        lines.append(f"| {g}{ref_s} | {r['strict']:.1f}% | {r['lenient']:.1f}% | {r['reward']:.4f} | {r['n']} |")

    unknown = [r for r in runs if not r["group"]]
    if unknown:
        lines += ["", "## Other completed runs on disk (not mapped to a rebuttal group)", "",
                  "| run | strict SR | n |", "|---|---|---|"]
        for r in unknown[:40]:
            lines.append(f"| {r['name']} | {r['strict']:.1f}% | {r['n']} |")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
