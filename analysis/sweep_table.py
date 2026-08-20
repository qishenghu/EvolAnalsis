"""扫评汇总:greedy / sampled pass@1 / pass@3(无偏估计)。

pass@k 用 Chen et al. 的无偏估计量:每题 n 次采样中 c 次成功时
    pass@k = 1 - C(n-c, k) / C(n, k)
再对任务取平均。n=4、k=3 时它比"至少成一次"更保守也更可比。
"""

import argparse
import collections
import glob
import json
from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def load(sweep_dir: str, step: int, mode: str):
    rows = []
    for f in sorted(glob.glob(f"{sweep_dir}/{step}_{mode}_shard[0-9].jsonl")):
        rows += [json.loads(l) for l in open(f)]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", required=True)
    ap.add_argument("--steps", default="10 20 30 40 50 60 70")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    print(f"{'step':>5} {'greedy':>8} {'pass@1':>8} {'pass@3':>8} {'tasks':>6}  {a.label}")
    for step in [int(s) for s in a.steps.split()]:
        g = load(a.sweep_dir, step, "greedy")
        s = load(a.sweep_dir, step, "sampled")
        if not g and not s:
            continue
        gr = (sum(1 for r in g if r.get("success")) / len(g)) if g else float("nan")
        per = collections.defaultdict(list)
        for r in s:
            per[str(r.get("task_id"))].append(bool(r.get("success")))
        if per:
            p1 = sum(sum(v) / len(v) for v in per.values()) / len(per)
            p3 = sum(pass_at_k(len(v), sum(v), 3) for v in per.values() if len(v) >= 3) / len(per)
        else:
            p1 = p3 = float("nan")
        print(f"{step:>5} {gr:>8.1%} {p1:>8.1%} {p3:>8.1%} {len(per):>6}")


if __name__ == "__main__":
    main()
