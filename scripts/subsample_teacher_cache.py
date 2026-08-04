"""Subsample a teacher trajectory cache (list-of-dicts pickle) for cache-size ablations.

Rebuttal experiment (NeurIPS 2026): reviewers y9x6/UyKJ asked how DUET behaves
with smaller teacher caches. We subsample uniformly at the trajectory level with
a fixed seed, which shrinks both per-task depth and task coverage — i.e. a
genuinely smaller cache, not a curated one.

Usage:
    python scripts/subsample_teacher_cache.py \
        --input data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl \
        --fraction 0.10 --seed 1234 \
        --output data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags_sub10.pkl
"""
import argparse
import pickle
import random
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    assert isinstance(data, list), f"expected list cache, got {type(data)}"

    n_keep = max(1, int(round(len(data) * args.fraction)))
    rng = random.Random(args.seed)
    subset = rng.sample(data, n_keep)

    def stats(trajs, label):
        tasks = Counter(t.get("task_id") for t in trajs)
        print(f"{label}: {len(trajs)} trajectories, {len(tasks)} unique tasks, "
              f"mean {len(trajs)/max(1,len(tasks)):.2f} traj/task")

    stats(data, "full cache")
    stats(subset, f"subsample ({args.fraction:.0%}, seed {args.seed})")

    with open(args.output, "wb") as f:
        pickle.dump(subset, f)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
