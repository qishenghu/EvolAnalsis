#!/usr/bin/env python3
"""Build the DeepSearch (Musique 3-4 hop) train/val/test splits.

Protocol (2026-08-05, user directive: training uses hard 3-4 hop only):
  - Source: FlashRAG-normalized MuSiQue-Ans (train 19,938 / dev 2,417).
  - Pool  : train items with hop == len(metadata.question_decomposition) in {3,4}
            (5,562 items). All items are answerable (musique-ans).
  - Split : follows the collector convention exactly
            (python_random_seed_then_shuffle_then_prefix_without_replacement):
            sorted pool ids -> Random(2026).shuffle -> [:1600] train curriculum,
            [1600:1800] val (disjoint by construction); dev 3-4 hop (1,165) is
            the frozen final test set. Hop ratio is preserved in expectation
            (~3.7:1); exact composition is recorded in the manifest.
  - IDs   : namespaced as musique_<orig id> to avoid collisions with any future
            second QA source.

Outputs (repo data/deepsearch/):
  tasks_train_pool.jsonl   task_id/question/golden_answers/hop  (5,562)
  tasks_test_dev34.jsonl   same schema                          (1,165)
  task_ids_1600_seed2026.txt / task_ids_val200_seed2026.txt /
  task_ids_test_dev34.txt  ordered id lists (+ sha256 printed and saved to
  SPLIT_MANIFEST.json alongside counts and per-hop composition).
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SEED = 2026
N_TRAIN = 1600
N_VAL = 200
HOPS = (3, 4)


def sha_ids(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def load(path, prefix):
    items = []
    for line in Path(path).open():
        r = json.loads(line)
        md = r.get("metadata", {})
        hop = len(md.get("question_decomposition", []))
        items.append(
            {
                "task_id": f"musique_{prefix}_{r['id']}" if not str(r["id"]).startswith(prefix) else f"musique_{r['id']}",
                "question": r["question"],
                "golden_answers": r["golden_answers"],
                "hop": hop,
            }
        )
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path,
                    default=Path("/projects_vol/gp_wangwy/qisheng/duet_h200/deepsearch/raw/musique"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data" / "deepsearch")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_all = load(args.raw_dir / "train.jsonl", "train")
    dev_all = load(args.raw_dir / "dev.jsonl", "dev")
    pool = [t for t in train_all if t["hop"] in HOPS]
    test = [t for t in dev_all if t["hop"] in HOPS]

    # collector convention: sorted pool ids -> seeded shuffle -> prefix
    by_id = {t["task_id"]: t for t in pool}
    ordered_ids = sorted(by_id)
    random.Random(SEED).shuffle(ordered_ids)
    picks = {
        "train": [by_id[i] for i in ordered_ids[:N_TRAIN]],
        "val": [by_id[i] for i in ordered_ids[N_TRAIN:N_TRAIN + N_VAL]],
    }

    def dump_tasks(path, items):
        with path.open("w") as f:
            for t in sorted(items, key=lambda t: t["task_id"]):
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    def dump_ids(path, items):
        ids = [t["task_id"] for t in items]
        path.write_text("\n".join(ids) + "\n")
        return sha_ids(ids)

    dump_tasks(args.out_dir / "tasks_train_pool.jsonl", pool)
    dump_tasks(args.out_dir / "tasks_test_dev34.jsonl", test)
    manifest = {
        "seed": SEED,
        "source": "FlashRAG_datasets musique (musique-ans normalized)",
        "pool_hops": dict(Counter(t["hop"] for t in pool)),
        "splits": {},
    }
    for name, fname, items in (
        ("train", f"task_ids_{N_TRAIN}_seed{SEED}.txt", picks["train"]),
        ("val", f"task_ids_val{N_VAL}_seed{SEED}.txt", picks["val"]),
        ("test", "task_ids_test_dev34.txt", sorted(test, key=lambda t: t["task_id"])),
    ):
        sha = dump_ids(args.out_dir / fname, items)
        manifest["splits"][name] = {
            "file": fname,
            "count": len(items),
            "hops": dict(Counter(t["hop"] for t in items)),
            "ordered_newline_sha256": sha,
        }
        print(f"{name}: {len(items)} items  hops={manifest['splits'][name]['hops']}  sha={sha[:16]}…")
    (args.out_dir / "SPLIT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print("manifest ->", args.out_dir / "SPLIT_MANIFEST.json")


if __name__ == "__main__":
    main()
