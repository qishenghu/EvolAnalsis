"""Offline calibration of soft (TF-IDF cosine) state matching for the SC.

Rebuttal artifact (NeurIPS 2026): measures, on the real 72B ALFWorld teacher
cache, (a) exact-hash hit rate under matcher-side word-dropout noise,
(b) soft-matching recall and progress MAE, (c) cross-task false-positive rate,
across similarity thresholds. Results: NeurIPS_2026_Latex/data/soft_match_calibration.md

Usage: python scripts/calibrate_soft_matching.py
"""
import pickle
import random
from collections import defaultdict
from types import SimpleNamespace

from agentevolver.module.exp_manager.state_progress import ExpertProgressMap

CACHE = "data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl"
NOISE = 0.3
N_TASKS = 100
N_QUERY_TASKS = 60


def main():
    with open(CACHE, "rb") as f:
        data = pickle.load(f)

    t2t = defaultdict(list)
    for d in data:
        t2t[d["task_id"]].append(SimpleNamespace(steps=d["messages"]))

    rng = random.Random(7)
    task_ids = rng.sample(list(t2t.keys()), N_TASKS)
    t2t_sub = {t: t2t[t] for t in task_ids}

    m_hash = ExpertProgressMap(t2t_sub, env_type="alfworld", match_mode="hash")
    m_hash_noisy = ExpertProgressMap(t2t_sub, env_type="alfworld", match_mode="hash")
    m_hash_noisy.obs_noise_p = NOISE

    print(f"\nnoise p={NOISE}, {N_QUERY_TASKS} tasks x <=20 obs each:")
    print(f"{'theta':>6} {'hash-hit':>9} {'soft-hit':>9} {'soft-MAE':>9} {'novel-FP':>9}")
    for theta in (0.3, 0.4, 0.5, 0.6, 0.7):
        m_soft = ExpertProgressMap(
            t2t_sub, env_type="alfworld", match_mode="soft",
            soft_sim_threshold=theta, obs_noise_p=NOISE,
        )
        hash_hits = soft_hits = n = fp = n_fp = 0
        soft_mae = 0.0
        for ti, t in enumerate(task_ids[:N_QUERY_TASKS]):
            pmap = m_hash.progress_maps.get(t, {})
            for obs, true_prog in rng.sample(list(pmap.items()), min(20, len(pmap))):
                n += 1
                if m_hash_noisy.get_potential(t, obs) > 0:
                    hash_hits += 1
                v = m_soft.get_potential(t, obs)
                if v > 0:
                    soft_hits += 1
                    soft_mae += abs(v - true_prog)
            other = task_ids[(ti + 31) % len(task_ids)]
            for obs in list(m_hash.progress_maps.get(other, {}).keys())[:10]:
                n_fp += 1
                if m_soft.get_potential(t, obs) > 0:
                    fp += 1
        print(f"{theta:>6} {hash_hits/n:>9.1%} {soft_hits/n:>9.1%} "
              f"{soft_mae/max(soft_hits,1):>9.3f} {fp/max(n_fp,1):>9.1%}")


if __name__ == "__main__":
    main()
