"""Is DR3's discriminator separable on response length alone?

DR3 estimates w = pi_theta/pi_teacher from a discriminator over student-side
features (log-prob statistics + response length). If teacher and on-policy
trajectories differ enough in raw length, the discriminator can reach high
accuracy without learning anything about policy identity — the "your
discriminator is a length detector" objection. This script measures the
single-feature AUC so the claim can be answered with a number.

Usage:
  python scripts/diagnose_dr3_length_separability.py \
      --teacher data/teacher_trajectories/deepseek_v4/alfworld_dsv4pro_full.jsonl \
      --student-log logs/alfworld_qwen35_4b_grpo_b1024.log \
      --tokenizer /data/shared_models/Qwen3.5-4B-thinkraw
"""
import argparse
import json
import math
import random
import re


def auc(pos, neg):
    """P(random pos > random neg), ties counted as 0.5 (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    merged = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(merged):
        j = i
        while j < len(merged) and merged[j][0] == merged[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based average rank for the tie group
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum = sum(ranks[k] for k, (_, lab) in enumerate(merged) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--student-log", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-teacher", type=int, default=400)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # teacher: assistant tokens per trajectory (the quantity DR3 sees as resp_len)
    t_len, t_turns, t_think = [], [], []
    with open(args.teacher) as f:
        for line in f:
            if len(t_len) >= args.max_teacher:
                break
            rec = json.loads(line)
            msgs = [m for m in rec.get("messages", []) if m.get("role") == "assistant"]
            if len(msgs) <= 1:
                continue
            msgs = msgs[1:]  # drop the preamble ack turn
            n = sum(len(tok.encode(m.get("content", ""), add_special_tokens=False)) for m in msgs)
            th = sum(len(tok.encode(m["content"].split("</think>")[0], add_special_tokens=False))
                     for m in msgs if "</think>" in m.get("content", ""))
            t_len.append(n)
            t_turns.append(len(msgs))
            t_think.append(th)

    # student: per-step mean/std of response length from the training log.
    # response_length counts the whole multi-turn response span (actions +
    # observations), so scale by the logged llm-token ratio to get the
    # assistant-token count DR3 actually aggregates over.
    means, stds, ratios = [], [], []
    with open(args.student_log, errors="ignore") as f:
        for line in f:
            if "response_length/mean" not in line:
                continue
            m = re.search(r"response_length/mean:([0-9.]+)", line)
            s = re.search(r"response_length/std:([0-9.]+)", line)
            r = re.search(r"diag/llm_token_ratio_in_response:([0-9.]+)", line)
            d = re.search(r"diag/response_len_onpolicy_std:([0-9.]+)", line)
            if m:
                means.append(float(m.group(1)))
                stds.append(float((s or d).group(1)) if (s or d) else float("nan"))
                ratios.append(float(r.group(1)) if r else float("nan"))
    if not means:
        raise SystemExit("no response_length metrics in the student log yet")

    ratio = [x for x in ratios if not math.isnan(x)]
    ratio = sum(ratio) / len(ratio) if ratio else 1.0
    mu = sum(means) / len(means) * ratio
    sd_src = [x for x in stds if not math.isnan(x)]
    sd = (sum(sd_src) / len(sd_src) * ratio) if sd_src else mu * 0.4

    # sample a student population from the logged mean/std (log-normal keeps it positive)
    random.seed(0)
    sigma = math.sqrt(math.log(1 + (sd / mu) ** 2)) if mu > 0 else 0.5
    mean_log = math.log(max(mu, 1)) - sigma ** 2 / 2
    s_len = [math.exp(random.gauss(mean_log, sigma)) for _ in range(1000)]

    a = auc(s_len, t_len)
    print(f"teacher trajectories: n={len(t_len)}")
    print(f"  assistant tokens  p50={pct(t_len,50):.0f}  p90={pct(t_len,90):.0f}")
    print(f"  turns             p50={pct(t_turns,50):.0f}  mean={sum(t_turns)/len(t_turns):.1f}")
    print(f"  think tokens      p50={pct(t_think,50):.0f}  share={sum(t_think)/max(sum(t_len),1):.1%}")
    print(f"student (from {len(means)} logged steps, llm-token ratio {ratio:.3f})")
    print(f"  assistant tokens  mean~{mu:.0f}  std~{sd:.0f}")
    print()
    print(f"LENGTH-ONLY AUC (student vs teacher) = {a:.3f}")
    print("  >0.90 => the discriminator can separate on length alone;")
    print("          DR3's w_hat would carry length, not policy identity.")


if __name__ == "__main__":
    main()
