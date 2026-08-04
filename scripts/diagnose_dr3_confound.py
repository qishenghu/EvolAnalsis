"""DR3 discriminator confound diagnostic (NeurIPS 2026 rebuttal).

Reviewers UyKJ (Q3) and y9x6 (W2) ask: since the teacher cache is
success-filtered, is the DR3 discriminator separating TEACHER-vs-STUDENT
(distributional gap, as claimed) or merely SUCCESS-vs-FAILURE?

Protocol (offline, ~10 min on 1 GPU):
  1. Load saved on-policy student rollouts from a real DUET training run
     (success and failure buckets) + teacher trajectories from the cache.
  2. Score all trajectories with the same policy model; build the DR3 scalar
     feature family (assistant-token log-prob mean/std/min, low-prob tail
     ratios, response length — the advantage-free v3 family; reward/success
     is never an input, by construction).
  3. Train three probes on these features:
       D_all  : student(all)          vs teacher
       D_succ : student(success-only) vs teacher
       D_fail : student(failure-only) vs teacher
     If D_succ ~= D_all ~= high acc, separation is distributional, not
     success-driven (conditioning on success does not weaken it).
  4. Confound check: D_all's predicted P(student) on successful vs failed
     student rollouts. A success-detector would score successful student
     rollouts as "teacher-like" (low P(student)); a distribution detector
     scores both buckets as student.

Usage:
  python scripts/diagnose_dr3_confound.py \
      --traj_dir checkpoints/agentevolver/alfworld_qwen1.5b_duet_v39c_postfix/Trajectory \
      --teacher_pkl data/teacher_trajectories/qwen72b/alfworld_qwen72b_filtered_react_tags.pkl \
      --model_path models/Qwen2.5-1.5B-Instruct \
      --n_per_bucket 300 \
      --out NeurIPS_2026_Latex/data/dr3_confound_diagnostic.md

Note: scoring uses a fixed policy snapshot (default: the base student model,
approximating the early-training regime where the confound would matter most).
Pass a merged later checkpoint via --model_path for a late-training replica.
"""
import argparse
import glob
import json
import os
import pickle
import random

import torch


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def _step_of(path):
    try:
        return int(path.rsplit("step_", 1)[1].split(".")[0])
    except Exception:
        return -1


def load_student_rollouts(traj_dir, n_per_bucket, rng, step_min=None, step_max=None):
    succ, fail = [], []
    files = sorted(glob.glob(os.path.join(traj_dir, "trajectories_step_*.jsonl")))
    for fp in files:
        st = _step_of(fp)
        if step_min is not None and st < step_min:
            continue
        if step_max is not None and st > step_max:
            continue
        with open(fp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                msgs = d.get("messages") or []
                if not msgs:
                    continue
                rec = {"messages": msgs, "task_id": d.get("task_id"),
                       "step": st, "success": bool(d.get("success"))}
                (succ if rec["success"] else fail).append(rec)
    n_s, n_f = len(succ), len(fail)
    rng.shuffle(succ)
    rng.shuffle(fail)
    return succ[:n_per_bucket], fail[:n_per_bucket], (n_s, n_f)


def load_teacher(teacher_pkl, n, rng, task_whitelist=None):
    with open(teacher_pkl, "rb") as f:
        data = pickle.load(f)
    if task_whitelist:
        data = [d for d in data if d.get("task_id") in task_whitelist] or data
    rng.shuffle(data)
    out = []
    for d in data[: n * 2]:
        msgs = d.get("messages") or []
        if msgs:
            out.append({"messages": msgs, "task_id": d.get("task_id"), "success": True})
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------------------
# Scoring: assistant-token log-probs under a fixed policy
# ----------------------------------------------------------------------------

def assistant_token_logprobs(model, tokenizer, messages, device, max_len=8192):
    """Token log-probs restricted to assistant-message tokens.

    Renders the conversation incrementally with the chat template and marks
    the token span contributed by each assistant message.
    """
    spans = []
    prev_ids = None
    for i in range(len(messages)):
        try:
            ids = tokenizer.apply_chat_template(
                messages[: i + 1], tokenize=True, add_generation_prompt=False)
        except Exception:
            return None
        if prev_ids is not None and messages[i].get("role") == "assistant":
            spans.append((len(prev_ids), len(ids)))
        prev_ids = ids
    if prev_ids is None or not spans:
        return None
    input_ids = torch.tensor(prev_ids[:max_len], device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_ids).logits.float()
    logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
    tok_lp = logprobs.gather(-1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    keep = []
    for a, b in spans:
        a, b = max(a, 1), min(b, input_ids.shape[1])
        if b > a:
            keep.append(tok_lp[a - 1: b - 1])
    if not keep:
        return None
    return torch.cat(keep)


def features_from_logprobs(lp):
    """DR3 scalar feature family (advantage/reward-free)."""
    lp = lp.float()
    return torch.tensor([
        lp.mean(),
        lp.std() if lp.numel() > 1 else torch.tensor(0.0),
        lp.min(),
        (lp < -2.0).float().mean(),
        (lp < -4.0).float().mean(),
        torch.tensor(float(lp.numel())).log1p(),
    ])


# ----------------------------------------------------------------------------
# Probe: tiny logistic regression (torch, no sklearn dependency)
# ----------------------------------------------------------------------------

def train_probe(X, y, rng, epochs=300, lr=0.05, test_frac=0.25):
    idx = list(range(len(y)))
    rng.shuffle(idx)
    n_test = max(1, int(len(idx) * test_frac))
    te, tr = idx[:n_test], idx[n_test:]
    Xm, Xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - Xm) / Xs
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        logit = Xn[tr] @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = lambda ii: torch.sigmoid(Xn[ii] @ w + b)
        acc = ((pred(te) > 0.5).float() == y[te]).float().mean().item()
        score_all = torch.sigmoid(Xn @ w + b)
    return acc, score_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", required=True)
    ap.add_argument("--teacher_pkl", required=True)
    ap.add_argument("--model_path", default="models/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--n_per_bucket", type=int, default=300)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--step_min", type=int, default=None,
                    help="Only use student rollouts from training steps >= this")
    ap.add_argument("--step_max", type=int, default=None)
    ap.add_argument("--tag", default="", help="Label for this slice in the report")
    ap.add_argument("--out", default="NeurIPS_2026_Latex/data/dr3_confound_diagnostic.md")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16).to(device).eval()

    succ, fail, pool = load_student_rollouts(
        args.traj_dir, args.n_per_bucket, rng, args.step_min, args.step_max)
    tasks = {r["task_id"] for r in succ + fail}
    teacher = load_teacher(args.teacher_pkl, args.n_per_bucket, rng, tasks)
    print(f"loaded: student-success={len(succ)}, student-fail={len(fail)}, teacher={len(teacher)} "
          f"(pool: {pool[0]} succ / {pool[1]} fail; steps "
          f"{args.step_min if args.step_min is not None else 'all'}-"
          f"{args.step_max if args.step_max is not None else 'all'})")

    def featurize(recs, label):
        feats = []
        for i, r in enumerate(recs):
            lp = assistant_token_logprobs(model, tokenizer, r["messages"], device, args.max_len)
            if lp is None or lp.numel() == 0:
                continue
            feats.append((features_from_logprobs(lp.cpu()), r))
            if (i + 1) % 50 == 0:
                print(f"  scored {i+1}/{len(recs)} {label}")
        return feats

    f_succ = featurize(succ, "student-success")
    f_fail = featurize(fail, "student-fail")
    f_tch = featurize(teacher, "teacher")

    def stack(a, b):
        X = torch.stack([x for x, _ in a] + [x for x, _ in b])
        y = torch.cat([torch.ones(len(a)), torch.zeros(len(b))])  # 1=student, 0=teacher
        return X, y

    results = {}
    X_all, y_all = stack(f_succ + f_fail, f_tch)
    acc_all, scores_all = train_probe(X_all, y_all, rng)
    results["D_all  (student-all  vs teacher)"] = acc_all
    acc_succ, _ = train_probe(*stack(f_succ, f_tch), rng)
    results["D_succ (student-succ vs teacher)"] = acc_succ
    acc_fail, _ = train_probe(*stack(f_fail, f_tch), rng)
    results["D_fail (student-fail vs teacher)"] = acc_fail

    n_s, n_f = len(f_succ), len(f_fail)
    p_student_succ = scores_all[:n_s].mean().item()
    p_student_fail = scores_all[n_s:n_s + n_f].mean().item()
    p_student_tch = scores_all[n_s + n_f:].mean().item()

    slice_desc = args.tag or (
        f"training steps {args.step_min if args.step_min is not None else 1}"
        f"-{args.step_max if args.step_max is not None else 'end'}")
    lines = [
        f"# DR3 discriminator confound diagnostic — {slice_desc}",
        "",
        f"Policy snapshot for scoring: `{args.model_path}`  |  "
        f"rollout slice: {slice_desc} (pool {pool[0]} succ / {pool[1]} fail)  |  "
        f"n scored: succ={n_s}, fail={n_f}, teacher={len(f_tch)}  |  "
        "features: assistant-token logprob mean/std/min, tail(<-2), tail(<-4), log-length "
        "(advantage/reward-free family; success is never an input)",
        "",
        "| probe | held-out accuracy |",
        "|---|---|",
    ]
    for k, v in results.items():
        lines.append(f"| {k} | {v:.1%} |")
    lines += [
        "",
        "**Confound check** — D_all mean predicted P(student):",
        "",
        f"- student successful rollouts: {p_student_succ:.3f}",
        f"- student failed rollouts:     {p_student_fail:.3f}",
        f"- teacher rollouts:            {p_student_tch:.3f}",
        "",
        "Reading: if the discriminator were a success detector, D_succ would be "
        "near chance and successful student rollouts would score teacher-like "
        "(low P(student)). If it is distributional, D_succ stays high and both "
        "student buckets score as student.",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
