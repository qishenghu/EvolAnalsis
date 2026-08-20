"""尾巴 token 画像(2026-08-07):think 段 CLL 最深的 tail-frac token 长什么样。

输入:cll_teacher_profile.py 的 per-token jsonl(含 f16 压缩的 cll/entropy/logp
与 segments_rle)+ 对应教师原始 jsonl(取 completion_content 重新分词对齐,
与打分脚本同一调用:encode(content, add_special_tokens=False),逐 decision
校验长度一致,不一致即跳过并计数)。CPU only,登录节点可跑。

产出(--out 前缀):
  <out>.json  机器可读统计
  <out>.txt   人读报告(top 词表/熵/位置/连片/上下文样例)
"""

import argparse
import base64
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def expand_rle(rle: str):
    labels = []
    for part in rle.split(","):
        k, v = part.split(":")
        labels.extend([k] * int(v))
    return labels


def f16(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


def classify_token(text: str) -> str:
    t = text.strip()
    if not t:
        return "whitespace"
    if any("CJK" in unicodedata.name(c, "") for c in t[:2]):
        return "cjk"
    if all(not c.isalnum() for c in t):
        return "punct"
    if t.isdigit():
        return "digit"
    return "word"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", required=True, help="cll_*_2000.jsonl")
    ap.add_argument("--teacher", required=True, help="教师原始轨迹 jsonl")
    ap.add_argument("--tokenizer", default="/projects_vol/gp_wangwy/models/Qwen3.5-4B")
    ap.add_argument("--tail-frac", type=float, default=0.10)
    ap.add_argument("--out", required=True, help="输出前缀(不带扩展名)")
    ap.add_argument("--n-examples", type=int, default=10)
    ap.add_argument("--top-k", type=int, default=40)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=True)

    by_line = {}
    with open(a.analysis) as f:
        for ln in f:
            r = json.loads(ln)
            by_line.setdefault(int(r["line"]), []).append(r)

    ids_l, cll_l, h_l, lp_l = [], [], [], []
    # 每个 decision 一条:(global_start, n_think, meta_idx);think 内序即拼接序
    dec_spans, dec_meta = [], []
    skipped = 0
    n_dec = 0
    with open(a.teacher) as f:
        for lineno, ln in enumerate(f, start=1):
            recs = by_line.get(lineno)
            if not recs:
                continue
            trec = json.loads(ln)
            trace = trec.get("decision_trace") or []
            by_step = {int(d.get("step_index", i)): d for i, d in enumerate(trace)}
            for r in recs:
                d = by_step.get(int(r["step_index"]))
                if d is None:
                    skipped += 1
                    continue
                content = str(d.get("completion_content", ""))
                ids = tok.encode(content, add_special_tokens=False)
                cll = f16(r["cll_f16_b64"])
                if len(ids) != len(cll):
                    skipped += 1
                    continue
                labels = expand_rle(r["segments_rle"])
                if len(labels) != len(ids):
                    skipped += 1
                    continue
                think = np.array([lab == "think" for lab in labels])
                if not think.any():
                    continue
                idx = np.nonzero(think)[0]
                start = sum(len(x) for x in ids_l)
                ids_l.append(np.asarray(ids, dtype=np.int32)[idx])
                cll_l.append(cll[idx])
                h_l.append(f16(r["entropy_f16_b64"])[idx])
                lp_l.append(f16(r["logp_f16_b64"])[idx])
                dec_spans.append((start, len(idx), len(dec_meta)))
                dec_meta.append(
                    {
                        "rollout_id": r.get("rollout_id"),
                        "step_index": r.get("step_index"),
                        "full_ids": np.asarray(ids, dtype=np.int32),
                        "think_pos": idx,
                    }
                )
                n_dec += 1

    IDS = np.concatenate(ids_l)
    CLL = np.concatenate(cll_l)
    H = np.concatenate(h_l)
    LP = np.concatenate(lp_l)
    del ids_l, cll_l, h_l, lp_l
    n = len(IDS)
    thr = float(np.quantile(CLL, a.tail_frac))
    tail = CLL <= thr
    nt = int(tail.sum())

    # ---- 词表:tail 高频 + 富集倍数(该 token 落尾率 / 基线落尾率)----
    cnt_all = Counter(IDS.tolist())
    cnt_tail = Counter(IDS[tail].tolist())
    base_rate = nt / n
    top = []
    for tid, c in cnt_tail.most_common(a.top_k):
        tot = cnt_all[tid]
        top.append(
            {
                "token": tok.decode([tid]),
                "tail_n": c,
                "share_of_tail": c / nt,
                "overall_n": tot,
                "tail_rate": c / tot,
                "enrichment": (c / tot) / base_rate,
                "type": classify_token(tok.decode([tid])),
            }
        )
    top_share = sum(t["tail_n"] for t in top) / nt
    type_share = Counter()
    for tid, c in cnt_tail.items():
        type_share[classify_token(tok.decode([tid]))] += c

    # ---- 熵:尾巴处学生是"自信地另有所想"还是"本来就懵" ----
    def hstats(x):
        return {
            "mean": float(x.mean()),
            "p25": float(np.quantile(x, 0.25)),
            "p50": float(np.quantile(x, 0.50)),
            "p75": float(np.quantile(x, 0.75)),
            "frac_H_lt_0.5": float((x < 0.5).mean()),
            "frac_H_gt_1.5": float((x > 1.5).mean()),
        }

    # ---- 位置:tail 在 think 段内的归一化位置 decile 直方图 ----
    pos_hist = np.zeros(10)
    runs = Counter()
    tail_in_run_ge3 = 0
    first_tok_tail = 0
    for start, k, _ in dec_spans:
        m = tail[start : start + k]
        if k > 1:
            pos = np.nonzero(m)[0] / (k - 1)
            pos_hist += np.histogram(pos, bins=10, range=(0, 1))[0]
        if m[0]:
            first_tok_tail += 1
        # 连长统计
        run = 0
        for v in m:
            if v:
                run += 1
            elif run:
                runs[run] += 1
                if run >= 3:
                    tail_in_run_ge3 += run
                run = 0
        if run:
            runs[run] += 1
            if run >= 3:
                tail_in_run_ge3 += run
    pos_hist = (pos_hist / max(pos_hist.sum(), 1)).tolist()
    n_runs = sum(runs.values())
    run_stats = {
        "n_runs": n_runs,
        "mean_len": nt / max(n_runs, 1),
        "frac_tail_in_runs_ge2": 1 - runs[1] / max(nt, 1),
        "frac_tail_in_runs_ge3": tail_in_run_ge3 / max(nt, 1),
        "max_len": max(runs) if runs else 0,
        "len_hist": {str(k): runs[k] for k in sorted(runs)[:12]},
        "first_think_token_tail_rate": first_tok_tail / max(n_dec, 1),
    }

    # ---- 最深样例的上下文 ----
    order = np.argsort(CLL)[: a.n_examples * 3]
    examples, seen = [], set()
    starts = np.array([s for s, _, _ in dec_spans])
    for gi in order:
        di = int(np.searchsorted(starts, gi, side="right") - 1)
        s, k, mi = dec_spans[di]
        meta = dec_meta[mi]
        key = (meta["rollout_id"], meta["step_index"])
        if key in seen:
            continue
        seen.add(key)
        li = int(meta["think_pos"][gi - s])  # completion 内位置
        fid = meta["full_ids"]
        examples.append(
            {
                "rollout_id": meta["rollout_id"],
                "step_index": meta["step_index"],
                "cll": float(CLL[gi]),
                "H": float(H[gi]),
                "logp": float(LP[gi]),
                "token": tok.decode([int(fid[li])]),
                "context": tok.decode(fid[max(0, li - 40) : li])
                + " ⟦" + tok.decode([int(fid[li])]) + "⟧ "
                + tok.decode(fid[li + 1 : li + 25]),
            }
        )
        if len(examples) >= a.n_examples:
            break

    out = {
        "analysis": a.analysis,
        "teacher": a.teacher,
        "n_decisions": n_dec,
        "n_skipped": skipped,
        "n_think_tokens": n,
        "tail_frac": a.tail_frac,
        "tail_threshold_cll": thr,
        "n_tail": nt,
        "top_tokens": top,
        "top_share_of_tail": top_share,
        "tail_type_share": {k: v / nt for k, v in type_share.most_common()},
        "entropy_tail": hstats(H[tail]),
        "entropy_nontail": hstats(H[~tail]),
        "logp_tail_p50": float(np.median(LP[tail])),
        "position_decile_hist": pos_hist,
        "run_stats": run_stats,
        "examples": examples,
    }
    Path(a.out + ".json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    L = []
    L.append(f"== 尾巴画像 {Path(a.analysis).name}  (threshold CLL<={thr:.3f}, {nt}/{n} tokens, skipped={skipped})")
    L.append(f"-- top{a.top_k} 占尾巴 {top_share:.1%};类型构成 " + ", ".join(f"{k}:{v/nt:.1%}" for k, v in type_share.most_common()))
    L.append(f"{'token':<16}{'tail_n':>8}{'尾巴份额':>9}{'落尾率':>8}{'富集':>7}  type")
    for t in top[:25]:
        L.append(f"{t['token']!r:<16}{t['tail_n']:>8}{t['share_of_tail']:>9.2%}{t['tail_rate']:>8.1%}{t['enrichment']:>6.1f}x  {t['type']}")
    L.append(f"-- 熵@尾巴: {out['entropy_tail']}")
    L.append(f"-- 熵@非尾: {out['entropy_nontail']}")
    L.append(f"-- 位置 decile: " + " ".join(f"{p:.2f}" for p in pos_hist))
    L.append(f"-- 连片: {run_stats}")
    for e in examples:
        L.append(f"---- cll={e['cll']:.2f} H={e['H']:.2f} {e['rollout_id']} step{e['step_index']}")
        L.append("     " + e["context"].replace("\n", "⏎"))
    Path(a.out + ".txt").write_text("\n".join(L))
    print("\n".join(L[:40]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
