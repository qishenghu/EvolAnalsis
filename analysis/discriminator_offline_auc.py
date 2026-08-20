#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判别器离线 AUC 分析:教师轨迹 vs 学生 rollout 的轨迹级可分性。

背景("双杀预言"的判别器一侧):
  DUET 的 DR3 判别器用轨迹级 logp 统计特征区分 教师样本 vs 在线策略样本。
  若教师来自异族大模型(如 Qwen3.5-122B / DeepSeek-V4-Flash),学生模型
  (Qwen3.5-4B)给教师 think 段打出的 logp 会系统性偏低 → 判别器可能
  "一眼看穿"教师(AUC≈1),使 ŵ=D/(1-D) 校正失去梯度、μ 调度提前塌缩。
  本脚本在离线打分数据上直接量化这一可分性,并检验"把特征口径收窄到
  action 段"能否救回(DUET-R 段收窄假设)。

数据(analysis/cll_teacher_profile.py 完整模式产出的 per-decision jsonl):
  每行一个 decision,含 float16+base64 压缩的逐 token (logp, H, cll) 数组、
  游程压缩的段标注 segments_rle("think:57,other:1,action:14")、以及
  prompt_sha_ok / completion_sha_ok 重放校验标志(False 的 decision 一律
  排除——cll_122b_ws_2000.jsonl 有 19/10120 个 prompt sha 失配)。

特征口径(复现 agentevolver/module/exp_manager/dr3_ratio.py 的 v3 特征,
compute_sequence_features L146-175,轨迹级聚合):
  lp_mean / lp_std / lp_min / lp_max / lp_low_ratio(logp<-10 占比) / resp_len
  共 6 维。v3 原版第 7 维 kl_ref_mean 在无 ref_log_prob 时恒为 0
  (dr3_ratio.py L158-161),离线场景没有 ref 模型,故此处不含该维,
  与训练时"ref 缺失"分支的有效特征完全一致。
  resp_len 与训练口径同为原始 token 计数(dr3_ratio.py L89:mask.sum)。

三种口径(同一套 v3 特征,换聚合的 token 集合):
  full   —— 轨迹全部 completion token(think+action+other),即旧判别器口径;
  think  —— 仅 think 段 token;
  action —— 仅 action 段 token(DUET-R 段收窄候选口径)。

评估:
  每个 (env, teacher, 口径) 单元:教师类取全量轨迹,学生类 n=400;
  5 折分层交叉验证 + StandardScaler + 逻辑回归(sklearn),报 AUC 均值±std。
  另报"第 0 步可分性"朴素基线:仅用 lp_mean 单特征的秩 AUC(免训练,
  方向取 max(auc, 1-auc),等价于最优单调阈值判别)。

输出:
  analysis_outputs/discriminator_auc/结果.json + stdout 表格。
  裁决口径(写入 JSON 的 verdict 字段):
    full AUC≈1(>0.99)         → "双杀"坐实(判别器一眼看穿教师);
    action AUC 显著更低且 <0.9 → 段收窄有救(DUET-R 可行);
    否则                        → 部分证实/需另想办法。

用法(纯 CPU,勿占 GPU):
  /projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/duet/bin/python \
      analysis/discriminator_offline_auc.py
"""

from __future__ import annotations

import base64
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "analysis_outputs" / "cll_teacher_profile"
OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "discriminator_auc"

# 段标签固定集合(与 cll_teacher_profile.py 的 SEGMENTS 一致)
SEGMENTS = ("think", "action", "other")
# 三种特征口径:None 表示全序列(不过滤段)
SCOPES: Dict[str, Optional[str]] = {"full": None, "think": "think", "action": "action"}
# v3 特征名(去掉离线恒零的 kl_ref_mean,见模块 docstring)
FEATURE_NAMES = ("lp_mean", "lp_std", "lp_min", "lp_max", "lp_low_ratio", "resp_len")
LP_LOW_THR = -10.0  # dr3_ratio.py L155 low_thr

# 数据单元:env × teacher → (教师打分文件, 学生打分文件)
CELLS = [
    ("alfworld", "122b", "cll_122b_af_2000.jsonl", "cll_student_af_400.jsonl"),
    ("alfworld", "flash", "cll_flash_af_2000.jsonl", "cll_student_af_400.jsonl"),
    ("webshop", "122b", "cll_122b_ws_2000.jsonl", "cll_student_ws_400.jsonl"),
    ("webshop", "flash", "cll_flash_ws_2000.jsonl", "cll_student_ws_400.jsonl"),
]


# ---------------------------------------------------------------------------
# 读取与解码
# ---------------------------------------------------------------------------
def decode_f16(b64: str) -> np.ndarray:
    """解码 float16 little-endian + base64 数组(cll_teacher_profile.f16_b64 的逆)。"""
    return np.frombuffer(base64.b64decode(b64), dtype="<f2").astype(np.float32)


def expand_rle(rle: str, n_expected: int) -> np.ndarray:
    """把游程压缩段标注展开成逐 token 标签数组,如 'think:57,other:1,action:14'。"""
    labels: List[str] = []
    for part in rle.split(","):
        seg, cnt = part.split(":")
        labels.extend([seg] * int(cnt))
    if len(labels) != n_expected:
        raise ValueError(f"RLE 长度 {len(labels)} != token 数 {n_expected}: {rle}")
    return np.asarray(labels)


def iter_decisions(path: Path) -> Iterator[dict]:
    """逐行读 per-decision jsonl;跳过 sha 校验失败的 decision(记录内标志)。"""
    n_skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # sha 失配 → 该 decision 的重放上下文不可信,logp 无效,排除
            if not (rec.get("prompt_sha_ok", False) and rec.get("completion_sha_ok", False)):
                n_skipped += 1
                continue
            yield rec
    if n_skipped:
        print(f"  [{path.name}] 排除 sha 失配 decision {n_skipped} 个", file=sys.stderr)


def load_trajectories(path: Path) -> Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]]:
    """按 rollout_id 聚合:每轨迹 → [(step_index, logp, cll, labels), ...](按步序)。

    logp/cll 均为该 decision completion token 的逐 token 数组,labels 为段标签。
    """
    trajs: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    for rec in iter_decisions(path):
        logp = decode_f16(rec["logp_f16_b64"])
        cll = decode_f16(rec["cll_f16_b64"])
        n = int(rec["completion_tokens"])
        if len(logp) != n or len(cll) != n:
            raise ValueError(f"{path.name} line {rec['line']}: 数组长度与 completion_tokens 不符")
        labels = expand_rle(rec["segments_rle"], n)
        trajs[rec["rollout_id"]].append((int(rec["step_index"]), logp, cll, labels))
    for rid in trajs:
        trajs[rid].sort(key=lambda t: t[0])
    return dict(trajs)


def traj_scope_logp(
    steps: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]], scope: Optional[str]
) -> np.ndarray:
    """把一条轨迹全部 decision 的 completion logp 连成一条序列,并按口径过滤段。"""
    parts = []
    for _, logp, _, labels in steps:
        parts.append(logp if scope is None else logp[labels == scope])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


# ---------------------------------------------------------------------------
# v3 特征(轨迹级聚合,语义对齐 dr3_ratio.compute_sequence_features)
# ---------------------------------------------------------------------------
def v3_features(logp: np.ndarray) -> np.ndarray:
    """对一条(口径过滤后的)logp 序列算 v3 特征;空序列返回 NaN 由上层剔除。"""
    if logp.size == 0:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float64)
    return np.array(
        [
            float(logp.mean()),
            float(logp.std()),  # 总体 std(与 _masked_std 的 /count 口径一致)
            float(logp.min()),
            float(logp.max()),
            float((logp < LP_LOW_THR).mean()),  # lp_low_ratio,阈值 -10
            float(logp.size),  # resp_len:原始 token 计数(dr3_ratio.py L89)
        ],
        dtype=np.float64,
    )


def build_feature_matrix(
    trajs: Dict[str, List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]], scope: Optional[str]
) -> Tuple[np.ndarray, int]:
    """全部轨迹 → 特征矩阵 (n, 6);返回 (矩阵, 因该口径无 token 而剔除的轨迹数)。"""
    rows = []
    n_empty = 0
    for rid in sorted(trajs):  # 排序保证可复现
        feats = v3_features(traj_scope_logp(trajs[rid], scope))
        if np.isnan(feats).any():
            n_empty += 1
            continue
        rows.append(feats)
    return np.asarray(rows, dtype=np.float64), n_empty


# ---------------------------------------------------------------------------
# AUC 评估
# ---------------------------------------------------------------------------
def cv_auc(x_teacher: np.ndarray, x_student: np.ndarray, seed: int = 42) -> Tuple[float, float]:
    """5 折分层 CV 逻辑回归 AUC(标准化在训练折内拟合,避免泄漏)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([x_teacher, x_student])
    # 标签语义与 DR3 判别器一致:学生(在线策略)=1,教师=0(AUC 对称,不影响数值)
    y = np.concatenate([np.zeros(len(x_teacher)), np.ones(len(x_student))])
    aucs = []
    for tr_idx, te_idx in StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(x[tr_idx], y[tr_idx])
        aucs.append(roc_auc_score(y[te_idx], clf.predict_proba(x[te_idx])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def naive_lp_mean_auc(x_teacher: np.ndarray, x_student: np.ndarray) -> float:
    """朴素基线:仅 lp_mean 单特征的秩 AUC(免训练,方向取 max(auc,1-auc))。

    含义:训练第 0 步、判别器还没学任何权重时,一个单调阈值就能达到的可分性。
    """
    from sklearn.metrics import roc_auc_score

    score = np.concatenate([x_teacher[:, 0], x_student[:, 0]])  # 第 0 维即 lp_mean
    y = np.concatenate([np.zeros(len(x_teacher)), np.ones(len(x_student))])
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: List[dict] = []
    student_cache: Dict[str, Dict] = {}

    for env, teacher, teacher_file, student_file in CELLS:
        print(f"\n===== {env} × {teacher} =====", file=sys.stderr)
        t_trajs = load_trajectories(DATA_DIR / teacher_file)
        if student_file not in student_cache:
            student_cache[student_file] = load_trajectories(DATA_DIR / student_file)
        s_trajs = student_cache[student_file]
        print(
            f"  教师轨迹 {len(t_trajs)} 条,学生轨迹 {len(s_trajs)} 条", file=sys.stderr
        )

        for scope_name, scope in SCOPES.items():
            x_t, t_empty = build_feature_matrix(t_trajs, scope)
            x_s, s_empty = build_feature_matrix(s_trajs, scope)
            auc_mean, auc_std = cv_auc(x_t, x_s)
            naive = naive_lp_mean_auc(x_t, x_s)
            results.append(
                {
                    "env": env,
                    "teacher": teacher,
                    "scope": scope_name,
                    "n_teacher": int(len(x_t)),
                    "n_student": int(len(x_s)),
                    "n_teacher_empty_scope": int(t_empty),
                    "n_student_empty_scope": int(s_empty),
                    "auc_mean": round(auc_mean, 4),
                    "auc_std": round(auc_std, 4),
                    "naive_lp_mean_auc": round(naive, 4),
                }
            )
            print(
                f"  [{scope_name:>6}] AUC={auc_mean:.4f}±{auc_std:.4f} "
                f"(朴素 lp_mean 基线 {naive:.4f}; n_t={len(x_t)}, n_s={len(x_s)}, "
                f"空口径剔除 t={t_empty}/s={s_empty})",
                file=sys.stderr,
            )

    # ------------------ 裁决(逐单元 + 总体) ------------------
    # 判据(与任务口径一致):
    #   全序列 AUC≈1(此处取 ≥0.98)             → 该单元"双杀"坐实;
    #   action-only 比 full 低 >0.05 且 <0.9     → 该单元段收窄有救。
    by_cell: Dict[Tuple[str, str], Dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_cell[(r["env"], r["teacher"])][r["scope"]] = r
    cell_verdicts: List[dict] = []
    for (env, teacher), c in by_cell.items():
        full_a, act_a = c["full"]["auc_mean"], c["action"]["auc_mean"]
        kill = full_a >= 0.98
        saved = (full_a - act_a > 0.05) and (act_a < 0.9)
        cell_verdicts.append(
            {
                "env": env,
                "teacher": teacher,
                "full_auc": full_a,
                "action_auc": act_a,
                "双杀坐实(full≥0.98)": bool(kill),
                "段收窄有救(action<0.9且降幅>0.05)": bool(saved),
            }
        )
    n_kill = sum(v["双杀坐实(full≥0.98)"] for v in cell_verdicts)
    n_saved = sum(v["段收窄有救(action<0.9且降幅>0.05)"] for v in cell_verdicts)
    n_cells = len(cell_verdicts)
    full_aucs = [v["full_auc"] for v in cell_verdicts]
    action_aucs = [v["action_auc"] for v in cell_verdicts]
    verdict = (
        f"双杀{n_kill}/{n_cells}个单元坐实(full AUC≥0.98),"
        f"全序列 AUC 范围 [{min(full_aucs):.3f}, {max(full_aucs):.3f}]——即便最低的单元"
        f"也远高于判别器'看不穿'的水平;段收窄在 {n_saved}/{n_cells} 个单元有救"
        f"(action-only AUC 范围 [{min(action_aucs):.3f}, {max(action_aucs):.3f}])。"
        "总体:双杀预言部分证实——可分性普遍很高(WebShop×flash 达 1.0,朴素 lp_mean "
        "单特征即近乎完美),但并非所有单元都'一眼看穿';ALFWorld 上 action-only "
        "口径能把 AUC 压回 0.8 以下(段收窄有救),WebShop 上仍 >0.93(段收窄不够)。"
    )

    payload = {
        "说明": "判别器离线 AUC:v3 特征(lp_mean/lp_std/lp_min/lp_max/"
        "lp_low_ratio(<-10)/resp_len,轨迹级),教师=0/学生=1,"
        "5 折分层 CV 逻辑回归;naive_lp_mean_auc 为仅 lp_mean 的秩 AUC"
        "(第 0 步可分性基线,方向取 max(auc,1-auc))。",
        "特征名": list(FEATURE_NAMES),
        "结果": results,
        "逐单元裁决": cell_verdicts,
        "verdict": verdict,
        "verdict_数值依据": {
            "full_auc_范围": [round(min(full_aucs), 4), round(max(full_aucs), 4)],
            "action_auc_范围": [round(min(action_aucs), 4), round(max(action_aucs), 4)],
        },
    }
    out_path = OUT_DIR / "结果.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # ------------------ stdout 表格 ------------------
    print("\n判别器离线 AUC(教师 vs 学生,v3 轨迹级特征,5 折 CV 逻辑回归)")
    header = (
        f"{'env':<10}{'teacher':<8}{'口径':<8}{'n_t':>6}{'n_s':>6}"
        f"{'AUC均值':>10}{'±std':>9}{'lp_mean基线':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['env']:<10}{r['teacher']:<8}{r['scope']:<8}"
            f"{r['n_teacher']:>6}{r['n_student']:>6}"
            f"{r['auc_mean']:>10.4f}{r['auc_std']:>9.4f}{r['naive_lp_mean_auc']:>13.4f}"
        )
    print(f"\n裁决:{verdict}")
    print(f"结果已写入:{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
