#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate-1 三张正式图:CLL 染色分布 / 家族轴对比 / 判别器离线 AUC。

数据来源:
  * analysis_outputs/cll_teacher_profile/cll_*_2000.jsonl —— 教师全量 per-decision
    打分(analysis/cll_teacher_profile.py 完整模式产出;sha 失配 decision 按记录
    内标志排除,复用 analysis/discriminator_offline_auc.py 的读取器)。
  * analysis_outputs/discriminator_auc/结果.json —— 任务一的 AUC 结果
    (先运行 analysis/discriminator_offline_auc.py)。

输出(PNG 200dpi + PDF,全中文标签与图注):
  analysis_outputs/figures_gate1/fig1_cll_染色分布.{png,pdf}
  analysis_outputs/figures_gate1/fig2_家族轴对比.{png,pdf}
  analysis_outputs/figures_gate1/fig3_判别器AUC.{png,pdf}

设计规范(dataviz 技能,浅色模式参考调色板):
  * 表面 #fcfcfb;主墨 #0b0b0b / 次墨 #52514e / 弱墨 #898781;
    网格发丝线 #e1e0d9(实线);基线 #c3c2b7。
  * 分类色按固定槽位:蓝 #2a78d6 / 橙 #eb6834 / 青绿 #1baf7a(前三槽
    已通过全配对色盲安全校验)。实体跨图保持同色:think 段=蓝、
    action 段=青绿;fig2 的两个教师家族用槽 1/2(蓝/橙);fig3 的
    "全序列"口径取下一空闲槽(橙)。
  * 细标记、发丝网格、图例 + 少量直接标注;文字一律用墨色而非系列色。
  * 图中所有数值保留三位小数。

用法(纯 CPU):
  /projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/duet/bin/python \
      analysis/make_figures_gate1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

from discriminator_offline_auc import (  # noqa: E402(复用同一读取/排除逻辑)
    DATA_DIR,
    decode_f16,
    expand_rle,
    iter_decisions,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_DIR = PROJECT_ROOT / "analysis_outputs" / "figures_gate1"
AUC_JSON = PROJECT_ROOT / "analysis_outputs" / "discriminator_auc" / "结果.json"

# ---------------- 调色板(dataviz 参考实例,浅色模式) ----------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"  # 主墨:标题
INK2 = "#52514e"  # 次墨:图注/数值标注
MUTED = "#898781"  # 弱墨:轴刻度
GRID = "#e1e0d9"  # 发丝网格(实线)
BASE = "#c3c2b7"  # 轴基线
C_BLUE = "#2a78d6"  # 槽1:think 段 / Qwen3.5-122B
C_ORANGE = "#eb6834"  # 槽2:DeepSeek-V4-Flash / "全序列"口径
C_AQUA = "#1baf7a"  # 槽3:action 段

# 四个数据组(env × teacher)的固定顺序与显示名
GROUPS: List[Tuple[str, str]] = [
    ("cll_122b_af_2000.jsonl", "ALFWorld\nQwen3.5-122B"),
    ("cll_flash_af_2000.jsonl", "ALFWorld\nDSV4-Flash"),
    ("cll_122b_ws_2000.jsonl", "WebShop\nQwen3.5-122B"),
    ("cll_flash_ws_2000.jsonl", "WebShop\nDSV4-Flash"),
]


def setup_style() -> None:
    """全局绘图风格:中文字体、细网格、去多余边框。"""
    plt.rcParams.update(
        {
            # 拉丁字形/符号用 DejaVu,中文按字形回退到 Droid Sans Fallback
            # (本机唯一覆盖中文的字体)。注意必须把具体字体名直接写进
            # font.family:若走通用名 sans-serif,matplotlib 只会解析出
            # 列表里第一个字体,逐字形回退(mpl>=3.6)不会生效。
            "font.family": ["DejaVu Sans", "Droid Sans Fallback"],
            "axes.unicode_minus": False,  # 中文字体缺 U+2212,用 ASCII 连字符
            "pdf.fonttype": 42,  # PDF 内嵌 TrueType,保证中文可复制/显示
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASE,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "axes.labelcolor": INK2,
            "text.color": INK,
        }
    )


def style_axes(ax) -> None:
    """单个坐标系的统一修饰:只留底边基线,y 向发丝网格。"""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.tick_params(length=0)


# ---------------------------------------------------------------------------
# 数据:per-token CLL 池(按组 × 段)
# ---------------------------------------------------------------------------
def load_cll_pools() -> Dict[str, Dict[str, np.ndarray]]:
    """每个教师文件 → {'think': 全部 think token 的 CLL, 'action': ...}。"""
    pools: Dict[str, Dict[str, np.ndarray]] = {}
    for fname, _ in GROUPS:
        parts: Dict[str, List[np.ndarray]] = {"think": [], "action": []}
        for rec in iter_decisions(DATA_DIR / fname):
            cll = decode_f16(rec["cll_f16_b64"])
            labels = expand_rle(rec["segments_rle"], int(rec["completion_tokens"]))
            for seg in parts:
                mask = labels == seg
                if mask.any():
                    parts[seg].append(cll[mask])
        pools[fname] = {seg: np.concatenate(p) for seg, p in parts.items()}
        print(
            f"  [{fname}] think tokens={pools[fname]['think'].size:,} "
            f"action tokens={pools[fname]['action'].size:,}",
            file=sys.stderr,
        )
    return pools


# ---------------------------------------------------------------------------
# 小提琴形状:直方图密度(免 KDE,百万级 token 也快)
# ---------------------------------------------------------------------------
def violin_shape(
    values: np.ndarray, y_lo: float, y_hi: float, n_bins: int = 440, smooth_sigma: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, float]:
    """返回 (bin 中心, 归一化半宽 0..1, 低于显示下限的占比)。

    用细直方图 + 高斯平滑近似密度;宽度按各自最大值归一(每把小提琴
    自身最宽处 = 1),形状忠实于密度,不做开方等夸张变换。
    """
    below = float((values < y_lo).mean())
    clipped = values[(values >= y_lo) & (values <= y_hi)]
    hist, edges = np.histogram(clipped, bins=n_bins, range=(y_lo, y_hi))
    centers = 0.5 * (edges[:-1] + edges[1:])
    # 高斯核平滑(半径 4σ)
    radius = int(4 * smooth_sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / smooth_sigma) ** 2)
    kernel /= kernel.sum()
    dens = np.convolve(hist.astype(np.float64), kernel, mode="same")
    if dens.max() > 0:
        dens /= dens.max()
    return centers, dens, below


def draw_violin(
    ax,
    x0: float,
    values: np.ndarray,
    color: str,
    y_lo: float,
    y_hi: float,
    half_width: float = 0.34,
) -> Dict[str, float]:
    """画一把小提琴(密度轮廓 + p1–p99 内线 + 中位数点),返回分位数。"""
    centers, dens, below = violin_shape(values, y_lo, y_hi)
    w = dens * half_width
    # 密度≈0 处轮廓宽度为 0,若照画会在小提琴中轴拖出一条贯穿全高的
    # "细杆",把尾部延展夸大到显示下限 → 用 NaN 截断,尾部延展只由
    # 内部 p1–p99 线表达。
    w_line = np.where(dens > 0.002, w, np.nan)
    ax.fill_betweenx(
        centers, x0 - w, x0 + w, facecolor=color, alpha=0.28, linewidth=0, zorder=2
    )
    ax.plot(x0 - w_line, centers, color=color, linewidth=1.2, zorder=3)
    ax.plot(x0 + w_line, centers, color=color, linewidth=1.2, zorder=3)
    q = {
        "p1": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p99": float(np.percentile(values, 99)),
        "below": below,
    }
    # 内线:p1–p99 的分布延展(细线),中位数用白圈点(表面色描边)
    ax.plot(
        [x0, x0],
        [max(q["p1"], y_lo), min(q["p99"], y_hi)],
        color=color,
        linewidth=1.4,
        solid_capstyle="round",
        zorder=4,
    )
    ax.scatter(
        [x0],
        [q["p50"]],
        s=26,
        facecolor=color,
        edgecolor=SURFACE,
        linewidth=1.2,
        zorder=5,
    )
    return q


def zero_line(ax, x_min: float, x_max: float) -> None:
    """CLL=0 参考线:虚线次墨(阈值语义),右端文字标注。"""
    ax.plot(
        [x_min, x_max], [0, 0], color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1
    )
    ax.text(
        x_max + 0.04,
        0,
        "CLL=0",
        ha="left",
        va="center",
        fontsize=9,
        color=INK2,
    )


# ---------------------------------------------------------------------------
# 图 1:CLL 染色分布(4 组 × think/action)
# ---------------------------------------------------------------------------
def make_fig1(pools: Dict[str, Dict[str, np.ndarray]]) -> None:
    y_lo, y_hi = -8.5, 2.5
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.subplots_adjust(left=0.08, right=0.93, top=0.87, bottom=0.28)
    style_axes(ax)

    offsets = {"think": -0.42, "action": +0.42}
    colors = {"think": C_BLUE, "action": C_AQUA}
    xticks, xlabels = [], []
    for gi, (fname, disp) in enumerate(GROUPS):
        xc = gi * 2.2
        xticks.append(xc)
        xlabels.append(disp.replace("\n", " × "))
        for seg in ("think", "action"):
            q = draw_violin(ax, xc + offsets[seg], pools[fname][seg], colors[seg], y_lo, y_hi)
            if seg == "think":
                # 选择性直接标注:think 的 p10(负尾读数)与截断占比
                ax.plot(
                    [xc + offsets[seg] - 0.30, xc + offsets[seg] + 0.30],
                    [q["p10"], q["p10"]],
                    color=INK2,
                    linewidth=1.0,
                    zorder=6,
                )
                ax.annotate(
                    f"p10={q['p10']:.3f}",
                    xy=(xc + offsets[seg], q["p10"] - 0.25),
                    ha="center",
                    va="top",
                    fontsize=8.5,
                    color=INK2,
                )
                if q["below"] > 0.001:
                    ax.text(
                        xc + offsets[seg],
                        y_lo + 0.12,
                        f"{100 * q['below']:.3f}% < {y_lo:g}(未画出)",
                        ha="center",
                        va="bottom",
                        fontsize=7.5,
                        color=MUTED,
                    )

    zero_line(ax, -1.1, GROUPS.index(GROUPS[-1]) * 2.2 + 1.1)
    ax.set_xlim(-1.3, (len(GROUPS) - 1) * 2.2 + 1.8)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xticks(xticks)
    ax.set_xticklabels([g[1] for g in GROUPS], fontsize=9.5)
    ax.set_ylabel("每 token CLL = log p + H(0 = 学生自采样的期望水平)", fontsize=10)

    # 图例(≥2 系列必配)+ 一处关键读数标注
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[s], alpha=0.4, edgecolor=colors[s])
        for s in ("think", "action")
    ]
    ax.legend(
        handles,
        ["think 段", "action 段"],
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        labelcolor=INK2,
    )
    ax.text(
        GROUPS.index(GROUPS[0]) * 2.2 + 0.42,
        1.55,
        "action 段:分布整体贴 0(“全绿”)",
        ha="left",
        va="center",
        fontsize=9,
        color=INK2,
    )

    ax.set_title(
        "教师轨迹 CLL 染色分布:action 段全体贴 0,think 段主体近零但负尾长",
        fontsize=13,
        color=INK,
        pad=14,
        loc="left",
    )
    fig.text(
        0.08,
        0.035,
        "图 1|四组教师全量轨迹(2 环境 × 2 教师)completion token 的逐 token CLL 分布,按 <think>/<action> 段拆开;\n"
        "小提琴宽度为各自归一化密度,竖线为 p1–p99,圆点为中位数,横刻线为 think 段 p10。读数:action 段中位数≈0.000\n"
        "且 p1–p99 几乎收拢在 0 附近(学生几乎可无损复刻);think 段主体亦贴 0,但负尾深长(p10 最低至 -1.728),\n"
        "异族教师(DSV4-Flash)负尾更重。",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK2,
        linespacing=1.55,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig1_cll_染色分布.{ext}", dpi=200)
    plt.close(fig)
    print("fig1 已输出", file=sys.stderr)


# ---------------------------------------------------------------------------
# 图 2:家族轴对比(think 段,122B vs Flash,AF/WS 两面板)
# ---------------------------------------------------------------------------
def make_fig2(pools: Dict[str, Dict[str, np.ndarray]]) -> None:
    y_lo, y_hi = -8.5, 2.5
    panels = [
        ("ALFWorld", "cll_122b_af_2000.jsonl", "cll_flash_af_2000.jsonl"),
        ("WebShop", "cll_122b_ws_2000.jsonl", "cll_flash_ws_2000.jsonl"),
    ]
    colors = {"122b": C_BLUE, "flash": C_ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 6.0), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.30, wspace=0.10)

    for ax, (env_name, f122, ffl) in zip(axes, panels):
        style_axes(ax)
        qs = {}
        for xi, (key, fname) in enumerate([("122b", f122), ("flash", ffl)]):
            q = draw_violin(
                ax, float(xi), pools[fname]["think"], colors[key], y_lo, y_hi, half_width=0.40
            )
            qs[key] = q
            # 突出尾部:p10 刻线 + 读数(三位小数)
            ax.plot(
                [xi - 0.36, xi + 0.36], [q["p10"], q["p10"]], color=INK2, linewidth=1.1, zorder=6
            )
            ax.annotate(
                f"p10={q['p10']:.3f}",
                xy=(xi, q["p10"] - 0.28),
                ha="center",
                va="top",
                fontsize=9,
                color=INK2,
            )
            if q["below"] > 0.001:
                ax.text(
                    xi,
                    y_lo + 0.12,
                    f"{100 * q['below']:.3f}% < {y_lo:g}(未画出)",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    color=MUTED,
                )
        zero_line(ax, -0.55, 1.55)
        ax.set_xlim(-0.75, 1.95)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Qwen3.5-122B\n(同族教师)", "DSV4-Flash\n(异族教师)"], fontsize=9.5)
        dp10 = qs["flash"]["p10"] - qs["122b"]["p10"]
        ax.set_title(
            f"{env_name}(p10 差 {dp10:+.3f})", fontsize=11, color=INK, pad=8, loc="left"
        )

    axes[0].set_ylabel("think 段每 token CLL(0 = 学生自采样的期望水平)", fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[k], alpha=0.4, edgecolor=colors[k])
        for k in ("122b", "flash")
    ]
    axes[1].legend(
        handles,
        ["Qwen3.5-122B(同族)", "DSV4-Flash(异族)"],
        loc="upper right",
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
    )
    fig.suptitle(
        "家族轴对比(仅 think 段):异族教师的负尾一致更深",
        fontsize=13,
        color=INK,
        x=0.08,
        y=0.95,
        ha="left",
    )
    fig.text(
        0.08,
        0.035,
        "图 2|同一环境、同一任务集、同一采集契约与上下文渲染,唯一变量是教师家族:同族 Qwen3.5-122B\n"
        "(与学生 Qwen3.5-4B 同系)vs 异族 DeepSeek-V4-Flash;小提琴为 think 段逐 token CLL 密度(各自归一),\n"
        "横刻线并标注 p10。读数:两个环境里异族教师的 p10 都显著更低(ALFWorld -0.896→-1.728,\n"
        "WebShop -0.140→-1.212),即 think 负尾主要由“教师家族”驱动,而非环境或任务。",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK2,
        linespacing=1.55,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig2_家族轴对比.{ext}", dpi=200)
    plt.close(fig)
    print("fig2 已输出", file=sys.stderr)


# ---------------------------------------------------------------------------
# 图 3:判别器离线 AUC(env×teacher 分组,三口径并列)
# ---------------------------------------------------------------------------
def make_fig3() -> None:
    payload = json.loads(AUC_JSON.read_text(encoding="utf-8"))
    rows = payload["结果"]
    cell = {(r["env"], r["teacher"], r["scope"]): r for r in rows}
    group_keys = [
        ("alfworld", "122b", "ALFWorld\nQwen3.5-122B"),
        ("alfworld", "flash", "ALFWorld\nDSV4-Flash"),
        ("webshop", "122b", "WebShop\nQwen3.5-122B"),
        ("webshop", "flash", "WebShop\nDSV4-Flash"),
    ]
    scopes = [
        ("full", "全序列", C_ORANGE),
        ("think", "仅 think 段", C_BLUE),
        ("action", "仅 action 段", C_AQUA),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.30)
    style_axes(ax)

    bar_w = 0.24
    gap = 0.03  # 相邻柱间表面色间隙
    base = 0.5  # 柱自随机水平 0.5 起画(AUC=0.5 即“完全不可分”)
    for si, (scope, _, color) in enumerate(scopes):
        xs, hs, errs = [], [], []
        for gi, (env, teacher, _) in enumerate(group_keys):
            r = cell[(env, teacher, scope)]
            xs.append(gi + (si - 1) * (bar_w + gap))
            hs.append(r["auc_mean"] - base)
            errs.append(r["auc_std"])
        ax.bar(
            xs,
            hs,
            width=bar_w,
            bottom=base,
            color=color,
            zorder=3,
            yerr=errs,
            error_kw={"ecolor": INK2, "elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
        )
        for x, h, e in zip(xs, hs, errs):
            # 数值标注放在误差线帽之上,避免与误差线重叠
            ax.text(
                x,
                base + h + e + 0.008,
                f"{base + h:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=INK2,
            )

    # 参考线:0.5(随机,即基线本身)与 0.9(“可用判别器”阈值)、1.0(完美)
    ax.axhline(base, color=BASE, linewidth=0.8, zorder=1)
    ax.axhline(0.9, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.text(3.62, 0.9, "AUC=0.9", ha="left", va="bottom", fontsize=9, color=INK2)
    ax.axhline(1.0, color=GRID, linewidth=0.8, zorder=1)
    ax.text(3.62, 0.502, "AUC=0.5(随机)", ha="left", va="bottom", fontsize=9, color=MUTED)

    ax.set_xlim(-0.6, 4.3)
    ax.set_ylim(0.5, 1.06)
    ax.set_xticks(range(len(group_keys)))
    ax.set_xticklabels([g[2] for g in group_keys], fontsize=9.5)
    ax.set_ylabel("教师 vs 学生 判别 AUC(5 折 CV 均值 ± 标准差)", fontsize=10)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for _, _, c in scopes
    ]
    ax.legend(
        handles,
        [label for _, label, _ in scopes],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.13),
        ncols=3,
        frameon=False,
        fontsize=9.5,
        labelcolor=INK2,
    )
    ax.set_title(
        "判别器离线 AUC:全序列/仅 think 近乎看穿,仅 action 显著回落",
        fontsize=13,
        color=INK,
        pad=30,
        loc="left",
    )
    fig.text(
        0.08,
        0.035,
        "图 3|复现 DUET 旧判别器 v3 轨迹级特征(lp_mean/lp_std/lp_min/lp_max/lp_low_ratio/resp_len),\n"
        "教师类取全量轨迹、学生类 n=400,5 折分层交叉验证逻辑回归;柱自 AUC=0.5(随机水平)起画,误差线为折间标准差。\n"
        "读数:全序列口径 AUC 0.931–1.000(3/4 单元 ≥0.98,“双杀”大体坐实);仅 action 口径在 ALFWorld\n"
        "回落到 0.9 以下(0.787/0.797,段收窄有救),在 WebShop 仍 ≥0.932(段收窄不够)。",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=INK2,
        linespacing=1.55,
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig3_判别器AUC.{ext}", dpi=200)
    plt.close(fig)
    print("fig3 已输出", file=sys.stderr)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    make_fig3()  # 只依赖结果.json,先出
    pools = load_cll_pools()
    make_fig1(pools)
    make_fig2(pools)
    print(f"三张图已写入 {OUT_DIR}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
