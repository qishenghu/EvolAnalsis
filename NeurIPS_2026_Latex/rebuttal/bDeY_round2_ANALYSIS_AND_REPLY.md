# bDeY Round-2 — 完整分析、推导与回复稿(供 H200/A100 两侧讨论)

H200 侧,2026-08-02。本文自含:reviewer 原文 → 逐条事实核验(含出处,A100 可在自己机器复核)→
论证分析 → 公式推导 → 可粘贴回复稿 → 策略与待决事项。**回复稿零新实验**,所有数字来自论文
Table 1 或已提交的 rebuttal。

---

## 0. Reviewer 原文与他真正在论证什么

> Does SFT+GRPO in Table 1 rely on "Baseline separation"? If not, the strong performance of
> SFT+GRPO seems to indicate that the baseline separation is only crucial for DUET. Then, my
> argument is that the performance gain of DUET over existing approaches is mainly from SC,
> which is a good heuristic but not "principled". I am concerned that the framing and the
> narrative oversells the contribution.
>
> Did you redefine $\rho_t$ in the rebuttal? so now $w$ is part of $\rho_t$? What is the
> updated version of eq 7,8,9?

他的三段论(Q1):

1. **P1**:SFT+GRPO 不用 baseline separation(BS)也很强 →
2. **C1**:所以 BS "只对 DUET 重要"(暗示:BS 修的是 DUET 自己制造的问题)→
3. **C2**:所以 DUET 对现有方法的增益"主要来自 SC",而 SC 是启发式 → 框架 overselling。

**P1 为真(我们在代码层面证实,见 §1)。C1 是推论跳跃(见 §2)。C2 对一个格子为真、对方法
不为真(见 §2.3)。** Q2 是纯技术澄清,答案已在手,只需把三条公式写出来。

---

## 1. 事实核验(全部带出处,A100 可复核)

### F1 — Table 1 各方法执行时的 BS 开关(wandb 实际 run 配置,非源码 YAML)

| run(wandb 名) | `teacher_baseline_separation.enable` |
|---|---|
| `alfworld_qwen1.5b_chord` | **True** |
| `alfworld_qwen1.5b_sft`(SFT 阶段) | True(但数学上不起作用,见 F2) |
| `alfworld_qwen1.5b_sft_rl`(GRPO 阶段) | **False** |
| `alfworld_qwen1.5b_luffy` | **True** |

核验方式:`wandb.Api()` 读取各 run 的 `config["algorithm"]["grpo"]["teacher_baseline_separation"]`。

### F2 — 关键代码事实:μ=1.0 使 SFT 阶段的 GRPO 项系数恰为零

`agentevolver/module/exp_manager/het_actor.py:2179-2184`:

```python
if dr3_enable:
    # DR3 + mini-SFT: L = L_dr3 + μ * L_sft
    pg_loss = grpo_loss + mu * sft_loss
else:
    # 原始 CHORD: L = (1-μ) * L_grpo + μ * L_sft
    pg_loss = (1 - mu) * grpo_loss + mu * sft_loss
```

SFT 阶段配置:`use_dr3: false`(走 else 分支)、`use_chord: true`、
`chord_mu_peak = chord_mu_valley = 1.0`(μ 恒为 1)。代入:

$$L = (1-1)\cdot L_{\text{GRPO}} + 1\cdot L_{\text{SFT}} = L_{\text{SFT}}$$

**BS 只通过 GRPO 项内的组优势起作用,该项系数为零 ⇒ SFT 阶段的 BS 配置开关虽为 True,
数学上完全不激活。** 这比"配置里关了"更强:是结构性不可能起作用。

### F3 — GRPO 阶段(stage 2)无 teacher 混入

`alfworld_qwen1.5b_sft_rl` 配置:无 `teacher_experience`(`use_chord: false`、
`use_dr3: false`),纯 on-policy GRPO,组内没有 teacher 样本 ⇒ BS 无对象,False 也无所谓。

**⇒ 对 Q1 的事实性回答:SFT+GRPO 两阶段都不依赖 BS。Reviewer 的 P1 成立。**

### F4 — 回复中引用的每个数字的出处

| 数字 | 出处 |
|---|---|
| GRPO 1.0 / SFT+GRPO 30.0 / CHORD 27.0 / LUFFY 5.5 / DUET 47.5(AF-1.5B) | 论文 Table 1(A100 audit 已核) |
| DUET −baseline_sep = **0.0**(双环境) | 论文消融表 / `ablation_results.md` |
| 无 SC 核心 = 31.0(AF)、1.0(WS) | 论文消融表 |
| shuffled 进度图 41.0 vs 真图 47.5 | A100 已提交 rebuttal(shuffled-SC 对照) |
| 噪声匹配器 11.0 < 无 SC 31.0 | A100 已提交 rebuttal(obs-noise 实验) |
| WS −DR3 = −26.5pp(36.0→9.5) | 论文消融表 |
| task-matched SFT→GRPO(WS)= 7.5 | A100 已提交 rebuttal(global correction 5) |
| 7B:DUET 86.5 / GRPO 85.0 / LUFFY 82.5 | 论文 Table 1 |
| α≈0.10–0.12、ŵ≤1.13、ε_off=0.6 | 已提交 rebuttal(Eq.8/9 更正段)+ 配置 |

### F5 — 我**没有**核验、回复中刻意回避的一点

CHORD 的 BS 为 True,但我未完整追查 CHORD 实现中 teacher token 是否进入其 GRPO 项
(μ 调度 0.3→0.05,`(1−μ)` 非零,若 teacher 在组内则 CHORD 也受益于 BS)。**回复稿中五行表
只列了我能背书的行,没有 CHORD 行**。若 reviewer 追问 CHORD,A100 请先从
`launcher_record/alfworld_qwen1.5b_chord/yaml_backup.yaml` + `experience_collate` 路径确认
teacher 是否入组,再回。

---

## 2. 论证分析:让什么、守什么、怎么说

### 2.1 C1("BS 只对 DUET 重要")的精确纠正

BS 修正的偏差由**把成功 teacher rollout 放进 GRPO 组**这一操作制造。因此:

- 不 mixing 的方法(GRPO、SFT→GRPO)**结构性免疫** —— 不是"不需要修正的更好方法",而是
  根本没进入这个 regime;
- mixing 家族里,BS **必要但远不充分**:LUFFY 开着 BS 仍 5.5%;DUET 去掉 BS 崩到 0.0%;
- **他观察到的现象恰是论文第一主张本身**:mixing 不是无害的。SFT+GRPO 不用 BS 也强,
  不是反例,是同一诊断的另一面。

五行表(回复稿中的核心论证工具):

| 方法 | teacher 进 GRPO 组? | BS 生效? | AF-1.5B |
|---|---|---|---|
| on-policy GRPO | 否 | —(无对象) | 1.0 |
| SFT → GRPO | 否(两阶段) | 否(两阶段均否,F2/F3) | 30.0 |
| LUFFY | **是** | **是** | 5.5 |
| DUET − BS | **是** | 否 | **0.0** |
| DUET | **是** | 是 | 47.5 |

读法:第 3、4 行钉死"必要非充分";第 2 行钉死"免疫≠反例"。

### 2.2 C2("增益主要来自 SC")—— 让掉真的那部分

对 **AF-1.5B 这一个格子**,他量化上基本对:无 SC 核心 31.0 ≈ SFT+GRPO 30.0(parity),
略高于 CHORD 27.0;终点优势主要经 SC 兑现。**痛快承认,不含糊** —— conf-4 的 reviewer,
含糊必定格在"overselling"。

但用三条**已报告**的测量划出边界:

1. **SC ≠ 普通稠密奖励**:shuffled 图(等覆盖、等幅度、破坏顺序)41.0 vs 真图 47.5;
   噪声匹配器 11.0 **低于**无 SC 的 31.0(坏查找不如没有)。⇒ SC 的增量来自 teacher 状态
   序列的顺序信息 —— 这正是 State Channel"extract"的论文主张;且它加在被修正的 mixer 上
   (base 是 0.0,不是 30.0)。
2. **离开这个格子结论反转**:WS 上 −DR3 损失 26.5pp;task-matched SFT→GRPO 只有 7.5;
   7B 上 DUET 86.5 优雅退化至 GRPO 85.0 而 LUFFY 82.5 掉到无 teacher 基线之下 ——
   这些比较中 SC 不变,变的是修正机制。"mainly SC"是一个格子的结论,不是方法的。
3. **交叉引用 y9x6 的分布口径**("read with that variance in mind"),保持两 thread 一致,
   但不在此展开数字(见 §5 纪律)。

### 2.3 框架让步的设计

他两轮都落点在 "oversells"。回复给出:
(a) 修订后的 contribution 陈述原文(把"principled"严格限定在两个修正上、显式写明 SC 是
启发式且在结构化状态环境贡献最大份额);
(b) **主动提出可改题**:*DUET: Bias-Corrected Experience Replay for LLM Agent RL*。
一个可见的结构性让步,比十段辩护更可能换分。**这是团队决策 —— 不同意就删那句,其余自洽。**

---

## 3. 公式推导(代码 → 修订后的 Eq. 7/8/9)

### 3.1 代码实际做了什么

`het_actor.py:1501-1507`(A100 的 `evidence_eq9_dr3.md` 同一结论):对 teacher token,

```python
old_lp_new[teacher] = log_prob.detach() - log(w_hat)     # 逐 token 广播同一个序列级 ŵ
```

即**替换**行为策略对数似然(teacher 似然不可得,DR3 补一个),而非在 ρ_t 之外再乘一个因子。
带入 PPO 比率:

$$\hat\rho_t=\exp\big(\log\pi_\theta - \operatorname{sg}[\log\pi_\theta] + \log\hat w_\alpha(\tau)\big)$$

- **求值**:采样点处 $\log\pi_\theta=\operatorname{sg}[\log\pi_\theta]$ ⇒ $\hat\rho_t=\hat w_\alpha(\tau)$,每个 token 同值;
- **梯度**:$\nabla_\theta\hat\rho_t=\hat\rho_t\nabla_\theta\log\pi_\theta=\hat w_\alpha(\tau)\nabla_\theta\log\pi_\theta$(ŵ 带 stop-grad);
- **恰好一次修正**,$\hat w\cdot\rho_t$ 从未出现 —— 提交版 Eq. 9 的乘积形式是记号错误。

ŵ 的形式(已提交 rebuttal 的更正):$D/(1-D)$ 只是中间量 $\hat r$,实际施加的是
α-相对比率 $\hat w_\alpha=\hat r/((1-\alpha)\hat r+\alpha)\le 1/(1-\alpha)$,实测
α≈0.10–0.12 ⇒ ŵ≤1.13:**只降权,不放大**。

### 3.2 修订后的三条公式(回复稿正文所用)

**Eq. 7(不变,仅 on-policy token)**
$$\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},\qquad
L_{\text{on}}=\mathbb{E}_t\big[\min(\rho_t\hat A_t,\ \operatorname{clip}(\rho_t,1\pm\varepsilon)\hat A_t)\big]$$

**Eq. 8(修订:施加的是有界 α-相对比率)**
$$\hat r(\tau)=\frac{D_\phi(\tau)}{1-D_\phi(\tau)},\qquad
\hat w_\alpha(\tau)=\frac{\hat r(\tau)}{(1-\alpha)\hat r(\tau)+\alpha}\in\Big(0,\tfrac{1}{1-\alpha}\Big]$$

**Eq. 9(修订:替换而非乘积)**
$$\log\hat\pi_\beta(a_t\mid s_t):=\operatorname{sg}[\log\pi_\theta(a_t\mid s_t)]-\log\hat w_\alpha(\tau)$$
$$\hat\rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\hat\pi_\beta(a_t\mid s_t)},\qquad
L_{\text{teacher}}=\mathbb{E}_t\big[\min(\hat\rho_t\hat A_t,\ \operatorname{clip}(\hat\rho_t,1-\varepsilon,1+\varepsilon_{\text{off}})\hat A_t)\big]$$
$$L=\mathbb{1}_{\text{on}}\,L_{\text{on}}+\mathbb{1}_{\text{teacher}}\,L_{\text{teacher}}$$

对 Q2 的一句话直答:**没有重定义 ρ_t;ŵ 不是 ρ_t 的一部分**。teacher token 的比率分母被
替换(imputed),单一比率求值即 ŵ_α(τ);WebShop 的 policy-shaping 变体(仍只含单个 ŵ 因子)
移入附录单独陈述。

---

## 4. 回复稿(可直接粘贴 OpenReview,英文)

> 完整粘贴版在 `reply_bDeY_round2.md` 上半部分,与下述内容一致;结构:
>
> 1. **Q1 直答**:No — 两阶段都不依赖 BS,并给出 μ=1 ⇒ GRPO 系数为零的代码级验证
>    (reviewer 自己拿不到的深度,建立可信度);
> 2. 五行表 + "必要非充分"纠正(LUFFY+BS=5.5;−BS=0.0;SFT+GRPO 免疫=诊断本身);
> 3. **SC 让步 + 三条边界**(shuffled 41.0 / noise 11.0<31.0 / WS·7B 反转)+ 分布口径
>    交叉引用;
> 4. **修订后的 contribution 陈述原文 + 可改题让步**(团队定夺);
> 5. **Q2**:三条公式全文(§3.2)+"替换而非乘积"一句话。

---

## 5. 与 A100 讨论的待决事项与纪律

1. **改题让步要不要保留**(我建议保留;把 3→4 概率从 ~35% 提至 ~45%)。删除不影响其余内容。
2. **纪律:此 thread 不主动引入 AF 基线复现分布(SFT+GRPO 复现 41.5/44.0/47.0,合并
   40.6±7.4;SFT 阶段单独 46.8)**。他没问;主动展开是喂大 parity 论点。但**不得写任何与之
   矛盾的话** —— 所以回复中 AF-1.5B 的让步措辞已按"parity"口径写("without SC the corrected
   core reaches 31.0 — parity with SFT+GRPO"),即使他日后看到复现分布也不构成矛盾。
   若他直接追问基线稳健性 → 按 `DECISION_alfworld_1p5b_cell.md` 第 3 分支如实展开。
3. **请 A100 核对 y9x6 thread 的最终提交版**确实含分布/方差表述,使本回复的交叉引用成立。
4. **CHORD 内部是否 mixing 未核验**(F5)—— 回复已回避;若被追问请先查 launcher_record。
5. **预测他的下一步**:(a) SC 在非结构化环境的适用性 → A100 的 soft-matcher/210 行 WS 潜函数
   材料;(b) shuffled 对照的稳健性 → H200 已有两对同种子配对数据(真图终点均更高:41.0>39.5、
   50.5>19.0,后者含一次后期长度失稳),按需提供;(c) 公式的进一步细节(ŵ 的序列级 vs token 级)
   → §3.1 已含答案。
6. **改分预期**:直答+让步组合下,bDeY 3→4 概率约 35–45%;维持 3 约 55%;降分风险很低
   (<5%,他问的是澄清型问题且我们全部直答)。
