**title / abstract 讲的是 experience replay，但 introduction 和 conclusion 大量使用 teacher replay**。这两个词不是完全冲突，但读者可能会觉得你的 framing 在摇摆：到底贡献是 “teacher-guided RL” 还是 “experience replay for LLM agents”？目前 abstract 第一行已经用了 “Experience replay from strong teacher agents”，但 intro 第 2 段开始基本都在说 “teacher replay”。

我建议把主叙述统一成：

> **teacher experience replay** 是本文研究对象；
> **teacher replay** 可以作为简写；
> **experience replay** 是更广的 RL lens；
> DUET 是一个 principled teacher-experience-replay framework。

这样 title、abstract、introduction、related work、method 都能对齐。

## Introduction 里建议调整的地方

### 1. 第二段开头要从 “teacher replay” 改成 “teacher experience replay”

现在是：

> A natural way to address this cold-start problem is to replay successful trajectories from a stronger teacher model.

这句话内容没错，但它没有把 “experience replay” 这个 title concept 立起来。建议改成：

```latex
A natural way to address this cold-start problem is \emph{teacher experience replay}: reusing successful interaction trajectories collected from a stronger teacher model.
These trajectories provide the student with high-reward experience that it is unlikely to discover early in training.
```

这样一开始就定义了 teacher experience replay，而且 “interaction trajectories” 比 “demonstrations” 更贴近你的 action/state dual-channel 设计。

---

### 2. 后面少用 “teacher demonstrations”，多用 “teacher trajectories / teacher experience”

现在 introduction 第 2 段说：

> LUFFY mixes teacher demonstrations into GRPO rollout groups...

这也可以，但 “demonstrations” 会把读者引向 imitation learning，而你的论文想强调的是 **experience replay**。建议改成：

```latex
LUFFY mixes teacher trajectories into GRPO rollout groups, while CHORD combines teacher-mixed RL with a weighted imitation objective.
```

或者更贴近 title：

```latex
LUFFY mixes teacher experience into GRPO rollout groups, while CHORD combines teacher-mixed RL with a weighted imitation objective.
```

我更推荐第一版：**teacher trajectories** 更具体，读起来也更自然。

---

### 3. “Most existing uses of teacher replay” 改成 “teacher experience replay”

现在是：

> Most existing uses of teacher replay, however, treat the teacher trajectory mainly as an action sequence...

建议改成：

```latex
Most existing uses of teacher experience replay, however, treat a teacher trajectory mainly as an action sequence: the teacher's outputs either enter the policy-gradient batch or serve as supervised targets.
```

顺便这里有 typo：PDF 里是 `the teachers outputs`，需要改成 `the teacher's outputs`。

---

### 4. 第三段可以更强调 “experience = action + state”

你现在这段已经在讲 state signal，但还可以更自然地和 experience replay 对齐：

当前：

> This view leaves part of the teacher trajectory unused.

建议改成：

```latex
This action-centric view underuses the ``experience'' in teacher experience replay.
In interactive tasks, a successful trajectory contains not only the teacher's outputs, but also the sequence of environment states that mark progress toward completion.
These state-level signals can guide exploration without requiring the student to imitate the teacher's exact actions, which is useful when multiple reasoning traces or action sequences can solve the same task.
```

我觉得这句很关键：**underuses the “experience” in teacher experience replay**。它能自然解释为什么你的 title 是 experience replay，而不是 imitation learning。

---

### 5. 第四段第一句保留 “teacher experience replay”，后面用 “direct teacher mixing”

现在是：

> At the same time, even teacher experience replay is not statistically benign...

这个方向对了，但可以更准确。问题不是 teacher experience replay 本身不 benign，而是 **naively mixing teacher experience into GRPO** 不 benign。建议改成：

```latex
At the same time, naive teacher experience replay is not statistically benign in GRPO-style training.
When teacher and student rollouts are normalized together, two problems arise.
```

另外这里有 typo：

> creating an distribution mismatch

应该改成：

```latex
creating an \textbf{uncorrected distribution mismatch}
```

或者：

```latex
creating a distribution mismatch
```

如果你前面 bold 了 baseline contamination，我建议这里用：

```latex
Second, teacher trajectories are off-policy, creating an \textbf{uncorrected distribution mismatch}.
```

这样和 contributions 对齐。

---

### 6. 方法段第一句改成 “teacher-experience-replay framework”

现在是：

> We propose DUET, a teacher-replay framework...

建议改成：

```latex
We propose DUET, a teacher-experience-replay framework that uses teacher trajectories as two complementary sources of signal.
```

或者稍微更顺：

```latex
We propose DUET, a framework for teacher experience replay that uses each teacher trajectory as two complementary sources of signal.
```

我更推荐第二句。它避免了复合词太长，也更自然。

---

### 7. 实验段里 “fixed trajectory experience” 要改

现在 PDF 里这句非常不自然：

> together with a fixed trajectory experience collected from a 72B teacher model

建议改成：

```latex
We evaluate DUET on ALFWorld and WebShop using Qwen2.5-1.5B/3B students and a fixed trajectory cache collected from a Qwen2.5-72B teacher.
```

如果你不想强调 Qwen：

```latex
We evaluate DUET on ALFWorld and WebShop using 1.5B and 3B students and a fixed trajectory cache collected from a 72B teacher.
```

同时这句：

> Through experiments, DUET achieves strong success rate compared to other baselines.

建议改成：

```latex
DUET achieves the highest success rate among all reproduced baselines in all four model--environment settings.
```

这更像论文表达，也更具体。

---

### 8. Contributions 里第 2 点改成 teacher experience replay

当前：

```latex
We propose DUET, a dual-channel teacher-replay framework...
```

建议：

```latex
\item We propose DUET, a dual-channel framework for teacher experience replay that uses teacher actions for corrected action-level learning and teacher states for progress-based reward shaping.
```

第 1 点也可以从 “direct teacher mixing” 改成 “naive teacher experience replay”：

```latex
\item We identify two failure modes of naive teacher experience replay in GRPO: baseline contamination from high-reward teacher rollouts, and uncorrected distribution mismatch when teacher likelihoods are unavailable.
```

这样 title, intro, contributions 都统一了。

## 我建议替换的 introduction 关键段落

下面这几段可以直接替换你 intro 的第 2 到第 6 段：

```latex
A natural way to address this cold-start problem is \emph{teacher experience replay}: reusing successful interaction trajectories collected from a stronger teacher model.
These trajectories provide the student with high-reward experience that it is unlikely to discover early in training.
Recent methods follow this idea with promising results.
LUFFY~\citep{yan2025luffy} mixes teacher trajectories into GRPO rollout groups, while CHORD~\citep{zhang2025chord} combines teacher-mixed RL with a weighted imitation objective.
These methods show that teacher data can substantially accelerate LLM-agent training.
Most existing uses of teacher experience replay, however, treat a teacher trajectory mainly as an action sequence: the teacher's outputs either enter the policy-gradient batch or serve as supervised targets.

This action-centric view underuses the ``experience'' in teacher experience replay.
In interactive tasks, a successful trajectory contains not only the teacher's outputs, but also the sequence of environment states that mark progress toward completion.
These state-level signals can guide exploration without requiring the student to imitate the teacher's exact actions, which is useful when multiple reasoning traces or action sequences can solve the same task.

At the same time, naive teacher experience replay is not statistically benign in GRPO-style training.
When teacher and student rollouts are normalized together, two problems arise.
First, successful teacher trajectories can inflate the group reward statistics, reducing the advantages assigned to the few successful on-policy rollouts (\textbf{baseline contamination}).
This directly weakens the exploratory successes that RL should amplify.
Second, teacher trajectories are off-policy, creating an \textbf{uncorrected distribution mismatch}.
In principle, their gradients should be corrected by a teacher--student importance ratio, but exact correction is often unavailable when the teacher is closed-source or its tokenizer does not align with the student's.
Naive replay can therefore keep teacher gradients influential even after the student no longer benefits from copying them.
Both problems are most severe in the cold-start regime, where teacher experience replay is most needed.

We propose DUET, a framework for teacher experience replay that uses each teacher trajectory as two complementary sources of signal.
The Action Channel determines how teacher outputs affect the policy update.
It separates teacher and on-policy baselines so that teacher rewards do not distort on-policy credit assignment, and uses a discriminator-based density-ratio weight to approximate the missing teacher--student correction when exact likelihood ratios are unavailable.
The same weight reduces teacher-action gradients as the student improves, so teacher influence is controlled by the estimated distribution gap rather than by a fixed training-step schedule.
In parallel, the State Channel uses teacher observations to build a progress map over environment states.
This map provides dense reward shaping for on-policy rollouts, guiding exploration toward teacher-like progress without requiring the student to copy teacher actions.
Because the State Channel shapes only on-policy rollouts while the Action Channel corrects teacher-action gradients, DUET can use both signals together without reintroducing the baseline contamination caused by naive teacher mixing.

We evaluate DUET on ALFWorld and WebShop using 1.5B and 3B students and a fixed trajectory cache collected from a 72B teacher.
DUET achieves the highest success rate among all reproduced baselines in all four model--environment settings.
The gains are largest for the weaker 1.5B students, where cold start is most severe: DUET improves over the strongest prior baseline by 17.5 percentage points on both ALFWorld and WebShop.
Ablations show that baseline separation is essential for stable training, while density-ratio correction and the State Channel provide complementary gains.
```

Contributions 对应改成：

```latex
Our contributions are:
\begin{enumerate}
    \item We identify two failure modes of naive teacher experience replay in GRPO: baseline contamination from high-reward teacher rollouts, and uncorrected distribution mismatch when teacher likelihoods are unavailable.

    \item We propose DUET, a dual-channel framework for teacher experience replay that uses teacher actions for corrected action-level learning and teacher states for progress-based reward shaping.

    \item We show that DUET improves LLM-agent RL across two interactive environments and two student model scales, with the largest gains in the cold-start regime.
\end{enumerate}
```

## Conclusion 也要同步

你现在 conclusion 第一词就是 “teacher replay”。建议改成：

```latex
We studied teacher experience replay for reinforcement learning of LLM agents under GRPO-style training.
Although teacher trajectories provide valuable cold-start experience, directly mixing them with on-policy rollouts can bias group-relative advantage estimation and leave teacher-action gradients uncorrected when teacher likelihoods are unavailable.
DUET addresses these issues with two complementary uses of teacher experience: corrected action-level replay through baseline separation and density-ratio weighting, and state-level guidance through progress-based reward shaping.
Experiments on ALFWorld and WebShop show that DUET improves over teacher-replay and imitation-based baselines across student model scales, especially when the student has little initial on-policy success.
We hope this perspective encourages future work to treat teacher trajectories as structured interaction experience, rather than only as action sequences to imitate.
```

这样 title 里的 **Experience Replay**、abstract 里的 **experience replay from strong teacher agents**、intro 里的 **teacher experience replay**、conclusion 里的 **structured interaction experience** 就统一起来了。
