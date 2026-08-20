#!/bin/bash
# =============================================================================
# P0 队列运行器(2026-08-10):单槽串行跑完整个网格,**崩溃自动续跑**。
#
# 由来:CATALYST 主跑在 step 76 因 AlfWorld env_service 500 崩溃,守卫只记录
# 不重启,GPU 空转 10.5 小时。训练侧 resume_mode=auto + 每 50 步 checkpoint,
# 重投即可从断点续跑,没有理由人工守着。
#
# 用法:nohup bash scripts/p0_queue_runner.sh &  (队列写在下面 QUEUE 数组)
# 日志:$SCRATCH/logs/p0_queue.log(每次提交/退出/续跑都有一行)
# 停止:touch $SCRATCH/logs/P0_QUEUE_STOP(在两个作业之间生效)
# =============================================================================
set -u
REPO=/home/qisheng001/DUET_H200/EvolAnalsis
SCRATCH=/projects_vol/gp_wangwy/qisheng/duet_h200
LOG=$SCRATCH/logs/p0_queue.log
STOP=$SCRATCH/logs/P0_QUEUE_STOP
TRAIN=$SCRATCH/logs/p0_q35af.train.log
C=config/duet_paper_experiments_configs/iclr2027/train_h200
MAX_RETRY=6          # 单个格子最多续跑次数(每次都从最近 checkpoint 继续)
# 无进展熔断:连续 N 次尝试 checkpoint 步数不前进就停下报警,而不是把剩余
# 重试次数全烧在同一个必然失败的格子上。2 次足够区分"偶发崩溃"与"每次都在
# 同一处死"——resume_mode=disable 那次就是连续从 0 重来却看不出来。
MAX_NO_PROGRESS=2
# 与 run_train_p0.pbs 的默认吞吐 lane 一致;校验器据此核对 yaml。
LANE_MNS="${P0_MNS:-16}"
LANE_GMU="${P0_GMU:-0.35}"
cd "$REPO" || exit 1

# 队列:name|conf|target_steps
# 2026-08-15:CATALYST v3(机制修复:分布课程+课程critic+学生状态池)
# vs GRPO 新协议(save_freq 10 + gate 0.025,与 v3 逐项对齐)。
# seed s1/s2 待 s0 对照落地后追加;CHORD/LUFFY 再之后。
# 2026-08-17:v4 研发期的无悔基线(GRPO10 补 seed;方法无关,可复用)。
QUEUE=(
  "p0_catalystv4_af_s3|$C/alfworld_qwen35_4b_catalystv4_s3_grid100.yaml|100"
)

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

# 从 checkpoint 目录读已完成步数(resume 的真实进度来源)
done_steps() {
    local name=$1 latest
    latest=$(ls -d "$REPO"/checkpoints/*/"$name"/global_step_* 2>/dev/null \
             | sed 's/.*global_step_//' | sort -n | tail -1)
    echo "${latest:-0}"
}

# 提交前配置体检(L3 resume_mode / L4 数据路径 / L8 save_freq / lane 一致性 /
# experiment_name 与 yaml 是否同名)。不合格就拒绝提交并记日志 —— 一次
# resume_mode=disable 让整个 100 步格子永远跑不完,那类问题必须在烧卡之前拦住。

# 校验器只需要 pyyaml。队列通常由 nohup 从登录 shell 拉起,PATH 上不一定有
# python,所以显式挑一个能 import yaml 的解释器;一个都挑不到就说清楚,
# 而不是把"跑不了校验器"误判成"配置不合格"。
pick_python() {
    local cand
    for cand in "${DUET_QUEUE_PYTHON:-}" \
                "/projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/duet/bin/python" \
                "$(command -v python3 2>/dev/null)" \
                "$(command -v python 2>/dev/null)"; do
        [ -n "$cand" ] && [ -x "$cand" ] || continue
        "$cand" -c "import yaml" >/dev/null 2>&1 && { echo "$cand"; return 0; }
    done
    return 1
}
PY=$(pick_python) || { log "FATAL: no python with pyyaml — cannot run the config validator"; exit 1; }
log "validator python: $PY"

validate_cell() {
    local name=$1 conf=$2 out rc
    out=$("$PY" scripts/validate_job_config.py "$conf" \
            --lane-mns "$LANE_MNS" --lane-gmu "$LANE_GMU" \
            --experiment-name "$name" 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep -E '^CHECK (ERROR|WARN)' | while IFS= read -r line; do
        log "$name preflight: $line"
    done
    return $rc
}

log "==== queue runner start (${#QUEUE[@]} cells) ===="
for entry in "${QUEUE[@]}"; do
    IFS='|' read -r NAME CONF TARGET <<< "$entry"

    if ! validate_cell "$NAME" "$CONF"; then
        log "$NAME REFUSED: config validation failed — not submitting (fix the yaml, then rerun the queue)"
        continue
    fi
    log "$NAME config validated ok"

    no_progress=0
    for attempt in $(seq 0 $MAX_RETRY); do
        [ -f "$STOP" ] && { log "STOPFILE present — runner exits"; exit 0; }
        have=$(done_steps "$NAME")
        if [ "$have" -ge "$TARGET" ]; then
            log "$NAME already at step $have >= $TARGET — skip"
            break
        fi
        JOB=$(qsub -v P0_NAME="$NAME",P0_CONF="$CONF" "$REPO/run_train_p0.pbs" 2>&1)
        case "$JOB" in
            *.gaas) ;;
            *) log "$NAME qsub FAILED: $JOB"; sleep 300; continue ;;
        esac
        JOBID=${JOB%%.*}
        log "$NAME attempt=$attempt job=$JOBID (from step $have, target $TARGET)"
        # 等作业结束;预防性换血(L2/N3):单次尝试推进 ≥ CAP 步即主动轮换,
        # 换新 env 栈再续跑 —— v1/v3/GRPO10 全死在 step~76 的 mmap 天花板,
        # 且死前 ~10 步喂的是垂死环境的变质数据(v3 的 60→70 跳水主因之一)。
        # CAP=50 与 save_freq=10 对齐,轮换发生在 checkpoint 边界,损失 ≤1 步。
        ATTEMPT_STEP_CAP="${ATTEMPT_STEP_CAP:-50}"
        # N8:qstat 失败 ≠ 作业消失 —— 调度器前端宕机时 qstat 也非零退出,
        # 误判会导致对仍在跑的作业重复提交(2026-08-18 实遇,qsub 同挂才未酿祸)。
        # 服务器不可达时保守假定作业仍在跑,只记日志等待。
        while :; do
            qout=$(qstat "$JOBID" 2>&1); qrc=$?
            if [ $qrc -ne 0 ]; then
                if echo "$qout" | grep -qi "cannot connect\|Connection refused\|Communication failure\|End of File"; then
                    log "$NAME job=$JOBID: PBS server unreachable — assuming job alive, waiting"
                else
                    break   # 作业真的不在了
                fi
            fi
            sleep 120
            cur=$(done_steps "$NAME")
            if [ "$((cur - have))" -ge "$ATTEMPT_STEP_CAP" ] && [ "$cur" -lt "$TARGET" ]; then
                log "$NAME job=$JOBID ROTATE at step $cur (cap=$ATTEMPT_STEP_CAP, fresh env stack on resume)"
                qdel "$JOBID" 2>/dev/null || true
            fi
        done
        sleep 30
        after=$(done_steps "$NAME")
        laststep=$(grep -oE "step:[0-9]+ - " "$TRAIN" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
        log "$NAME job=$JOBID ended; checkpoint_step=$after last_log_step=${laststep:-?}"
        if [ "$after" -ge "$TARGET" ]; then
            log "$NAME COMPLETE at step $after"
            break
        fi

        # 无进展熔断。"续跑"只有在 checkpoint 步数真的往前走时才叫续跑;
        # 步数不动说明每次都死在同一处(env 栈起不来、resume 没生效、盘满……),
        # 继续重投只是把队列槽位烧掉,还会一次次覆盖 validation_log。
        if [ "$after" -le "$have" ]; then
            no_progress=$((no_progress + 1))
            log "$NAME NO PROGRESS ($no_progress/$MAX_NO_PROGRESS): step stayed at $after across job $JOBID"
            if [ "$no_progress" -ge "$MAX_NO_PROGRESS" ]; then
                log "$NAME ALARM: $MAX_NO_PROGRESS consecutive attempts made no checkpoint progress (step $after < target $TARGET)."
                log "$NAME ALARM: refusing to burn more queue slots. Check $TRAIN and $SCRATCH/logs/p0_q35af.live.log;"
                log "$NAME ALARM: usual causes — env stack fails to start, resume not taking effect (L3), disk full, node preflight abort (L1)."
                break
            fi
        else
            no_progress=0
        fi

        if [ "$attempt" -eq "$MAX_RETRY" ]; then
            log "$NAME giving up after $MAX_RETRY retries (step $after)"
        else
            log "$NAME incomplete (step $after) — auto-resuming"
        fi
    done
done
log "==== queue runner done ===="
