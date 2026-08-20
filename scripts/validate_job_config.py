#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-submit health check for DUET training configs (and for the job scripts).

This is the machine-readable half of ``docs/infra/LANDMINES.md``.  Every rule
here exists because a specific landmine cost real GPU hours; the docstring of
each rule names it.

Two modes:

  1. config check (default)::

        python scripts/validate_job_config.py <train.yaml> [--lane-mns 16] ...
        python scripts/validate_job_config.py --all      # every train_h200 yaml

  2. script audit (recurrence guard for the *next* new job script)::

        python scripts/validate_job_config.py --audit-scripts

Exit codes: 0 = clean (warnings allowed), 1 = at least one ERROR, 2 = usage /
load failure.  Nothing here writes to the repo; it only reads and reports.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - environment problem, not a rule
    print("CHECK FATAL: pyyaml is not importable in this interpreter", file=sys.stderr)
    sys.exit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent

# Ray's working_dir only packages the code tree.  A relative path that points
# inside one of these top-level dirs resolves in a worker; anything else (above
# all ``data/``) must be absolute or the worker raises FileNotFoundError.  (L4)
RAY_PACKAGED_DIRS = {"agentevolver", "config", "cookbook", "external", "runtime_files"}

TRAIN_CONFIG_GLOB = "config/duet_paper_experiments_configs/iclr2027/train_h200/*.yaml"

# Job scripts that must source the unified preamble.  Anything matching
# run_*.pbs is audited; these are exempt with a stated reason.
AUDIT_EXEMPT: Dict[str, str] = {}


# --------------------------------------------------------------------------- #
# report plumbing
# --------------------------------------------------------------------------- #
class Report:
    """Collects findings for one target and prints them in a stable format."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.errors: List[Tuple[str, str]] = []
        self.warnings: List[Tuple[str, str]] = []
        self.notes: List[Tuple[str, str]] = []

    def error(self, rule: str, msg: str) -> None:
        self.errors.append((rule, msg))

    def warn(self, rule: str, msg: str) -> None:
        self.warnings.append((rule, msg))

    def note(self, rule: str, msg: str) -> None:
        self.notes.append((rule, msg))

    @property
    def ok(self) -> bool:
        return not self.errors

    def emit(self) -> None:
        for rule, msg in self.notes:
            print(f"CHECK NOTE  [{rule}] {self.target}: {msg}")
        for rule, msg in self.warnings:
            print(f"CHECK WARN  [{rule}] {self.target}: {msg}")
        for rule, msg in self.errors:
            print(f"CHECK ERROR [{rule}] {self.target}: {msg}")
        verdict = "PASS" if self.ok else "FAIL"
        print(
            f"CHECK {verdict}  {self.target} "
            f"({len(self.errors)} error, {len(self.warnings)} warn, {len(self.notes)} note)"
        )


# --------------------------------------------------------------------------- #
# yaml loading + minimal hydra-style interpolation
# --------------------------------------------------------------------------- #
_INTERP = re.compile(r"\$\{([A-Za-z0-9_.]+)\}")


def _dig(root: Any, dotted: str) -> Any:
    cur = root
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def resolve(value: Any, root: Dict[str, Any], depth: int = 0) -> Any:
    """Expand ``${a.b}`` references against the same document (best effort)."""
    if depth > 8 or not isinstance(value, str):
        return value

    def _sub(m: "re.Match[str]") -> str:
        got = _dig(root, m.group(1))
        return m.group(0) if got is None else str(got)

    out = _INTERP.sub(_sub, value)
    return resolve(out, root, depth + 1) if out != value else out


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #
def rule_resume_mode(cfg: Dict[str, Any], rep: Report, allow_disable: bool) -> None:
    """L3 — resume_mode: disable silently restarts a 'resumed' run from step 0.

    The catalyst grid100 cell inherited ``disable`` from its smoke ancestor, so
    every queue retry re-ran step 0 and overwrote the checkpoint and the
    validation_log.  A 100-step cell could never finish and its history was
    destroyed on each attempt.  Long runs must be ``auto``.
    """
    mode = _dig(cfg, "trainer.resume_mode")
    if mode is None:
        rep.error("resume_mode", "trainer.resume_mode is absent — set it to 'auto'")
        return
    if mode == "auto":
        return
    if allow_disable:
        rep.note(
            "resume_mode",
            f"resume_mode={mode!r} accepted via --allow-disable "
            "(caller archives stale outputs before every start)",
        )
        return
    rep.error(
        "resume_mode",
        f"trainer.resume_mode={mode!r}; a requeue would restart at step 0 and "
        "overwrite checkpoints/validation_log. Use 'auto', or pass "
        "--allow-disable if this config is a throwaway smoke run whose caller "
        "archives stale output dirs first.",
    )


def _iter_path_fields(cfg: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """Yield (dotted_key, value) for every data-ish path in the config."""
    stack: List[Tuple[str, Any]] = [("", cfg)]
    wanted_leaf = {"data_path", "file", "hints_file", "task_file", "train_files", "val_files"}
    while stack:
        prefix, node = stack.pop()
        if isinstance(node, dict):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (dict, list)):
                    stack.append((key, v))
                elif isinstance(v, str) and k in wanted_leaf:
                    yield key, v
        elif isinstance(node, list):
            for i, v in enumerate(node):
                key = f"{prefix}[{i}]"
                if isinstance(v, (dict, list)):
                    stack.append((key, v))


def rule_data_paths(cfg: Dict[str, Any], rep: Report) -> None:
    """L4 — Ray's working_dir only packages the code tree.

    ``data/`` and ``scripts/`` are NOT in the Ray package, so a relative path
    there resolves on the submitting host and raises FileNotFoundError inside
    the worker.  Relative paths are legal only when their first component is a
    packaged dir; everything else must be absolute AND exist right now.
    """
    checked = 0
    for key, raw in _iter_path_fields(cfg):
        value = resolve(raw, cfg)
        if not value or value in ("null", "None"):
            continue
        # Only judge things that look like real data artefacts.
        if not re.search(r"\.(jsonl?|pkl|txt|npz|parquet)$", value):
            continue
        checked += 1
        p = Path(value)
        if not value.startswith("/"):
            first = p.parts[0] if p.parts else ""
            if first not in RAY_PACKAGED_DIRS:
                rep.error(
                    "data_path",
                    f"{key}={value!r} is a relative path outside the Ray package "
                    f"({sorted(RAY_PACKAGED_DIRS)}); the worker will FileNotFoundError. "
                    "Make it absolute.",
                )
                continue
            if not (REPO_ROOT / p).is_file():
                rep.error("data_path", f"{key}={value!r} is packaged but missing on disk")
            continue
        if not p.is_file():
            rep.error("data_path", f"{key}={value!r} does not exist")
    if checked == 0:
        rep.note("data_path", "no data_path/hints file fields to check")


def _ckpt_dirs(cfg: Dict[str, Any], name_override: Optional[str] = None) -> List[Path]:
    """Where this experiment's checkpoints live (config-declared + conventional)."""
    yaml_name = resolve(_dig(cfg, "trainer.experiment_name"), cfg)
    name = name_override or yaml_name
    if not name:
        return []
    local_dir = resolve(_dig(cfg, "trainer.default_local_dir"), cfg) or ""
    if name_override and yaml_name and local_dir:
        local_dir = local_dir.replace(str(yaml_name), str(name_override))
    out: List[Path] = []
    if local_dir:
        out.append(Path(local_dir) if local_dir.startswith("/") else REPO_ROOT / local_dir)
    project = resolve(_dig(cfg, "trainer.project_name"), cfg) or "agentevolver"
    out.append(REPO_ROOT / "checkpoints" / str(project) / str(name))
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def _dir_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def ckpt_facts(cfg: Dict[str, Any], name_override: Optional[str] = None
               ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[Path]]:
    """(bytes_per_checkpoint, free_bytes, latest_step, dir) measured from disk.

    Only an *existing* checkpoint gives an honest size, so a first-ever run
    returns (None, ...) and the disk rules stay silent rather than guessing.
    """
    for d in _ckpt_dirs(cfg, name_override):
        if not d.exists():
            continue
        steps = sorted(
            (int(p.name.split("_")[-1]), p)
            for p in d.glob("global_step_*")
            if p.name.split("_")[-1].isdigit()
        )
        free = None
        try:
            free = __import__("shutil").disk_usage(d).free
        except OSError:
            pass
        if not steps:
            return None, free, 0, d
        return _dir_bytes(steps[-1][1]), free, steps[-1][0], d
    return None, None, None, None


def _gb(n: Optional[int]) -> str:
    return "?" if n is None else f"{n / 2**30:.0f}GB"


def rule_ckpt_disk(cfg: Dict[str, Any], rep: Report, name_override: Optional[str]) -> None:
    """NEW (2026-08-12) — the L8 cure has a disk ceiling nobody costed.

    ``max_actor_ckpt_to_keep`` defaults to null (keep everything) and one
    Qwen3.5-4B FSDP checkpoint is ~50GB.  save_freq=10 on a 100-step cell is
    ~500GB; the project volume had 188GB free when this rule was written.  So
    "just lower save_freq" can convert a recoverable crash into an unrecoverable
    disk-full.  Report the arithmetic instead of pretending it is free.
    """
    per, free, latest, where = ckpt_facts(cfg, name_override)
    if per is None or free is None or where is None:
        return
    save_freq = _dig(cfg, "trainer.save_freq")
    total = _dig(cfg, "trainer.total_training_steps")
    try:
        sf, tt, done = int(save_freq), int(total), int(latest or 0)
    except (TypeError, ValueError):
        return
    if sf <= 0 or tt <= done:
        return
    keep = _dig(cfg, "trainer.max_actor_ckpt_to_keep")
    remaining = (tt - done) // sf
    if keep:
        rep.note("ckpt_disk",
                 f"max_actor_ckpt_to_keep={keep} bounds the footprint at ~{_gb(per * int(keep))}")
        return
    need = per * remaining
    msg = (f"{remaining} more checkpoint(s) x ~{_gb(per)} = ~{_gb(need)} to reach step {tt} "
           f"from {done}; {_gb(free)} free on {where}. max_actor_ckpt_to_keep is unset "
           f"(keep everything).")
    if need > free * 0.9:
        rep.error("ckpt_disk", msg + " This run WILL hit disk-full before it finishes — "
                                     "bound max_actor_ckpt_to_keep, or merge+prune older steps first.")
    elif need > free * 0.5:
        rep.warn("ckpt_disk", msg + " Over half the free space; prune before submitting.")
    else:
        rep.note("ckpt_disk", msg)


def rule_save_freq(cfg: Dict[str, Any], rep: Report, allow_sparse: bool,
                   name_override: Optional[str] = None) -> None:
    """L8 — save_freq 50 made every crash cost 20+ recomputed steps.

    Graded on purpose:
      * save_freq > 10 AND save_freq < total_training_steps  -> ERROR.  The run
        intends to checkpoint mid-flight but does it so rarely that a crash
        throws away most of the progress since the last save.
      * save_freq > 10 AND save_freq >= total_training_steps -> WARN.  The run
        only ever saves at the end; fine for a 6-step throughput trial, not for
        anything you would want to resume.

    Downgraded to WARN when complying is physically impossible: if save_freq=10
    would need more disk than the volume has and retention is unbounded, the
    honest fix is bounded retention, not a rule violation nobody can satisfy.
    """
    save_freq = _dig(cfg, "trainer.save_freq")
    total = _dig(cfg, "trainer.total_training_steps")
    if save_freq is None:
        rep.error("save_freq", "trainer.save_freq is absent")
        return
    try:
        save_freq = int(save_freq)
    except (TypeError, ValueError):
        rep.error("save_freq", f"trainer.save_freq={save_freq!r} is not an integer")
        return
    if save_freq <= 0:
        rep.error("save_freq", f"trainer.save_freq={save_freq} disables checkpointing entirely")
        return
    if save_freq <= 10:
        return
    total_i: Optional[int]
    try:
        total_i = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_i = None

    if total_i is not None and save_freq >= total_i:
        rep.warn(
            "save_freq",
            f"save_freq={save_freq} >= total_training_steps={total_i}: this run only "
            "checkpoints at the very end, so a crash loses everything. Acceptable "
            "for a short throughput trial only.",
        )
        return
    msg = (
        f"save_freq={save_freq} with total_training_steps={total_i}: a crash throws "
        f"away up to {save_freq - 1} steps. Use <= 10 (L8)."
    )
    if allow_sparse:
        rep.note("save_freq", msg + "  [accepted via --allow-sparse-ckpt]")
        return

    # Is compliance even possible on this filesystem?
    per, free, latest, where = ckpt_facts(cfg, name_override)
    keep = _dig(cfg, "trainer.max_actor_ckpt_to_keep")
    if per and free and total_i is not None and not keep:
        at_ten = per * ((total_i - int(latest or 0)) // 10)
        if at_ten > free * 0.9:
            rep.warn(
                "save_freq",
                msg + f" NOT enforceable here: save_freq=10 would write ~{_gb(at_ten)} "
                      f"of checkpoints with only {_gb(free)} free on {where} and "
                      f"max_actor_ckpt_to_keep unset. Bound retention first, then lower save_freq.",
            )
            return
    rep.error("save_freq", msg)


def rule_lane(cfg: Dict[str, Any], rep: Report,
              lane_mns: Optional[int], lane_gmu: Optional[float]) -> None:
    """The rollout server lane and the trainer's view of it must agree.

    ``start_rollout_servers.sh`` is told MAX_NUM_SEQS / GPU_MEM_UTIL by the job
    script; the trainer reads its own copy from the yaml.  When they diverge the
    run is silently off-contract (different decode batching than the manifest
    claims), which invalidates any throughput or drift comparison.
    """
    rollout = _dig(cfg, "actor_rollout_ref.rollout") or {}
    if lane_mns is not None:
        got = rollout.get("max_num_seqs")
        if got is None:
            rep.error("lane", "actor_rollout_ref.rollout.max_num_seqs is absent")
        elif int(got) != int(lane_mns):
            rep.error(
                "lane",
                f"rollout.max_num_seqs={got} but the job script's lane is MNS={lane_mns}",
            )
    if lane_gmu is not None:
        got = rollout.get("gpu_memory_utilization")
        if got is None:
            rep.error("lane", "actor_rollout_ref.rollout.gpu_memory_utilization is absent")
        elif abs(float(got) - float(lane_gmu)) > 1e-9:
            rep.error(
                "lane",
                f"rollout.gpu_memory_utilization={got} but the job script's lane is GMU={lane_gmu}",
            )
    if lane_mns is None and lane_gmu is None:
        rep.note("lane", "no --lane-mns/--lane-gmu given; lane consistency not checked")


def rule_experiment_collision(cfg: Dict[str, Any], rep: Report,
                              name_override: Optional[str]) -> None:
    """An existing checkpoint dir means this submission resumes OR destroys it.

    With resume_mode=auto that is the intended behaviour and we just say so.
    With anything else the existing evidence is about to be overwritten, which
    is exactly how the catalyst grid100 history was lost.
    """
    yaml_name = resolve(_dig(cfg, "trainer.experiment_name"), cfg)
    name = name_override or yaml_name
    if not name:
        rep.error("experiment_name", "trainer.experiment_name is absent")
        return

    # NEW (2026-08-12) — job-script identity vs config identity.
    # run_train_p0.pbs takes P0_NAME for its logs/W&B tag but launches
    # `launcher.py --conf <yaml>` WITHOUT overriding trainer.experiment_name, so
    # the checkpoints land under the YAML's name.  scripts/p0_queue_runner.sh
    # then polls checkpoints/*/$P0_NAME/global_step_* for progress.  If the two
    # disagree the queue sees zero progress forever and burns every retry.
    if name_override and yaml_name and "OVERRIDE_ME" in str(yaml_name):
        rep.error(
            "experiment_name",
            f"yaml still holds the grid placeholder {yaml_name!r} while the job script "
            f"says {name_override!r}. The job script does NOT override "
            "trainer.experiment_name, so the run would train under the placeholder. "
            "Copy the template and substitute every OVERRIDE_ME occurrence first.",
        )
        return

    if name_override and yaml_name and str(name_override) != str(yaml_name) \
            and "OVERRIDE_ME" not in str(yaml_name):
        rep.error(
            "experiment_name",
            f"job script says experiment_name={name_override!r} but the yaml says "
            f"{yaml_name!r}. Checkpoints land under the YAML name, while the queue "
            "runner polls the job-script name — it would see zero progress and retry "
            "until it gives up. Make them equal.",
        )
        return

    if "OVERRIDE_ME" in str(name) and not name_override:
        rep.note(
            "experiment_name",
            f"experiment_name={name!r} is a grid template placeholder; "
            "pass --experiment-name <cell> to check the real cell",
        )
        return

    local_dir = resolve(_dig(cfg, "trainer.default_local_dir"), cfg) or ""
    if name_override and local_dir:
        local_dir = local_dir.replace(str(resolve(_dig(cfg, "trainer.experiment_name"), cfg)), name_override)
    candidates = []
    if local_dir:
        candidates.append(Path(local_dir) if local_dir.startswith("/") else REPO_ROOT / local_dir)
    project = resolve(_dig(cfg, "trainer.project_name"), cfg) or "agentevolver"
    candidates.append(REPO_ROOT / "checkpoints" / str(project) / str(name))

    existing = [c for c in candidates if c.exists()]
    if not existing:
        return

    steps = []
    for c in existing:
        steps += [int(d.name.split("_")[-1]) for d in c.glob("global_step_*") if d.name.split("_")[-1].isdigit()]
    latest = max(steps) if steps else 0
    mode = _dig(cfg, "trainer.resume_mode")
    where = existing[0]
    if mode == "auto":
        rep.note(
            "experiment_name",
            f"checkpoints already exist at {where} (latest global_step_{latest}); "
            "resume_mode=auto -> this submission RESUMES from there",
        )
    else:
        rep.error(
            "experiment_name",
            f"checkpoints already exist at {where} (latest global_step_{latest}) but "
            f"resume_mode={mode!r} -> this submission would OVERWRITE them and restart "
            "at step 0. Set resume_mode=auto, or archive/rename the experiment first.",
        )


def rule_validation(cfg: Dict[str, Any], rep: Report) -> None:
    """A run whose val settings are half-specified produces an unreadable curve.

    Requires: a test_freq that actually fires inside the run, a val batch size
    and a val task budget, a deterministic val decoder, and a validation_data_dir
    so the per-step val records survive the job.
    """
    trainer = cfg.get("trainer") or {}
    data = cfg.get("data") or {}
    rollout = _dig(cfg, "actor_rollout_ref.rollout") or {}

    test_freq = trainer.get("test_freq")
    total = trainer.get("total_training_steps")
    if test_freq is None:
        rep.error("val", "trainer.test_freq is absent")
    else:
        try:
            tf, tt = int(test_freq), int(total) if total is not None else None
            if tf <= 0:
                rep.warn("val", f"test_freq={tf} disables in-training validation")
            elif tt is not None and tf > tt:
                rep.warn(
                    "val",
                    f"test_freq={tf} > total_training_steps={tt}: validation never runs "
                    "in-flight (fine for a throughput trial, not for a result run)",
                )
        except (TypeError, ValueError):
            rep.error("val", f"trainer.test_freq={test_freq!r} is not an integer")

    if trainer.get("val_before_train") is None:
        rep.warn("val", "trainer.val_before_train is unset — no step-0 baseline point")
    if not trainer.get("validation_data_dir"):
        rep.error("val", "trainer.validation_data_dir is unset — per-step val records are lost")

    for key in ("val_batch_size", "max_val_tasks"):
        if data.get(key) in (None, 0):
            rep.error("val", f"data.{key} is unset/zero — the val slice is undefined")

    vk = rollout.get("val_kwargs")
    if not isinstance(vk, dict):
        rep.error("val", "actor_rollout_ref.rollout.val_kwargs is absent — val decoder undefined")
        return
    for key in ("n", "temperature", "top_p"):
        if key not in vk:
            rep.error("val", f"rollout.val_kwargs.{key} is unset — val decoder is not reproducible")
    if vk.get("do_sample") is True and float(vk.get("temperature", 0) or 0) == 0.0:
        rep.warn("val", "val_kwargs: do_sample=true with temperature=0 is contradictory")


# --------------------------------------------------------------------------- #
# driver: one config
# --------------------------------------------------------------------------- #
def check_config(path: Path, args: argparse.Namespace) -> Report:
    rel = str(path.relative_to(REPO_ROOT)) if path.is_absolute() and str(path).startswith(str(REPO_ROOT)) else str(path)
    rep = Report(rel)
    try:
        cfg = load_config(path)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the job script
        rep.error("load", f"cannot parse yaml: {exc}")
        return rep

    rule_resume_mode(cfg, rep, args.allow_disable)
    rule_data_paths(cfg, rep)
    rule_save_freq(cfg, rep, args.allow_sparse_ckpt, args.experiment_name)
    rule_lane(cfg, rep, args.lane_mns, args.lane_gmu)
    rule_experiment_collision(cfg, rep, args.experiment_name)
    rule_ckpt_disk(cfg, rep, args.experiment_name)
    rule_validation(cfg, rep)
    return rep


# --------------------------------------------------------------------------- #
# driver: script audit (recurrence guard)
# --------------------------------------------------------------------------- #
def audit_scripts() -> int:
    """Flag every run_*.pbs that does not source the unified preamble.

    This is the mechanism that makes the whole exercise stick: the next time
    somebody adds a job script by copying an old one, this reports it instead of
    waiting for the landmine to go off again.
    """
    scripts = sorted(REPO_ROOT.glob("run_*.pbs")) + sorted(REPO_ROOT.glob("scripts/run_*.pbs"))
    if not scripts:
        print("AUDIT: no run_*.pbs found")
        return 0

    bad: List[Tuple[str, List[str]]] = []
    print(f"{'script':<40} {'preamble':<9} {'gpu':<5} {'wandb':<6} {'notes'}")
    print("-" * 96)
    for s in scripts:
        text = s.read_text(encoding="utf-8", errors="replace")
        rel = str(s.relative_to(REPO_ROOT))
        sources_preamble = "duet_job_preamble.sh" in text
        has_gpu = "duet_preamble_gpu" in text or "pf_run_all" in text
        has_wandb = "duet_preamble_wandb" in text or "unset WANDB_API_KEY" in text

        notes: List[str] = []
        if rel in AUDIT_EXEMPT:
            notes.append(f"exempt: {AUDIT_EXEMPT[rel]}")
        else:
            if not sources_preamble:
                notes.append("does not source scripts/duet_job_preamble.sh")
            if not has_gpu:
                notes.append("no GPU-index guard (L1)")
            if not has_wandb:
                notes.append("no W&B credential scrub")
            # Cheap landmine greps that do not need the preamble.
            if "lsof -ti" in text and "-sTCP:LISTEN" not in text:
                notes.append("lsof without -sTCP:LISTEN (L9)")
            if notes:
                bad.append((rel, notes))

        print(
            f"{rel:<40} {'yes' if sources_preamble else 'NO':<9} "
            f"{'yes' if has_gpu else 'NO':<5} {'yes' if has_wandb else 'NO':<6} "
            f"{'; '.join(notes)}"
        )

    print("-" * 96)
    if bad:
        print(f"AUDIT FAIL: {len(bad)}/{len(scripts)} job script(s) miss the unified preamble:")
        for rel, notes in bad:
            print(f"  - {rel}: {'; '.join(notes)}")
        print("  fix: source \"$REPO/scripts/duet_job_preamble.sh\" and call the duet_preamble_* functions")
        print("  see docs/infra/LANDMINES.md")
        return 1
    print(f"AUDIT PASS: all {len(scripts)} job scripts source the unified preamble")
    return 0


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-submit health check for DUET training configs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("config", nargs="?", help="training yaml to check")
    ap.add_argument("--all", action="store_true",
                    help=f"check every yaml under {TRAIN_CONFIG_GLOB}")
    ap.add_argument("--audit-scripts", action="store_true",
                    help="scan run_*.pbs for the unified preamble instead of checking a config")
    ap.add_argument("--allow-disable", action="store_true",
                    help="accept resume_mode != auto (throwaway smoke runs whose caller archives outputs)")
    ap.add_argument("--allow-sparse-ckpt", action="store_true",
                    help="accept save_freq > 10 mid-run (only for deliberately unresumable runs)")
    ap.add_argument("--lane-mns", type=int, default=None,
                    help="MAX_NUM_SEQS the job script will give the rollout servers")
    ap.add_argument("--lane-gmu", type=float, default=None,
                    help="GPU_MEM_UTIL the job script will give the rollout servers")
    ap.add_argument("--experiment-name", default=None,
                    help="override trainer.experiment_name (grid cells set it via qsub -v)")
    args = ap.parse_args(argv)

    if args.audit_scripts:
        return audit_scripts()

    targets: List[Path]
    if args.all:
        targets = sorted(REPO_ROOT.glob(TRAIN_CONFIG_GLOB))
        if not targets:
            print(f"CHECK FATAL: no yaml matched {TRAIN_CONFIG_GLOB}", file=sys.stderr)
            return 2
    elif args.config:
        p = Path(args.config)
        if not p.is_absolute():
            p = (Path.cwd() / p) if (Path.cwd() / p).exists() else (REPO_ROOT / p)
        if not p.is_file():
            print(f"CHECK FATAL: no such config: {args.config}", file=sys.stderr)
            return 2
        targets = [p]
    else:
        ap.print_usage(sys.stderr)
        print("CHECK FATAL: give a config, --all, or --audit-scripts", file=sys.stderr)
        return 2

    reports = [check_config(t, args) for t in targets]
    for rep in reports:
        rep.emit()

    if len(reports) > 1:
        failed = [r for r in reports if not r.ok]
        print()
        print(f"SUMMARY: {len(reports) - len(failed)}/{len(reports)} configs pass")
        for r in failed:
            print(f"  FAIL {r.target}: " + "; ".join(f"[{k}] {m}" for k, m in r.errors))
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
