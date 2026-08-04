"""Audit the environment service over a run's time window.

Answers "was the environment healthy while this run executed?" from the AgentGym access log,
which is what we could not do for historical runs because the log was overwritten on every
launch (the env scripts now rotate it into logs/env_archive/).

Only the WebShop AgentGym server writes the access log this parses; the ALFWorld server does not,
so `--env alfworld` will report no request lines. For ALFWorld, health is judged from the run log
and the trajectory record instead (see scripts/check_replicate_health.py).

Usage:
  python scripts/audit_env_health.py --start "2026-07-26 01:12" --end "2026-07-26 03:41"
  python scripts/audit_env_health.py --log logs/webshop_agentgym.log
"""
import argparse
import glob
import re
from collections import Counter
from datetime import datetime

LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ - [\d.]+ - (\w+) (\S+) - (\d+) - ([\d.]+) seconds")


def parse(paths, start=None, end=None):
    codes, routes, times, stamps = Counter(), Counter(), [], []
    lifecycles = 0
    for p in paths:
        for line in open(p, errors="ignore"):
            if "Uvicorn running on" in line or "Started server process" in line:
                lifecycles += 1
            m = LINE.match(line)
            if not m:
                continue
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            stamps.append(ts)
            routes[m.group(3).split("?")[0]] += 1
            codes[m.group(4)] += 1
            times.append(float(m.group(5)))
    return codes, routes, times, stamps, lifecycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="append",
                    help="log file(s); default: the live log plus everything in logs/env_archive/")
    ap.add_argument("--env", default="webshop", choices=["webshop", "alfworld"])
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()

    paths = args.log or ([f"logs/{args.env}_agentgym.log"] +
                         sorted(glob.glob(f"logs/env_archive/{args.env}_agentgym_*.log")))
    paths = [p for p in paths if glob.glob(p)]
    if not paths:
        print("no environment logs found"); return

    fmt = lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M") if s else None
    codes, routes, times, stamps, lifecycles = parse(paths, fmt(args.start), fmt(args.end))
    if not stamps:
        print(f"no request lines in window; scanned {len(paths)} file(s)"); return

    stamps.sort()
    gaps = [(stamps[i + 1] - stamps[i]).total_seconds() for i in range(len(stamps) - 1)]
    bad = {c: n for c, n in codes.items() if not c.startswith("2")}

    print(f"files scanned      : {len(paths)}")
    print(f"window             : {stamps[0]} -> {stamps[-1]}")
    print(f"requests           : {sum(codes.values())}")
    print(f"status codes       : {dict(codes)}")
    print(f"non-2xx            : {bad if bad else 'none'}")
    print(f"server lifecycles  : {lifecycles}  ({'single' if lifecycles <= 2 else 'RESTARTED — investigate'})")
    print(f"largest gap        : {max(gaps):.0f}s" if gaps else "")
    print(f"slowest request    : {max(times):.1f}s   mean {sum(times)/len(times):.3f}s")
    print("route counts       :")
    for r, n in routes.most_common(10):
        print(f"    {r:28s} {n}")
    verdict = "HEALTHY" if not bad and lifecycles <= 2 else "NEEDS ATTENTION"
    print(f"\n=== {verdict}")


if __name__ == "__main__":
    main()
