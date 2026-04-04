import re

log_file = 'wandb/run-20260402_162739-j2rle81i/files/output.log'

metrics_by_step = {}
with open(log_file) as f:
    for line in f:
        line = line.strip()
        if not line.startswith('step:'):
            continue
        m = re.match(r'step:(\d+)', line)
        if not m: continue
        step = int(m.group(1))
        pairs = {}
        for kv in re.finditer(r'(\S+):(-?[\d.]+(?:e[+-]?\d+)?)', line):
            key, val = kv.group(1), float(kv.group(2))
            pairs[key] = val
        metrics_by_step[step] = pairs

print("Total steps: %d, range: %d-%d" % (len(metrics_by_step), min(metrics_by_step), max(metrics_by_step)))
print()

header = "%4s | %8s | %9s | %6s | %5s | %5s | %6s | %5s | %9s | %9s | %6s" % (
    "step", "disc_acc", "disc_loss", "w_mean", "w_std", "w_max", "w_clip", "clip%", "tch_share", "tch_scale", "reward")
print(header)
print("-" * len(header))

for step in range(1, 101):
    d = metrics_by_step.get(step, {})
    print("%4d | %8.4f | %9.4f | %6.4f | %5.4f | %5.4f | %6.4f | %5.3f | %9.4f | %9.4f | %6.4f" % (
        step,
        d.get("dr3/disc_acc", 0),
        d.get("dr3/disc_loss", 0),
        d.get("dr3/w_mean", 0),
        d.get("dr3/w_std", 0),
        d.get("dr3/w_max", 0),
        d.get("dr3/w_clip_upper", 0),
        d.get("dr3/w_clipfrac_off", 0),
        d.get("duet/teacher_gradient_share", 0),
        d.get("diag/teacher_loss_scale", 0),
        d.get("diag/reward_onpolicy_mean", 0),
    ))
