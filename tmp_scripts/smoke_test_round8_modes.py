"""
Round 8 Phase 1 preflight smoke test.

For each adaptive_mu mode (disc_acc, nll, ess_ratio, kl_lagrangian), extract the
exact branch code from het_actor.py and execute it against synthetic tensors +
mocked dr3_metrics. Assert:
  - mu is a Python float in [chord_mu_valley, chord_mu_peak]
  - no NaN/Inf
  - adaptive_metrics dict has correct keys
  - Multi-call evolution for kl_lagrangian (state mutates)

Usage:
    python3 tmp_scripts/smoke_test_round8_modes.py

Exit code 0 = all modes pass; non-zero = at least one mode failed.
"""
import math
import sys


# --------------------------------------------------------------------------- #
# Mock "self.config" — minimal OmegaConf-like shim supporting .get(key, default)
# --------------------------------------------------------------------------- #
class MockConfig(dict):
    def get(self, key, default=None):
        return super().get(key, default)


# --------------------------------------------------------------------------- #
# Mock host object carrying persistent EMA/anchor state (like HETDataParallelPPOActor)
# --------------------------------------------------------------------------- #
class Host:
    def __init__(self, cfg_dict):
        self.config = MockConfig(cfg_dict)


# --------------------------------------------------------------------------- #
# Branch: disc_acc (v39 family)
# --------------------------------------------------------------------------- #
def run_disc_acc(self, dr3_metrics, sft_loss_val, chord_mu_peak, chord_mu_valley):
    """Replicates lines 1767-1807 in het_actor.py."""
    _disc_acc_raw = float(dr3_metrics.get("dr3/disc_acc", 0.0)) if isinstance(dr3_metrics, dict) else 0.0
    _disc_ready = float(dr3_metrics.get("dr3/disc_trained_steps", 0.0)) if isinstance(dr3_metrics, dict) else 0.0
    _disc_acc_now = _disc_acc_raw if _disc_ready > 0 else 0.5
    d_ema_alpha = float(self.config.get("chord_mu_d_ema_alpha", 0.2))
    if not hasattr(self, "_disc_acc_ema"):
        self._disc_acc_ema = _disc_acc_now
    else:
        self._disc_acc_ema = (1 - d_ema_alpha) * self._disc_acc_ema + d_ema_alpha * _disc_acc_now
    d_floor = float(self.config.get("chord_mu_d_floor", 0.5))
    d_mapping = str(self.config.get("chord_mu_d_mapping", "linear")).lower().strip()
    if d_mapping == "sigmoid":
        d_k = float(self.config.get("chord_mu_d_sigmoid_k", 10.0))
        _gated = 1.0 / (1.0 + math.exp(d_k * (float(self._disc_acc_ema) - d_floor)))
    else:
        _scale = max(1e-6, 1.0 - d_floor)
        _gated = max(0.0, (1.0 - float(self._disc_acc_ema)) / _scale)
        _gated = min(1.0, _gated)
    mu = chord_mu_valley + (chord_mu_peak - chord_mu_valley) * _gated
    return mu, {
        "chord/mu_mode": 3.0,
        "chord/disc_acc_ema": float(self._disc_acc_ema),
        "chord/disc_acc_current": float(_disc_acc_now),
        "chord/disc_acc_raw": float(_disc_acc_raw),
        "chord/disc_ready": float(_disc_ready),
        "chord/mu_adaptive_gated": float(_gated),
        "chord/d_floor": d_floor,
    }


# --------------------------------------------------------------------------- #
# Branch: nll linear (v40b)
# --------------------------------------------------------------------------- #
def run_nll(self, dr3_metrics, sft_loss_val, chord_mu_peak, chord_mu_valley):
    _nll_now = float(sft_loss_val)
    nll_ema_alpha = float(self.config.get("chord_mu_nll_ema_alpha", 0.3))
    if not hasattr(self, "_nll_ema"):
        self._nll_ema = _nll_now
    else:
        self._nll_ema = (1 - nll_ema_alpha) * self._nll_ema + nll_ema_alpha * _nll_now

    nll_mapping = str(self.config.get("chord_mu_nll_mapping", "sigmoid")).lower().strip()
    if nll_mapping == "linear":
        nll_slope = float(self.config.get("chord_mu_nll_slope", 0.156))
        nll_intercept = float(self.config.get("chord_mu_nll_intercept", chord_mu_valley))
        mu_raw = nll_intercept + nll_slope * float(self._nll_ema)
        mu = max(chord_mu_valley, min(chord_mu_peak, mu_raw))
        _gated_nll = (mu - chord_mu_valley) / max(1e-6, (chord_mu_peak - chord_mu_valley))
    elif nll_mapping == "ratio":
        warm_n = int(self.config.get("chord_mu_nll_ratio_anchor_n", 3))
        if not hasattr(self, "_nll_anchor_buf"):
            self._nll_anchor_buf = []
            self._nll_anchor = None
        if self._nll_anchor is None and _nll_now > 0:
            self._nll_anchor_buf.append(_nll_now)
            if len(self._nll_anchor_buf) >= warm_n:
                self._nll_anchor = float(sum(self._nll_anchor_buf) / max(1, len(self._nll_anchor_buf)))
        _nll_anchor_cur = self._nll_anchor if self._nll_anchor is not None else max(_nll_now, 1e-3)
        ratio_pow = float(self.config.get("chord_mu_nll_ratio_pow", 1.0))
        _gated_nll = max(0.0, min(1.0, (float(self._nll_ema) / max(1e-6, _nll_anchor_cur)) ** ratio_pow))
        mu = chord_mu_valley + (chord_mu_peak - chord_mu_valley) * _gated_nll
    else:
        nll_target = float(self.config.get("chord_mu_nll_target", 0.65))
        nll_k = float(self.config.get("chord_mu_nll_k", 6.0))
        _gated_nll = 1.0 / (1.0 + math.exp(-nll_k * (self._nll_ema - nll_target)))
        mu = chord_mu_valley + (chord_mu_peak - chord_mu_valley) * _gated_nll
    return mu, {
        "chord/mu_mode": 4.0,
        "chord/nll_ema": float(self._nll_ema),
        "chord/nll_current": float(_nll_now),
        "chord/mu_adaptive_gated": float(_gated_nll),
        "chord/nll_mapping": 0.0 if nll_mapping == "sigmoid" else (1.0 if nll_mapping == "linear" else 2.0),
    }


# --------------------------------------------------------------------------- #
# Branch: ess_ratio (v41b saturating)
# --------------------------------------------------------------------------- #
def run_ess_ratio(self, dr3_metrics, sft_loss_val, chord_mu_peak, chord_mu_valley):
    _ess_now = float(dr3_metrics.get("dr3/ess_off_window", 0.0)) if isinstance(dr3_metrics, dict) else 0.0
    _ess_len = float(dr3_metrics.get("dr3/ess_window_len", 0.0)) if isinstance(dr3_metrics, dict) else 0.0
    ess_ema_alpha = float(self.config.get("chord_mu_ess_ema_alpha", 0.2))
    if not hasattr(self, "_ess_ema"):
        self._ess_ema = _ess_now
    else:
        self._ess_ema = (1 - ess_ema_alpha) * self._ess_ema + ess_ema_alpha * _ess_now
    min_window = float(self.config.get("chord_mu_ess_anchor_min_window", 8.0))
    if not hasattr(self, "_ess_anchor"):
        self._ess_anchor = None
    if self._ess_anchor is None and _ess_len >= min_window and _ess_now > 0:
        self._ess_anchor = float(self._ess_ema)
    _ess_anchor_cur = float(self._ess_anchor) if self._ess_anchor is not None else max(_ess_now, 1.0)
    ratio = float(self._ess_ema) / max(1e-6, _ess_anchor_cur)
    ess_mapping = str(self.config.get("chord_mu_ess_mapping", "saturating")).lower().strip()
    if ess_mapping == "sigmoid":
        ess_tau = float(self.config.get("chord_mu_ess_tau", 0.5))
        ess_k = float(self.config.get("chord_mu_ess_sigmoid_k", 8.0))
        _gated = 1.0 / (1.0 + math.exp(ess_k * (ratio - ess_tau)))
    elif ess_mapping == "velocity":
        if not hasattr(self, "_ess_prev"):
            self._ess_prev = _ess_now
        vel = _ess_now - float(self._ess_prev)
        self._ess_prev = _ess_now
        if not hasattr(self, "_ess_vel_ema"):
            self._ess_vel_ema = vel
        else:
            self._ess_vel_ema = 0.8 * self._ess_vel_ema + 0.2 * vel
        vel_beta = float(self.config.get("chord_mu_ess_velocity_beta", 2.0))
        _gated = max(0.0, min(1.0, math.exp(-vel_beta * max(0.0, self._ess_vel_ema))))
    else:
        ess_pow = float(self.config.get("chord_mu_ess_saturating_pow", 0.5))
        _gated = max(0.0, 1.0 - max(0.0, ratio) ** ess_pow)
        _gated = min(1.0, _gated)
    mu = chord_mu_valley + (chord_mu_peak - chord_mu_valley) * _gated
    return mu, {
        "chord/mu_mode": 5.0,
        "chord/ess_current": float(_ess_now),
        "chord/ess_ema": float(self._ess_ema),
        "chord/ess_anchor": float(_ess_anchor_cur),
        "chord/ess_ratio": float(ratio),
        "chord/mu_adaptive_gated": float(_gated),
    }


# --------------------------------------------------------------------------- #
# Branch: kl_lagrangian (v43a)
# --------------------------------------------------------------------------- #
def run_kl_lagrangian(self, dr3_metrics, sft_loss_val, chord_mu_peak, chord_mu_valley):
    _cost_now = float(sft_loss_val)
    if not hasattr(self, "_kl_cost_ema"):
        self._kl_cost_ema = _cost_now
    else:
        kl_cost_alpha = float(self.config.get("chord_mu_kl_cost_ema_alpha", 0.3))
        self._kl_cost_ema = (1 - kl_cost_alpha) * self._kl_cost_ema + kl_cost_alpha * _cost_now
    kl_eps_fixed = self.config.get("chord_mu_kl_eps_fixed", None)
    if kl_eps_fixed is not None:
        _kl_budget = float(kl_eps_fixed)
    else:
        kl_rho = float(self.config.get("chord_mu_kl_budget_rho", 0.9))
        if not hasattr(self, "_kl_budget_ema"):
            self._kl_budget_ema = self._kl_cost_ema
        else:
            self._kl_budget_ema = kl_rho * self._kl_budget_ema + (1 - kl_rho) * self._kl_cost_ema
        _kl_budget = float(self._kl_budget_ema)
    kl_eta = float(self.config.get("chord_mu_kl_eta", 0.3))
    if not hasattr(self, "_mu_lagrange_state"):
        self._mu_lagrange_state = float(chord_mu_peak)
    _step_mult = math.exp(kl_eta * (self._kl_cost_ema - _kl_budget))
    _new_mu = float(self._mu_lagrange_state) * _step_mult
    _new_mu = max(chord_mu_valley, min(chord_mu_peak, _new_mu))
    self._mu_lagrange_state = _new_mu
    mu = _new_mu
    _gated = (mu - chord_mu_valley) / max(1e-6, (chord_mu_peak - chord_mu_valley))
    return mu, {
        "chord/mu_mode": 6.0,
        "chord/kl_cost_ema": float(self._kl_cost_ema),
        "chord/kl_budget": float(_kl_budget),
        "chord/kl_step_mult": float(_step_mult),
        "chord/mu_lagrange_state": float(self._mu_lagrange_state),
        "chord/mu_adaptive_gated": float(_gated),
    }


# --------------------------------------------------------------------------- #
# Test harness
# --------------------------------------------------------------------------- #
def _check(mu, metrics, mu_valley, mu_peak, expected_keys):
    assert isinstance(mu, float), f"mu is not Python float: type={type(mu)}"
    assert not math.isnan(mu), "mu is NaN"
    assert not math.isinf(mu), "mu is Inf"
    assert mu_valley - 1e-9 <= mu <= mu_peak + 1e-9, f"mu out of range: {mu} not in [{mu_valley}, {mu_peak}]"
    for k in expected_keys:
        assert k in metrics, f"Missing metric {k!r}"
        assert isinstance(metrics[k], float), f"Metric {k!r} is not float: type={type(metrics[k])}"
        assert not math.isnan(metrics[k]), f"Metric {k!r} is NaN"
        assert not math.isinf(metrics[k]), f"Metric {k!r} is Inf"


def test_disc_acc():
    print("\n[disc_acc] v39b: d_ema_alpha=0.5, d_floor=0.5")
    cfg = {"chord_mu_d_ema_alpha": 0.5, "chord_mu_d_floor": 0.5}
    host = Host(cfg)
    mu_peak, mu_valley = 0.3, 0.05
    expected = ["chord/mu_mode", "chord/disc_acc_ema", "chord/disc_acc_current",
                "chord/disc_acc_raw", "chord/disc_ready", "chord/mu_adaptive_gated", "chord/d_floor"]

    # Warmup: disc hasn't trained (disc_trained_steps=0, disc_acc=0.0)
    dr3 = {"dr3/disc_acc": 0.0, "dr3/disc_trained_steps": 0.0}
    mu, m = run_disc_acc(host, dr3, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    assert abs(mu - mu_peak) < 1e-6, f"Warmup should give mu=mu_peak, got {mu}"
    print(f"  warmup: mu={mu:.4f}, ema={m['chord/disc_acc_ema']:.4f} (OK, mu=peak)")

    # Disc trained, acc=0.5 (chance)
    dr3 = {"dr3/disc_acc": 0.5, "dr3/disc_trained_steps": 2.0}
    mu, m = run_disc_acc(host, dr3, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  acc=0.5:  mu={mu:.4f}, ema={m['chord/disc_acc_ema']:.4f}")

    # Disc trained, acc=0.9 (well-separated)
    for _ in range(10):
        dr3 = {"dr3/disc_acc": 0.9, "dr3/disc_trained_steps": 2.0}
        mu, m = run_disc_acc(host, dr3, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  acc=0.9 (x10): mu={mu:.4f}, ema={m['chord/disc_acc_ema']:.4f} (should approach mu_valley)")
    assert mu < mu_peak, "After many high-acc steps, mu should shrink below peak"

    # Edge: disc_acc=1.0
    dr3 = {"dr3/disc_acc": 1.0, "dr3/disc_trained_steps": 2.0}
    mu, m = run_disc_acc(host, dr3, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  acc=1.0:  mu={mu:.4f} (should be at or near mu_valley)")

    return True


def test_nll_linear():
    print("\n[nll/linear] v40b: slope=0.156, intercept=0.05")
    cfg = {"chord_mu_nll_mapping": "linear", "chord_mu_nll_slope": 0.156,
           "chord_mu_nll_intercept": 0.05, "chord_mu_nll_ema_alpha": 0.3}
    host = Host(cfg)
    mu_peak, mu_valley = 0.3, 0.05
    expected = ["chord/mu_mode", "chord/nll_ema", "chord/nll_current",
                "chord/mu_adaptive_gated", "chord/nll_mapping"]

    # NLL = 0 (policy perfect teacher mimic)
    mu, m = run_nll(host, {}, 0.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    assert abs(mu - 0.05) < 1e-6, f"NLL=0: expected mu=intercept=0.05, got {mu}"
    print(f"  nll=0.0: mu={mu:.4f} (expected 0.05=intercept)")

    # NLL = 1.0 (hard teacher) — linear would give 0.05 + 0.156*1 = 0.206
    for _ in range(20):  # drive EMA to 1.0
        mu, m = run_nll(host, {}, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    assert abs(mu - (0.05 + 0.156)) < 0.01, f"NLL=1.0: expected mu~=0.206, got {mu}"
    print(f"  nll=1.0 (x20): mu={mu:.4f} (expected ~0.206)")

    # NLL = 5.0 (huge) — linear would exceed peak → clamped to 0.3
    for _ in range(30):
        mu, m = run_nll(host, {}, 5.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    assert abs(mu - 0.3) < 1e-6, f"NLL=5.0 should clamp to mu_peak=0.3, got {mu}"
    print(f"  nll=5.0 (x30): mu={mu:.4f} (expected 0.3=clamped peak)")

    return True


def test_ess_ratio_saturating():
    print("\n[ess_ratio/saturating] v41b: pow=0.5, anchor_min_window=8")
    cfg = {"chord_mu_ess_mapping": "saturating", "chord_mu_ess_saturating_pow": 0.5,
           "chord_mu_ess_ema_alpha": 0.2, "chord_mu_ess_anchor_min_window": 8.0}
    host = Host(cfg)
    mu_peak, mu_valley = 0.3, 0.05
    expected = ["chord/mu_mode", "chord/ess_current", "chord/ess_ema",
                "chord/ess_anchor", "chord/ess_ratio", "chord/mu_adaptive_gated"]

    # Warmup: ess=5 but window not full → anchor not set
    dr3 = {"dr3/ess_off_window": 5.0, "dr3/ess_window_len": 3.0}
    mu, m = run_ess_ratio(host, dr3, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  warmup (len=3): mu={mu:.4f}, anchor={m['chord/ess_anchor']:.3f} (anchor should be raw ess_now)")

    # Window fills, ess=10 (high) → anchor captured at ~10
    for _ in range(5):
        dr3 = {"dr3/ess_off_window": 10.0, "dr3/ess_window_len": 16.0}
        mu, m = run_ess_ratio(host, dr3, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  anchored ess=10 (x5): mu={mu:.4f}, anchor={m['chord/ess_anchor']:.3f}")
    assert host._ess_anchor is not None, "Anchor should have been captured"

    # ESS decreases (policy diverges) — ratio < 1 → gated > 0 → mu rises
    for _ in range(10):
        dr3 = {"dr3/ess_off_window": 2.0, "dr3/ess_window_len": 16.0}
        mu, m = run_ess_ratio(host, dr3, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  ess drops to 2 (x10): mu={mu:.4f}, ratio={m['chord/ess_ratio']:.3f} (mu should rise)")
    assert mu > mu_valley, "mu should rise when ESS drops far below anchor"

    # ESS super-high → ratio >> 1 → gated = max(0, 1 - ratio^0.5) = 0 → mu=mu_valley
    for _ in range(20):
        dr3 = {"dr3/ess_off_window": 100.0, "dr3/ess_window_len": 16.0}
        mu, m = run_ess_ratio(host, dr3, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    assert abs(mu - mu_valley) < 1e-6, f"ess super-high: mu should be valley, got {mu}"
    print(f"  ess=100 (x20): mu={mu:.4f}, ratio={m['chord/ess_ratio']:.3f} (expected mu_valley)")

    return True


def test_kl_lagrangian():
    print("\n[kl_lagrangian] v43a: eta=0.3, rho=0.9")
    cfg = {"chord_mu_kl_eta": 0.3, "chord_mu_kl_budget_rho": 0.9,
           "chord_mu_kl_cost_ema_alpha": 0.3}
    host = Host(cfg)
    mu_peak, mu_valley = 0.3, 0.05
    expected = ["chord/mu_mode", "chord/kl_cost_ema", "chord/kl_budget",
                "chord/kl_step_mult", "chord/mu_lagrange_state", "chord/mu_adaptive_gated"]

    # First call: cost=1.0 → cost_ema=1.0, budget=1.0, mult=1.0, μ=peak
    mu, m = run_kl_lagrangian(host, {}, 1.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    assert abs(mu - mu_peak) < 1e-6, f"first call: mu should stay at peak, got {mu}"
    assert abs(m["chord/kl_step_mult"] - 1.0) < 1e-9, f"first step_mult should be 1.0"
    print(f"  call1 (cost=1.0): mu={mu:.4f}, mult={m['chord/kl_step_mult']:.4f}")

    # Cost rises sharply to 3.0 → cost_ema rises faster than budget → mult > 1 → μ grows
    # But it's already at peak, so it stays capped
    for i in range(10):
        mu, m = run_kl_lagrangian(host, {}, 3.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  cost=3.0 (x10): mu={mu:.4f}, mult={m['chord/kl_step_mult']:.4f}")

    # Cost drops to 0.5 → cost_ema falls fast, budget lags → mult < 1 → μ shrinks
    prev_mu = mu
    for i in range(30):
        mu, m = run_kl_lagrangian(host, {}, 0.5, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  cost=0.5 (x30): mu={mu:.4f}, mult={m['chord/kl_step_mult']:.4f}")
    assert mu <= prev_mu + 1e-6, f"mu should not grow when cost drops, prev={prev_mu}, now={mu}"
    # (may not reach floor; budget eventually catches up. but should be < peak)
    assert mu < mu_peak, f"after cost drop, mu should be below peak, got {mu}"

    # Confirm state evolves over calls
    assert host._mu_lagrange_state == mu, "state should match returned mu"

    # Edge: huge cost spike shouldn't overflow
    mu, m = run_kl_lagrangian(host, {}, 100.0, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  cost=100.0 spike: mu={mu:.4f}, mult={m['chord/kl_step_mult']:.2e}")
    assert mu <= mu_peak + 1e-6, "mu must be clamped to peak"

    # Edge: cost=0 ("perfect match to teacher") should shrink mu toward valley eventually
    for i in range(100):
        mu, m = run_kl_lagrangian(host, {}, 0.01, mu_peak, mu_valley)
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  cost=0.01 (x100): mu={mu:.4f}")

    return True


def test_ess_ratio_sigmoid():
    print("\n[ess_ratio/sigmoid] additional: check sigmoid branch")
    cfg = {"chord_mu_ess_mapping": "sigmoid", "chord_mu_ess_tau": 0.5,
           "chord_mu_ess_sigmoid_k": 8.0, "chord_mu_ess_ema_alpha": 0.2,
           "chord_mu_ess_anchor_min_window": 8.0}
    host = Host(cfg)
    mu_peak, mu_valley = 0.3, 0.05
    # fill anchor
    for _ in range(5):
        dr3 = {"dr3/ess_off_window": 10.0, "dr3/ess_window_len": 16.0}
        mu, m = run_ess_ratio(host, dr3, 1.0, mu_peak, mu_valley)
    # ratio << tau (ESS drop) → gated close to 1
    for _ in range(20):
        dr3 = {"dr3/ess_off_window": 1.0, "dr3/ess_window_len": 16.0}
        mu, m = run_ess_ratio(host, dr3, 1.0, mu_peak, mu_valley)
    expected = ["chord/mu_mode", "chord/ess_current", "chord/ess_ema",
                "chord/ess_anchor", "chord/ess_ratio", "chord/mu_adaptive_gated"]
    _check(mu, m, mu_valley, mu_peak, expected)
    print(f"  ess<<tau: mu={mu:.4f} (expected near peak)")
    return True


def main():
    failures = []
    tests = [("disc_acc", test_disc_acc),
             ("nll_linear", test_nll_linear),
             ("ess_ratio_saturating", test_ess_ratio_saturating),
             ("ess_ratio_sigmoid", test_ess_ratio_sigmoid),
             ("kl_lagrangian", test_kl_lagrangian)]
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            failures.append((name, str(e)))
            print(f"[FAIL] {name}: {e}")
        except Exception as e:
            failures.append((name, f"{type(e).__name__}: {e}"))
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {len(failures)}/{len(tests)}")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
        sys.exit(1)
    else:
        print(f"ALL {len(tests)} TESTS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
