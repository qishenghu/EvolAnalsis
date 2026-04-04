"""
DUET State Channel: Expert Progress-Based Reward Shaping

This module implements the State Channel of the DUET framework. It constructs
a progress function Φ(s) from expert trajectory state sequences, computes
trajectory-level progress P(τ) = mean(Φ(s_t)), and provides shaped reward
R'(τ) = R(τ) + β·P(τ) to break reward sparsity in GRPO.

Key classes:
    ExpertProgressMap: Builds and caches per-task hash maps from teacher
        trajectory observations to progress values in [0, 1].

Key functions:
    normalize_observation: Strip dynamic suffixes (AVAILABLE ACTIONS, etc.)
    extract_observations_from_steps: Extract obs from Trajectory.steps
    extract_observations_from_batch_messages: Extract obs from batch data
"""

import re
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ---------------------------------------------------------------------------
# WebShop stage-based progress (used when match_mode="stage")
# ---------------------------------------------------------------------------

WEBSHOP_STAGE_PROGRESS = {
    "search_home": 0.0,
    "search_results": 0.2,
    "product_detail": 0.5,
    "purchase_complete": 1.0,
}

# Attribute-aware stage values (product_detail base is lower; quality fills the gap)
WEBSHOP_ATTR_AWARE_STAGE = {
    "search_home": 0.0,
    "search_results": 0.15,
    "product_detail": 0.35,
    "purchase_complete": 1.0,
}

# Known option type headers on WebShop product detail pages
_WEBSHOP_OPTION_TYPES = frozenset({
    "color", "size", "fit type", "style", "flavor name", "item shape",
    "material type", "pattern", "closure type",
})

# Sentinel keywords that end the options section on a product detail page
_WEBSHOP_OPTIONS_END = frozenset({
    "description", "features", "reviews", "buy now",
})


def classify_webshop_page(obs: str) -> str:
    """Classify a WebShop observation into a discrete page type.

    The classification relies on structural markers in the [SEP]-delimited
    observation format returned by the WebShop environment.
    """
    if not obs:
        return "search_home"

    # Purchase confirmation page
    if obs.startswith("Thank you for shopping"):
        return "purchase_complete"

    # Product detail page (has navigation + Buy Now)
    if "< Prev" in obs and ("Buy Now" in obs or "Description" in obs):
        return "product_detail"

    # Search results listing
    first_sep = obs.split(" [SEP] ", 1)[0]
    if first_sep.startswith("Instruction:") or first_sep == "Instruction:":
        if "Page " in obs and "Back to Search" in obs:
            return "search_results"

    # Homepage / search page
    if obs.startswith("WebShop"):
        return "search_home"

    # Fallback: treat as search home (e.g., invalid action messages)
    return "search_home"


def webshop_stage_potential(obs: str) -> float:
    """Return Phi(s) for a WebShop observation using stage classification."""
    return WEBSHOP_STAGE_PROGRESS.get(classify_webshop_page(obs), 0.0)


# ---------------------------------------------------------------------------
# Attribute-aware progress for WebShop (match_mode="attribute_aware")
# ---------------------------------------------------------------------------

def extract_instruction_attributes(obs: str) -> dict:
    """Extract target attributes (color, size, fit_type, price) from the
    instruction text embedded in a WebShop observation.

    The instruction always appears as:
        Instruction: [SEP] <text> [SEP] ...
    Attributes are at the tail of the instruction text in the form:
        "with color: <val>, and size: <val>, and price lower than <val> dollars"
    """
    attrs: dict = {}

    # Locate instruction text
    m = re.search(r'Instruction:\s*\[SEP\]\s*(.*?)\s*\[SEP\]', obs)
    if not m:
        # Fallback: instruction might be in a different format
        m = re.search(r'Find me\s+(.*?)(?:\[SEP\]|$)', obs)
    if not m:
        return attrs

    inst = m.group(1)

    # color: <value>  (greedy up to comma-and or end)
    cm = re.search(r'color:\s*(.+?)(?:,\s*and\s|$)', inst)
    if cm:
        attrs['color'] = cm.group(1).strip().lower()

    # size: <value>
    sm = re.search(r'(?<!\w)size:\s*(.+?)(?:,\s*and\s|$)', inst)
    if sm:
        attrs['size'] = sm.group(1).strip().lower()

    # fit type: <value>
    fm = re.search(r'fit type:\s*(.+?)(?:,\s*and\s|$)', inst)
    if fm:
        attrs['fit_type'] = fm.group(1).strip().lower()

    # price lower than <value> dollars
    pm = re.search(r'price lower than\s*([\d.]+)', inst)
    if pm:
        try:
            attrs['price_upper'] = float(pm.group(1))
        except ValueError:
            pass

    return attrs


def parse_product_detail_options(obs: str) -> Optional[dict]:
    """Parse available options and price from a WebShop product detail page.

    Returns dict with:
        'options': {option_type: [value, ...], ...}
        'price': float or None
    Or None if the observation is not a product detail page.
    """
    if "< Prev" not in obs:
        return None

    # Split by [SEP]
    parts = [p.strip() for p in obs.split("[SEP]")]

    # Find the index of "< Prev" — options start after it
    try:
        prev_idx = next(i for i, p in enumerate(parts) if p == "< Prev")
    except StopIteration:
        return None

    options: Dict[str, List[str]] = {}
    current_opt: Optional[str] = None
    price: Optional[float] = None

    for p in parts[prev_idx + 1:]:
        pl = p.strip().lower()

        # Check if this is a known option type header
        if pl in _WEBSHOP_OPTION_TYPES:
            current_opt = pl
            options[current_opt] = []
            continue

        # Check if this is a sentinel that ends the options section
        if pl in _WEBSHOP_OPTIONS_END:
            current_opt = None
            continue

        # Parse price
        if p.strip().startswith("Price:"):
            current_opt = None
            # Handle range prices like "$21.49 to $24.99" — take the lower
            pm = re.search(r'\$([\d.]+)', p)
            if pm:
                try:
                    price = float(pm.group(1))
                except ValueError:
                    pass
            continue

        # Rating line
        if p.strip().startswith("Rating:"):
            current_opt = None
            continue

        # If we're collecting values for an option type, add this value
        if current_opt is not None:
            options[current_opt].append(pl)

    return {'options': options, 'price': price}


def compute_attribute_match_score(obs: str) -> float:
    """Compute quality score in [0, 0.50] for a product detail page.

    Checks whether the target attributes from the instruction (color, size,
    fit_type, price) are available/satisfied on the current product page.
    Returns 0.50 * (n_matches / n_checks).  If no attributes are checkable,
    returns 0.15 (mild default bonus for reaching a product page).
    """
    attrs = extract_instruction_attributes(obs)
    product = parse_product_detail_options(obs)

    if product is None:
        return 0.0

    n_checks = 0
    n_matches = 0

    # Color check — if instruction requires color, always count it
    if 'color' in attrs:
        n_checks += 1
        if 'color' in product['options'] and attrs['color'] in product['options']['color']:
            n_matches += 1

    # Size check
    if 'size' in attrs:
        n_checks += 1
        if 'size' in product['options'] and attrs['size'] in product['options']['size']:
            n_matches += 1

    # Fit type check
    if 'fit_type' in attrs:
        n_checks += 1
        if 'fit type' in product['options'] and attrs['fit_type'] in product['options']['fit type']:
            n_matches += 1

    # Price check
    if 'price_upper' in attrs:
        n_checks += 1
        if product['price'] is not None and product['price'] <= attrs['price_upper']:
            n_matches += 1

    if n_checks == 0:
        return 0.15  # No checkable attributes; mild bonus

    return 0.50 * (n_matches / n_checks)


def webshop_attribute_aware_potential(obs: str) -> float:
    """Phi(s) = stage_progress + quality_score for WebShop.

    Combines coarse stage advancement with fine-grained attribute matching
    on product detail pages.  The quality component directly correlates with
    the WebShop reward function (attribute_match * price_satisfaction).
    """
    page_type = classify_webshop_page(obs)
    base = WEBSHOP_ATTR_AWARE_STAGE.get(page_type, 0.0)

    if page_type == "product_detail":
        base += compute_attribute_match_score(obs)

    return min(base, 1.0)


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------

def normalize_observation(obs_text: str, env_type: str = "alfworld") -> str:
    """
    Strip dynamic suffixes from environment observations so the same
    underlying state always hashes to the same key.

    ALFWorld appends "\\nAVAILABLE ACTIONS: ..." after every observation.
    WebShop appends "\\n\\nYou can use: ..." and "\\nClickable elements: ...".
    SciWorld appends "\\nPossible actions: ...".
    """
    if not obs_text:
        return ""

    text = obs_text.strip()

    if env_type == "alfworld":
        idx = text.find("\nAVAILABLE ACTIONS:")
        if idx != -1:
            text = text[:idx]
    elif env_type == "webshop":
        patterns = [
            r'\n\nYou can use:.*$',
            r'\n\nClickable elements:.*$',
            r'\nClickable elements:.*$',
            r'\nAvailable actions:.*$',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.DOTALL)
    elif env_type == "sciworld":
        patterns = [
            r'\nPossible actions:.*$',
            r'\nAvailable actions:.*$',
        ]
        for pat in patterns:
            text = re.sub(pat, '', text, flags=re.DOTALL)

    return text.strip()


# ---------------------------------------------------------------------------
# Observation extraction helpers
# ---------------------------------------------------------------------------

def extract_observations_from_steps(
    steps: List[dict],
    env_type: str = "alfworld",
    skip_initial: int = 3,
) -> List[str]:
    """
    Extract normalized environment observations from a Trajectory's .steps list.

    Teacher trajectories loaded via exp_manager have .steps as List[dict]
    (originally from the "messages" field in JSONL). The first *skip_initial*
    messages are typically: system prompt, assistant ack, initial user query.
    Subsequent role=="user" messages are environment observations.
    """
    observations = []
    for i, msg in enumerate(steps):
        if i < skip_initial:
            continue
        if msg.get("role") == "user":
            normalized = normalize_observation(msg.get("content", ""), env_type)
            if normalized:
                observations.append(normalized)
    return observations


def extract_observations_from_batch_messages(
    msg_data,
    env_type: str = "alfworld",
    skip_initial: int = 3,
) -> List[str]:
    """
    Extract normalized observations from the batch non_tensor_batch["messages"]
    format.  Each element is {"messages": [{"role": ..., "content": ...}, ...]}.
    """
    if msg_data is None:
        return []

    # Unwrap the {"messages": [...]} wrapper
    if isinstance(msg_data, dict) and "messages" in msg_data:
        msg_list = msg_data["messages"]
    elif isinstance(msg_data, (list, tuple)):
        msg_list = msg_data
    else:
        return []

    return extract_observations_from_steps(msg_list, env_type, skip_initial)


# ---------------------------------------------------------------------------
# ExpertProgressMap
# ---------------------------------------------------------------------------

class ExpertProgressMap:
    """
    Builds a per-task hash map from normalized environment observations to
    progress values in [0, 1].  Constructed once from teacher trajectories
    and queried during training to compute trajectory progress P(τ).

    For each task, the map records the *maximum* progress index at which
    each observation appears across all expert trajectories for that task.
    """

    def __init__(
        self,
        teacher_task2trajectories: Dict[str, list],
        env_type: str = "alfworld",
        match_mode: str = "hash",
    ):
        self.env_type = env_type
        self.match_mode = match_mode
        # task_id -> {normalized_obs_string -> progress_float}
        self.progress_maps: Dict[str, Dict[str, float]] = {}
        # For stage mode: just track which task_ids have teacher data
        self._task_ids: set = set()

        if match_mode in ("stage", "attribute_aware"):
            self._task_ids = set(teacher_task2trajectories.keys())
            logger.info(
                f"[State Channel] {match_mode} mode: "
                f"{len(self._task_ids)} tasks registered "
                f"(env_type={env_type})"
            )
            return

        total_states = 0
        total_tasks = 0

        for task_id, trajectories in teacher_task2trajectories.items():
            progress_map: Dict[str, float] = {}
            for traj in trajectories:
                steps = traj.steps if hasattr(traj, 'steps') else []
                if not steps:
                    continue
                obs_list = extract_observations_from_steps(steps, env_type)
                T = len(obs_list)
                if T == 0:
                    continue
                for j, obs in enumerate(obs_list):
                    # j/(T-1) so the last observation = 1.0
                    progress = j / max(T - 1, 1)
                    progress_map[obs] = max(progress_map.get(obs, 0.0), progress)
                total_states += T

            if progress_map:
                self.progress_maps[task_id] = progress_map
                total_tasks += 1

        total_keys = sum(len(m) for m in self.progress_maps.values())
        logger.info(
            f"[State Channel] Built ExpertProgressMap: "
            f"{total_tasks} tasks, {total_states} total expert observations, "
            f"{total_keys} unique state keys"
        )

    # ------------------------------------------------------------------
    # Core lookups
    # ------------------------------------------------------------------

    def has_task(self, task_id: str) -> bool:
        if self.match_mode in ("stage", "attribute_aware"):
            return task_id in self._task_ids
        return task_id in self.progress_maps

    def get_potential(self, task_id: str, observation: str) -> float:
        """Φ(s): return the state progress value in [0, 1], or 0.0 if unmatched."""
        if self.match_mode == "stage":
            if task_id not in self._task_ids:
                return 0.0
            return webshop_stage_potential(observation)
        if self.match_mode == "attribute_aware":
            if task_id not in self._task_ids:
                return 0.0
            return webshop_attribute_aware_potential(observation)
        pmap = self.progress_maps.get(task_id)
        if pmap is None:
            return 0.0
        return pmap.get(observation, 0.0)

    def compute_trajectory_progress(
        self, task_id: str, observations: List[str]
    ) -> float:
        """P(τ) = (1/T) Σ Φ(s_t)  — average progress over the trajectory."""
        if not observations:
            return 0.0
        total = sum(self.get_potential(task_id, obs) for obs in observations)
        return total / len(observations)

    def compute_step_deltas(
        self, task_id: str, observations: List[str]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute per-step potentials and deltas.
        Returns:
            potentials: [Φ(s_0), Φ(s_1), ...]
            deltas:     [Φ(s_1)-Φ(s_0), Φ(s_2)-Φ(s_1), ...]  (len = len(obs)-1)
        """
        potentials = [self.get_potential(task_id, obs) for obs in observations]
        deltas = [potentials[t + 1] - potentials[t] for t in range(len(potentials) - 1)]
        return potentials, deltas

    def get_coverage_stats(
        self, task_id: str, observations: List[str]
    ) -> Dict[str, float]:
        """Coverage diagnostics for a single trajectory."""
        if not observations:
            return {"coverage": 0.0, "matched": 0, "total": 0,
                    "mean_potential": 0.0, "max_potential": 0.0}
        potentials = [self.get_potential(task_id, obs) for obs in observations]
        if self.match_mode in ("stage", "attribute_aware"):
            # Stage/attribute_aware mode: every observation is classified
            matched = len(observations) if self.has_task(task_id) else 0
        else:
            pmap = self.progress_maps.get(task_id, {})
            # Check key existence (not > 0) since the first state has progress=0.0
            matched = sum(1 for obs in observations if obs in pmap)
        return {
            "coverage": matched / len(observations),
            "matched": matched,
            "total": len(observations),
            "mean_potential": sum(potentials) / len(potentials),
            "max_potential": max(potentials),
        }

    def get_global_stats(self) -> Dict[str, int]:
        """Summary statistics for the whole map."""
        if self.match_mode in ("stage", "attribute_aware"):
            return {
                "num_tasks": len(self._task_ids),
                "total_keys": 4,  # number of page types
                "match_mode": self.match_mode,
            }
        return {
            "num_tasks": len(self.progress_maps),
            "total_keys": sum(len(m) for m in self.progress_maps.values()),
        }
