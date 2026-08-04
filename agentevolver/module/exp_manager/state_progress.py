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
# SciWorld task-type-aware stage progress (match_mode="sciworld_stage")
# ---------------------------------------------------------------------------

# Per-task-type priority tables: signal_category -> Phi(s) in [0, 1]
# Calibrated against empirical median temporal positions from 792 teacher trajectories.
SCIWORLD_TASK_PRIORITIES: Dict[str, Dict[str, float]] = {
    "conductivity": {
        "noop": 0.00, "nav_door": 0.05, "nav_arrive": 0.10, "nav_workshop": 0.15,
        "room_desc": 0.10, "kitchen_desc": 0.10, "workshop_desc": 0.25,
        "open_container": 0.15, "pickup_thermometer": 0.20, "focus_thermometer": 0.25,
        "pickup": 0.30, "focus": 0.35, "move_object": 0.40,
        "substance_examined": 0.40, "place_apparatus": 0.40,
        "deactivate": 0.50, "substance_state": 0.50,
        "connect": 0.60, "activate": 0.70, "temperature_reading": 0.75,
        "wait": 0.80, "read_recipe": 0.10, "mix_result": 0.10,
        "final_placement": 1.00,
    },
    "temperature_measurement": {
        "noop": 0.00, "nav_door": 0.05, "nav_arrive": 0.10, "nav_workshop": 0.10,
        "room_desc": 0.10, "kitchen_desc": 0.15, "workshop_desc": 0.10,
        "open_container": 0.15, "pickup_thermometer": 0.25, "focus_thermometer": 0.35,
        "pickup": 0.40, "focus": 0.50, "move_object": 0.55,
        "substance_examined": 0.60, "substance_state": 0.60,
        "place_apparatus": 0.55, "activate": 0.10, "deactivate": 0.10,
        "connect": 0.10, "wait": 0.10,
        "temperature_reading": 0.80, "read_recipe": 0.10, "mix_result": 0.10,
        "final_placement": 1.00,
    },
    "phase_change": {
        "noop": 0.00, "nav_door": 0.03, "nav_arrive": 0.06, "nav_workshop": 0.06,
        "room_desc": 0.08, "kitchen_desc": 0.10, "workshop_desc": 0.08,
        "pickup_thermometer": 0.15, "pickup": 0.18, "open_container": 0.20,
        "focus_thermometer": 0.22, "focus": 0.25, "move_object": 0.30,
        "place_apparatus": 0.40, "activate": 0.50,
        "substance_examined": 0.55, "deactivate": 0.55,
        "temperature_reading": 0.60, "wait": 0.65,
        "substance_state": 0.75, "connect": 0.10,
        "read_recipe": 0.10, "mix_result": 0.10,
        "final_placement": 0.90,
    },
    "find_entity": {
        "noop": 0.00, "nav_door": 0.10, "nav_arrive": 0.15, "nav_workshop": 0.15,
        "room_desc": 0.20, "kitchen_desc": 0.20, "workshop_desc": 0.20,
        "open_container": 0.20, "pickup_thermometer": 0.10, "focus_thermometer": 0.10,
        "substance_examined": 0.30, "substance_state": 0.30,
        "focus": 0.45, "pickup": 0.55, "move_object": 0.65,
        "place_apparatus": 0.10, "activate": 0.10, "deactivate": 0.10,
        "connect": 0.10, "wait": 0.10, "temperature_reading": 0.10,
        "read_recipe": 0.10, "mix_result": 0.10,
        "final_placement": 1.00,
    },
    "life_stages": {
        "noop": 0.00, "nav_door": 0.15, "nav_arrive": 0.30, "nav_workshop": 0.30,
        "room_desc": 0.35, "kitchen_desc": 0.35, "workshop_desc": 0.35,
        "open_container": 0.10, "pickup_thermometer": 0.10, "focus_thermometer": 0.10,
        "pickup": 0.10, "move_object": 0.10, "substance_examined": 0.10,
        "substance_state": 0.10, "place_apparatus": 0.10, "activate": 0.10,
        "deactivate": 0.10, "connect": 0.10, "wait": 0.10,
        "temperature_reading": 0.10, "read_recipe": 0.10, "mix_result": 0.10,
        "focus": 1.00,
        "final_placement": 1.00,
    },
    "chemistry": {
        "noop": 0.00, "nav_door": 0.05, "nav_arrive": 0.10, "nav_workshop": 0.10,
        "room_desc": 0.10, "kitchen_desc": 0.15, "workshop_desc": 0.10,
        "open_container": 0.20, "pickup_thermometer": 0.10, "focus_thermometer": 0.10,
        "pickup": 0.25, "move_object": 0.45,
        "read_recipe": 0.35, "substance_examined": 0.50, "substance_state": 0.60,
        "place_apparatus": 0.40, "activate": 0.10, "deactivate": 0.10,
        "connect": 0.10, "wait": 0.10, "temperature_reading": 0.10,
        "mix_result": 0.85, "focus": 0.90,
        "final_placement": 1.00,
    },
    "circuit": {
        "noop": 0.00, "nav_door": 0.05, "nav_arrive": 0.10, "nav_workshop": 0.15,
        "room_desc": 0.10, "kitchen_desc": 0.10, "workshop_desc": 0.20,
        "open_container": 0.10, "pickup_thermometer": 0.10, "focus_thermometer": 0.10,
        "pickup": 0.20, "focus": 0.30, "move_object": 0.25,
        "substance_examined": 0.10, "substance_state": 0.10,
        "place_apparatus": 0.10, "activate": 0.80, "deactivate": 0.10,
        "connect": 0.65, "wait": 0.90, "temperature_reading": 0.10,
        "read_recipe": 0.10, "mix_result": 0.10,
        "final_placement": 1.00,
    },
    "generic": {
        "noop": 0.00, "nav_door": 0.05, "nav_arrive": 0.10, "nav_workshop": 0.10,
        "room_desc": 0.10, "kitchen_desc": 0.10, "workshop_desc": 0.10,
        "pickup": 0.20, "pickup_thermometer": 0.20, "open_container": 0.22,
        "focus": 0.30, "focus_thermometer": 0.30, "move_object": 0.35,
        "read_recipe": 0.40, "place_apparatus": 0.45,
        "deactivate": 0.50, "activate": 0.55, "substance_examined": 0.55,
        "connect": 0.60, "temperature_reading": 0.65,
        "substance_state": 0.70, "wait": 0.75, "mix_result": 0.85,
        "final_placement": 1.00,
    },
}


def detect_sciworld_task_type(task_desc: str) -> str:
    """Detect SciWorld task type from the task description string.

    Covers all 11 task types in the 792-trajectory teacher dataset.
    Returns one of: conductivity, temperature_measurement, phase_change,
    find_entity, life_stages, chemistry, circuit, generic.
    """
    td = task_desc.lower()
    if "electrically conductive" in td:
        return "conductivity"
    if re.search(r"measure.*temperature", td):
        return "temperature_measurement"
    if "melting point" in td:
        return "phase_change"
    if re.search(r"\bmelt\b", td):
        return "phase_change"
    if re.search(r"\bboil\b", td):
        return "phase_change"
    if re.search(r"\bfreeze\b", td):
        return "phase_change"
    if "change" in td and "state of matter" in td:
        return "phase_change"
    if re.search(r"find a\(n\)", td):
        return "find_entity"
    if "find a living" in td or "find a non-living" in td or "find a plant" in td:
        return "find_entity"
    if "life span" in td or "life stage" in td or "longest life" in td:
        return "life_stages"
    if "chemistry" in td or "recipe" in td or "create the substance" in td:
        return "chemistry"
    if "turn on" in td:
        return "circuit"
    return "generic"


def classify_sciworld_obs_signal(obs_text: str) -> str:
    """Classify a SciWorld observation into one of 24 semantic signal categories.

    Pattern matching order: more specific patterns first, then general ones.
    Returns 'noop' for errors, invalid actions, and unrecognized observations.
    """
    obs = obs_text.strip()
    if not obs:
        return "noop"
    obs_lower = obs.lower()
    first_line = obs.split("\n")[0].lower()

    # --- Room description (MUST be checked early — long text may contain
    #     temperature readings, substance names, etc. from listed objects) ---
    if "this room is called" in first_line or "this outside location" in first_line:
        if "workshop" in first_line:
            return "workshop_desc"
        if "kitchen" in first_line:
            return "kitchen_desc"
        return "room_desc"

    # --- Completion: placing object in colored answer box ---
    if re.search(
        r"you move the .* to the (red|blue|green|orange|purple|yellow) box",
        obs_lower,
    ):
        return "final_placement"
    if "disconnecting" in first_line and "box" in obs_lower:
        return "final_placement"

    # --- Chemistry ---
    if "mix to produce" in obs_lower:
        return "mix_result"
    if "recipe reads" in obs_lower or "the recipe" in obs_lower:
        return "read_recipe"

    # --- Experiment monitoring ---
    if "you decide to wait" in first_line:
        return "wait"
    if re.match(r"^(solid|liquid|gas)\s", first_line):
        return "substance_state"
    # Temperature reading: check first_line to avoid false positives from
    # room descriptions that list objects with temperature info
    if re.search(r"\d+\s*degrees\s*celsius", first_line):
        return "temperature_reading"

    # --- Circuit ---
    if "connected to" in first_line or "is now connected" in first_line:
        return "connect"

    # --- Apparatus activation ---
    if "is now activated" in first_line:
        return "activate"
    if "is now deactivated" in first_line:
        return "deactivate"

    # --- Substance examination ---
    if re.match(r"^a substance called ", first_line):
        return "substance_examined"

    # --- Experiment setup: place on apparatus ---
    if re.search(
        r"you move the .* to the (stove|blast furnace|freezer|sink|oven|bunsen burner)",
        obs_lower,
    ):
        return "place_apparatus"

    # --- Focus ---
    if "you focus on" in first_line:
        if "thermometer" in first_line:
            return "focus_thermometer"
        return "focus"

    # --- Container operations ---
    if re.search(
        r"(cupboard|fridge|freezer|blast furnace|oven|closet) is now open",
        first_line,
    ):
        return "open_container"

    # --- Pickup to inventory ---
    if "you move the" in first_line and "to the inventory" in obs_lower:
        if "thermometer" in first_line:
            return "pickup_thermometer"
        return "pickup"

    # --- Move object (generic) ---
    if "you move the" in first_line or "you move a" in first_line:
        return "move_object"

    # --- Navigation ---
    if "you move to" in first_line:
        if "workshop" in first_line:
            return "nav_workshop"
        return "nav_arrive"
    if "the door is already open" in first_line:
        return "nav_door"
    if "is now open" in first_line:
        return "nav_door"

    # --- Pour action ---
    if "you pour" in first_line:
        return "move_object"

    # --- Fallback: check full text for temperature reading (e.g., multi-line
    #     thermometer output where first line is the object name) ---
    if re.search(r"\d+\s*degrees\s*celsius", obs_lower):
        return "temperature_reading"

    # --- Everything else: errors, invalid format, "already" states, etc. ---
    return "noop"


def sciworld_stage_potential(obs: str, task_type: str = "generic") -> float:
    """Return Phi(s) for a SciWorld observation using task-type-aware stage classification."""
    signal = classify_sciworld_obs_signal(obs)
    table = SCIWORLD_TASK_PRIORITIES.get(task_type, SCIWORLD_TASK_PRIORITIES["generic"])
    return table.get(signal, 0.0)


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
        match_dropout: float = 0.0,
        soft_sim_threshold: float = 0.5,
        obs_noise_p: float = 0.0,
        shuffle_progress: bool = False,
    ):
        self.env_type = env_type
        self.match_mode = match_mode
        # Rebuttal diagnostics (NeurIPS 2026), all default-off = paper behavior:
        # - match_dropout: deterministically remove a fraction of teacher state
        #   keys at build time (md5-hashed so all ranks build identical maps).
        # - obs_noise_p: word-level dropout applied to the observation seen by
        #   the MATCHER only (policy input untouched), simulating noisy/partial
        #   observations where exact matching breaks. Deterministic per string.
        # - match_mode "soft": TF-IDF cosine similarity matching over the same
        #   teacher state map (exact-hit fast path first); returns the progress
        #   of the best match above soft_sim_threshold, else 0.
        # - shuffle_progress: permute the progress values among a task's own state
        #   keys. Hit rate, the value distribution, and hence the magnitude of the
        #   shaping bonus are all preserved; only the correspondence between a state
        #   and how far along it is gets destroyed. This is the matched-magnitude
        #   control asked for by reviewer y9x6 ("is the gain from the teacher-derived
        #   progress map, or just from a dense bonus of this size?").
        self.match_dropout = float(match_dropout or 0.0)
        self.soft_sim_threshold = float(soft_sim_threshold or 0.5)
        self.obs_noise_p = float(obs_noise_p or 0.0)
        self.shuffle_progress = bool(shuffle_progress)
        # task_id -> [(token->tfidf_weight, vector_norm, progress)]
        self._soft_profiles: Dict[str, list] = {}
        # task_id -> (token->idf, default_idf_for_unseen_tokens)
        self._soft_idf: Dict[str, tuple] = {}
        # value >= 0: matched progress; value < 0: cached miss
        self._soft_cache: Dict[Tuple[str, str], float] = {}
        self._soft_last_matched: bool = False
        # task_id -> {normalized_obs_string -> progress_float}
        self.progress_maps: Dict[str, Dict[str, float]] = {}
        # For stage mode: just track which task_ids have teacher data
        self._task_ids: set = set()
        # For sciworld_stage mode: task_id -> task_type string
        self._task_type_map: Dict[str, str] = {}

        if match_mode in ("stage", "attribute_aware"):
            self._task_ids = set(teacher_task2trajectories.keys())
            logger.info(
                f"[State Channel] {match_mode} mode: "
                f"{len(self._task_ids)} tasks registered "
                f"(env_type={env_type})"
            )
            return

        if match_mode == "sciworld_stage":
            self._task_ids = set(teacher_task2trajectories.keys())
            type_counts: Dict[str, int] = {}
            for task_id, trajectories in teacher_task2trajectories.items():
                # Extract task description from the first user message (step[2])
                task_desc = ""
                for traj in trajectories:
                    steps = traj.steps if hasattr(traj, "steps") else []
                    if len(steps) >= 3:
                        task_desc = steps[2].get("content", "")
                        break
                task_type = detect_sciworld_task_type(task_desc)
                self._task_type_map[task_id] = task_type
                type_counts[task_type] = type_counts.get(task_type, 0) + 1
            logger.info(
                f"[State Channel] sciworld_stage mode: "
                f"{len(self._task_ids)} tasks registered. "
                f"Task types: {type_counts}"
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

            if self.match_dropout > 0.0 and progress_map:
                import hashlib
                thresh = int(self.match_dropout * 10000)
                progress_map = {
                    obs: p for obs, p in progress_map.items()
                    if int(hashlib.md5(f"scdrop|{task_id}|{obs}".encode()).hexdigest(), 16) % 10000 >= thresh
                }

            if self.shuffle_progress and len(progress_map) > 1:
                import hashlib
                import random
                # Seeded per task so every rank builds the identical permutation.
                seed = int(hashlib.md5(f"scshuf|{task_id}".encode()).hexdigest()[:8], 16)
                keys = sorted(progress_map)
                vals = [progress_map[k] for k in keys]
                random.Random(seed).shuffle(vals)
                progress_map = dict(zip(keys, vals))

            if progress_map:
                self.progress_maps[task_id] = progress_map
                total_tasks += 1

        total_keys = sum(len(m) for m in self.progress_maps.values())
        logger.info(
            f"[State Channel] Built ExpertProgressMap: "
            f"{total_tasks} tasks, {total_states} total expert observations, "
            f"{total_keys} unique state keys"
            + (f" (match_dropout={self.match_dropout:.2f} applied)" if self.match_dropout > 0 else "")
        )

        if self.match_mode == "soft":
            import math
            from collections import Counter
            for task_id, pmap in self.progress_maps.items():
                df: Dict[str, int] = {}
                toks_list = []
                for obs, prog in pmap.items():
                    toks = Counter(obs.split())
                    toks_list.append((toks, prog))
                    for tok in toks:
                        df[tok] = df.get(tok, 0) + 1
                n_docs = max(len(toks_list), 1)
                idf = {t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()}
                default_idf = math.log(n_docs + 1) + 1.0  # tokens never seen in this task
                profiles = []
                for toks, prog in toks_list:
                    weighted = {t: c * idf[t] for t, c in toks.items()}
                    norm = math.sqrt(sum(v * v for v in weighted.values()))
                    if norm > 0:
                        profiles.append((weighted, norm, prog))
                self._soft_profiles[task_id] = profiles
                self._soft_idf[task_id] = (idf, default_idf)
            logger.info(
                f"[State Channel] soft mode: TF-IDF profiles built for "
                f"{len(self._soft_profiles)} tasks "
                f"(threshold={self.soft_sim_threshold}, obs_noise_p={self.obs_noise_p})"
            )
        elif self.obs_noise_p > 0:
            logger.info(f"[State Channel] obs_noise_p={self.obs_noise_p} active (matcher-side only)")

    # ------------------------------------------------------------------
    # Core lookups
    # ------------------------------------------------------------------

    def register_task_type(self, task_id: str, task_desc: str) -> None:
        """Register the task type for an on-policy task not in teacher data.

        Only relevant for sciworld_stage mode. If the task_id is already
        registered, this is a no-op.
        """
        if self.match_mode != "sciworld_stage":
            return
        if task_id not in self._task_type_map:
            self._task_type_map[task_id] = detect_sciworld_task_type(task_desc)

    def has_task(self, task_id: str) -> bool:
        if self.match_mode in ("stage", "attribute_aware"):
            return task_id in self._task_ids
        if self.match_mode == "sciworld_stage":
            # sciworld_stage can classify any observation universally;
            # return True even for on-policy tasks not in teacher set
            return True
        return task_id in self.progress_maps

    def _apply_obs_noise(self, obs: str) -> str:
        """Word-level dropout on the matcher's view of the observation.

        Deterministic per input string (md5-seeded) so results are reproducible
        and identical across ranks. The policy never sees this corruption.
        """
        import hashlib
        import random
        seed = int(hashlib.md5(("obsnoise|" + obs).encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        kept = [w for w in obs.split() if rng.random() >= self.obs_noise_p]
        return " ".join(kept) if kept else obs

    def _soft_potential(self, task_id: str, observation: str) -> float:
        """Best TF-IDF cosine match against the task's teacher states.

        Sets `_soft_last_matched` so callers can distinguish "matched, progress
        happens to be 0.0" from "no candidate above threshold".
        """
        import math
        from collections import Counter
        self._soft_last_matched = False
        profiles = self._soft_profiles.get(task_id)
        if not profiles:
            return 0.0
        idf, default_idf = self._soft_idf.get(task_id, ({}, 1.0))
        q = Counter(observation.split())
        # True TF-IDF cosine: query weighted with the same per-task idf.
        # Tokens unseen in this task's teacher states get max idf — they cannot
        # match any candidate, so they only penalize similarity (novel states
        # stay below threshold).
        q_w = {t: c * idf.get(t, default_idf) for t, c in q.items()}
        q_norm = math.sqrt(sum(v * v for v in q_w.values()))
        if q_norm == 0:
            return 0.0
        best_sim, best_prog = 0.0, 0.0
        for weighted, norm, prog in profiles:
            dot = 0.0
            for t, w in q_w.items():
                cw = weighted.get(t)
                if cw is not None:
                    dot += w * cw
            sim = dot / (q_norm * norm)
            if sim > best_sim:
                best_sim, best_prog = sim, prog
        if best_sim >= self.soft_sim_threshold:
            self._soft_last_matched = True
            return best_prog
        self._soft_last_matched = False
        return 0.0

    def _lookup(self, task_id: str, observation: str) -> Optional[float]:
        """Map-based lookup. Returns None on a MISS.

        Progress 0.0 is a legitimate value (the first state of a trajectory), so
        callers must not use `> 0` as a hit test. Both get_potential() and the
        coverage diagnostics go through here, which keeps the reported coverage
        consistent with the matching operator actually in use (exact vs soft,
        with or without observation noise).
        """
        pmap = self.progress_maps.get(task_id)
        if pmap is None:
            return None
        if self.obs_noise_p > 0.0:
            observation = self._apply_obs_noise(observation)
        hit = pmap.get(observation)
        if hit is not None:
            return hit
        if self.match_mode != "soft":
            return None
        key = (task_id, observation)
        cached = self._soft_cache.get(key)
        if cached is not None:
            return None if cached < 0.0 else cached
        val = self._soft_potential(task_id, observation)  # 0.0 also means "no match"
        matched = self._soft_last_matched
        if len(self._soft_cache) > 500_000:
            self._soft_cache.clear()
        self._soft_cache[key] = val if matched else -1.0
        return val if matched else None

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
        if self.match_mode == "sciworld_stage":
            task_type = self._task_type_map.get(task_id, "generic")
            return sciworld_stage_potential(observation, task_type)
        val = self._lookup(task_id, observation)
        return 0.0 if val is None else val

    def compute_trajectory_progress(
        self, task_id: str, observations: List[str],
        agg_mode: str = "mean",
    ) -> float:
        """Aggregate per-step potentials into a single trajectory progress P(tau).

        Args:
            task_id: Task identifier for potential lookup.
            observations: Normalized observation strings.
            agg_mode: Aggregation mode.
                "mean"  -- P(tau) = (1/T) sum Phi(s_t)  (original)
                "max"   -- P(tau) = max_t Phi(s_t)
                "last"  -- P(tau) = Phi(s_T)  (last observation, i.e. current state)
        """
        if not observations:
            return 0.0

        if agg_mode == "last":
            return self.get_potential(task_id, observations[-1])
        elif agg_mode == "max":
            return max(self.get_potential(task_id, obs) for obs in observations)
        else:  # "mean" (default)
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
        if self.match_mode in ("stage", "attribute_aware", "sciworld_stage"):
            # Stage/attribute_aware/sciworld_stage mode: every observation is classified
            matched = len(observations) if self.has_task(task_id) else 0
        else:
            # Goes through the same lookup path as get_potential (honours the
            # active matching operator and any observation noise), and treats a
            # progress of 0.0 as a hit — the first state of a trajectory has
            # progress 0.0.
            matched = sum(1 for obs in observations
                          if self._lookup(task_id, obs) is not None)
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
        if self.match_mode == "sciworld_stage":
            return {
                "num_tasks": len(self._task_ids),
                "total_keys": 24,  # number of signal categories
                "match_mode": self.match_mode,
                "task_types": len(set(self._task_type_map.values())),
            }
        return {
            "num_tasks": len(self.progress_maps),
            "total_keys": sum(len(m) for m in self.progress_maps.values()),
        }
