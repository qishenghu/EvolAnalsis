"""
Qwen3.5 hybrid-thinking context management tests (work items A1-pre/A1-core/F2).

Verifies, with the patched /data/shared_models/Qwen3.5-2B-think tokenizer
(enable_thinking defaults ON):
  (i)   concat(token_arr) over a simulated multi-turn episode EXACTLY equals
        apply_chat_template(full message list) — history <think> stripped,
        final turn intact;
  (ii)  the loss mask blacks out generation-prompt / assistant headers;
  (iii) action extraction ignores '</action>' / '<action>' inside think blocks;
plus a non-thinking regression check with the Qwen2.5 tokenizer (strip flag off).

Run with the DUET env python:
    /data/home/qisheng/miniconda3/envs/duet/bin/python -m pytest tests/test_qwen35_context.py -v
"""
import asyncio
import os
from collections import namedtuple
from types import SimpleNamespace

import pytest
import requests
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from agentevolver.module.context_manager.cmt_base import (
    chat_template_ids,
    extract_assistant_header_tokens,
)
from agentevolver.module.context_manager.context_policy import ContextBudgetError
from agentevolver.module.context_manager.cmt_linear import Linear_CMT
from agentevolver.module.trainer.simple_completion_callback import (
    SimpleCompletionCallback,
)
from agentevolver.schema.trajectory import Reward

QWEN35_THINK_DIR = "/data/shared_models/Qwen3.5-2B-think"
QWEN35_4B_THINK_DIR = "/data/shared_models/Qwen3.5-4B-think"
QWEN25_DIR = "/data/shared_models/Qwen2.5-1.5B-Instruct"

FakeToken = namedtuple("FakeToken", ["token_id", "logprob"], defaults=[-0.125])


def make_config(
    strip_think: bool,
    env_type: str = "alfworld",
    context_management: dict | None = None,
    max_prompt_length: int = 6144,
    max_response_length: int = 2048,
) -> OmegaConf:
    rollout = {
        "response_length": max_response_length,
        "max_model_len": max_prompt_length + max_response_length,
        "max_env_len": 1024,
        "sliding_window_size": -1,
        "strip_think_in_history": strip_think,
    }
    if context_management is not None:
        rollout["context_management"] = context_management
    return OmegaConf.create({
        "actor_rollout_ref": {"rollout": {
            **rollout,
        }},
        "data": {
            "max_prompt_length": max_prompt_length,
            "max_response_length": max_response_length,
        },
        "env_service": {"env_type": env_type, "env_params": {"action_format": "react_tags"}},
        "exp_manager": {"experience_template": "Here are some related experiences: {}"},
        "trainer": {"n_gpus_per_node": 1, "nnodes": 1},
    })


def make_llm_output(tokenizer, content: str) -> dict:
    """Mimic a vllm response: raw generated tokens (no opening <think>, ends with eos)."""
    token_ids = tokenizer(content, return_tensors="pt", padding=False)["input_ids"][0].tolist()
    return {
        "role": "assistant",
        "content": content,
        "tokens": [FakeToken(t) for t in token_ids + [tokenizer.eos_token_id]],
        "stop_reason": "stop",
    }


def make_finished_llm_output(tokenizer, content: str, finish_reason: str) -> dict:
    """Mimic the callback contract, including a genuine no-EOS length event."""
    token_ids = tokenizer(
        content, return_tensors="pt", padding=False
    )["input_ids"][0].tolist()
    if finish_reason == "stop":
        token_ids.append(tokenizer.eos_token_id)
    return {
        "role": "assistant",
        "content": content,
        "sampled_content": content,
        "tokens": [
            FakeToken(token_id, -0.01 * (index + 1))
            for index, token_id in enumerate(token_ids)
        ],
        "finish_reason": finish_reason,
        "truncated_by_length": finish_reason == "length",
        "stop_reason": None,
    }


def run_finish_reason_episode(tokenizer, snapshot_selection: str, data_id: str):
    cfg = make_config(
        strip_think=False,
        context_management={
            "enabled": True,
            "max_prompt_tokens": 512,
            "recent_turns": 1,
            "min_recent_turns": 1,
            "snapshot_training": True,
            "snapshot_selection": snapshot_selection,
        },
        max_prompt_length=512,
    )
    cmt = Linear_CMT(cfg, tokenizer)
    cmt.save_init_input([
        {"role": "system", "content": "Choose one action."},
        {"role": "user", "content": "initial observation"},
    ])
    completions = [
        ("stop", "reasoning complete\n</think>\n\n<action>\nlook\n</action>"),
        ("length", "unfinished reasoning without a closing tag"),
    ]
    for turn, (finish_reason, content) in enumerate(completions):
        prompt = cmt.prepare_next_llm_context()
        cmt.save_llm_output(
            make_finished_llm_output(tokenizer, content, finish_reason),
            prompt,
        )
        # A length completion is still an environment interaction. It is not
        # retried or discarded by the data layer.
        cmt.save_env_output({"content": f"environment result {turn}"})

    cmt.data_id = data_id
    cmt.rollout_id = data_id
    cmt.task_id = f"finish-{data_id}"
    cmt.reward = Reward(outcome=0.0, success_rate=0.0)
    return cfg, cmt


def run_episode(cmt, tokenizer, llm_contents, env_obs):
    """Simulate: init(system+user obs0) -> [llm, env] * n."""
    cmt.save_init_input([
        {"role": "system", "content": "You are an agent. Reply with <action>...</action>."},
        {"role": "user", "content": env_obs[0]},
    ])
    for turn, content in enumerate(llm_contents):
        input_msg_ref = cmt.prepare_next_llm_context()
        cmt.save_llm_output(make_llm_output(tokenizer, content), input_msg_ref)
        cmt.save_env_output({"content": env_obs[turn + 1]})
    cmt.data_id = "0"
    cmt.rollout_id = "0"
    cmt.task_id = "test_task"
    cmt.reward = Reward(outcome=1.0, success_rate=1.0)


@pytest.fixture(scope="module")
def qwen35_tok():
    if not os.path.isdir(QWEN35_THINK_DIR):
        pytest.skip(f"{QWEN35_THINK_DIR} not found")
    return AutoTokenizer.from_pretrained(QWEN35_THINK_DIR, trust_remote_code=True)


@pytest.fixture(scope="module")
def qwen25_tok():
    if not os.path.isdir(QWEN25_DIR):
        pytest.skip(f"{QWEN25_DIR} not found")
    return AutoTokenizer.from_pretrained(QWEN25_DIR, trust_remote_code=True)


@pytest.fixture(scope="module")
def qwen35_4b_tok():
    if not os.path.isdir(QWEN35_4B_THINK_DIR):
        pytest.skip(f"{QWEN35_4B_THINK_DIR} not found")
    return AutoTokenizer.from_pretrained(QWEN35_4B_THINK_DIR, trust_remote_code=True)


# 4-turn episode with fake think + action contents (vllm-style: no opening <think>)
LLM_CONTENTS = [
    "reasoning A about the kitchen\n</think>\n\n<action>\ngo north\n</action>",
    "reasoning B, maybe </action> is mentioned here\n</think>\n\n<action>\nopen door\n</action>",
    "reasoning C\n</think>\n\n<action>\ntake apple\n</action>",
    "reasoning D final\n</think>\n\n<action>\nput apple in fridge\n</action>",
]
ENV_OBS = ["obs 0: you are in a kitchen", "obs 1: hallway", "obs 2: door is open",
           "obs 3: you took the apple", "obs 4: task complete"]


def test_patched_template_default_gen_prompt(qwen35_tok):
    """The -think dir must yield the 5-token thinking generation prompt by default."""
    msgs = [{"role": "user", "content": "obs"}]
    no_gen = chat_template_ids(qwen35_tok, msgs, add_generation_prompt=False)
    with_gen = chat_template_ids(qwen35_tok, msgs, add_generation_prompt=True)
    assert with_gen[len(no_gen):] == [248045, 74455, 198, 248068, 198]
    # and extract_assistant_header_tokens must pick exactly this header
    assert extract_assistant_header_tokens(qwen35_tok) == [248045, 74455, 198, 248068, 198]


def test_snapshot_training_uses_exact_sampling_condition(qwen35_tok):
    """Compression of later prompts cannot rewrite an earlier training pair."""
    cfg = make_config(
        strip_think=False,
        context_management={
            "enabled": True,
            "max_prompt_tokens": 512,
            "recent_turns": 1,
            "min_recent_turns": 1,
            "history_observation_max_tokens": 32,
            "reasoning_history_tokens": 0,
            "snapshot_training": True,
            "snapshot_selection": "first",
        },
        max_prompt_length=512,
    )
    cmt = Linear_CMT(cfg, qwen35_tok)
    run_episode(cmt, qwen35_tok, LLM_CONTENTS, ENV_OBS)

    assert len(cmt.decision_snapshots) == len(LLM_CONTENTS)
    first = cmt.decision_snapshots[0]
    first_prompt_before = list(first.prompt_token_ids)
    sample = cmt.group_tokenize()[0]

    assert sample.prompt_ids == first_prompt_before
    assert sample.response_ids == first.completion_token_ids
    assert sample.input_ids == first.prompt_token_ids + first.completion_token_ids
    assert sample.prompt_loss_mask == [0] * len(first.prompt_token_ids)
    assert sample.response_loss_mask == [1] * len(first.completion_token_ids)
    assert first.prompt_hash == cmt.context_policy.ids_hash(first.prompt_token_ids)
    assert first.raw_prompt_hash == first.prompt_hash
    assert cmt.metadata["prompt_hash"] == first.prompt_hash
    assert cmt.metadata["raw_prompt_hash"] == first.raw_prompt_hash
    # The raw event log is not retroactively rewritten by the new policy.
    assert "reasoning A" in [
        msg.content_for_future for msg in cmt.full_context if msg.author == "llm"
    ][0]


def test_context_budget_is_deterministic_and_preserves_current_state(qwen35_tok):
    current_observation = (
        "CURRENT ROOM: kitchen\nAVAILABLE ACTIONS: go north, take apple, open fridge"
    )
    cfg = make_config(
        strip_think=False,
        context_management={
            "enabled": True,
            "max_prompt_tokens": 220,
            "recent_turns": 1,
            "min_recent_turns": 1,
            "history_observation_max_tokens": 24,
            "reasoning_history_tokens": 0,
            "snapshot_training": True,
            "snapshot_selection": "last",
        },
        max_prompt_length=220,
    )
    cmt = Linear_CMT(cfg, qwen35_tok)
    cmt.save_init_input([
        {"role": "system", "content": "GOAL: put the apple in the fridge"},
        {"role": "user", "content": "initial room"},
    ])
    for i in range(6):
        prompt = cmt.prepare_next_llm_context()
        cmt.save_llm_output(
            make_llm_output(qwen35_tok, f"reason {i}\n</think>\n\n<action>\nlook\n</action>"),
            prompt,
        )
        observation = current_observation if i == 5 else (
            (f"old observation {i} " * 20)
            + "\nAVAILABLE ACTIONS: look, go north, go south"
        )
        cmt.save_env_output({"content": observation})

    one = cmt.context_policy.build(cmt.full_context)
    two = cmt.context_policy.build(cmt.full_context)
    assert one.messages == two.messages
    assert one.prompt_token_ids == two.prompt_token_ids
    assert len(one.prompt_token_ids) <= 220
    rendered = "\n".join(message["content"] for message in one.messages)
    assert "GOAL: put the apple in the fridge" in rendered
    assert current_observation in rendered
    assert one.stats["raw_prompt_tokens"] > one.stats["managed_prompt_tokens"]
    assert one.stats["dropped_turns"] > 0


def test_context_budget_never_silently_crops_current_observation(qwen35_tok):
    cfg = make_config(
        strip_think=False,
        context_management={
            "enabled": True,
            "max_prompt_tokens": 80,
            "recent_turns": 1,
            "min_recent_turns": 1,
            "allow_current_observation_truncation": False,
            "snapshot_training": True,
        },
        max_prompt_length=80,
    )
    cmt = Linear_CMT(cfg, qwen35_tok)
    cmt.save_init_input([
        {"role": "system", "content": "goal"},
        {"role": "user", "content": "CURRENT " + ("clickable item " * 200)},
    ])
    with pytest.raises(ContextBudgetError, match="protected context does not fit"):
        cmt.prepare_next_llm_context()


def test_h15_managed_control_share_pretreatment_prompts_then_diverge(qwen35_4b_tok):
    common = {
        "enabled": True,
        "max_prompt_tokens": 22528,
        "min_recent_turns": 1,
        "recent_observation_max_tokens": -1,
        "allow_current_observation_truncation": False,
        "reasoning_history_tokens": 0,
        "snapshot_training": True,
        "snapshot_selection": "token_weighted",
        "snapshot_selection_seed": 2025,
    }
    managed_cfg = make_config(
        strip_think=False,
        env_type="webshop",
        context_management={
            **common,
            "recent_turns": 4,
            "history_observation_max_tokens": 512,
        },
        max_prompt_length=22528,
        max_response_length=10240,
    )
    control_cfg = make_config(
        strip_think=False,
        env_type="webshop",
        context_management={
            **common,
            "recent_turns": 15,
            "history_observation_max_tokens": -1,
        },
        max_prompt_length=22528,
        max_response_length=10240,
    )
    managed = Linear_CMT(managed_cfg, qwen35_4b_tok)
    control = Linear_CMT(control_cfg, qwen35_4b_tok)
    initial = [
        {"role": "system", "content": "buy the requested item"},
        {"role": "user", "content": "initial WebShop observation"},
    ]
    managed.save_init_input(initial)
    control.save_init_input(initial)

    managed_prompt_ids = []
    control_prompt_ids = []
    for turn in range(6):
        managed_prompt = managed.prepare_next_llm_context()
        control_prompt = control.prepare_next_llm_context()
        managed_prompt_ids.append(
            chat_template_ids(
                qwen35_4b_tok, managed_prompt, add_generation_prompt=True
            )
        )
        control_prompt_ids.append(
            chat_template_ids(
                qwen35_4b_tok, control_prompt, add_generation_prompt=True
            )
        )
        completion = (
            f"reasoning {turn}\n</think>\n\n"
            f"<action>\nclick[item-{turn}]\n</action>"
        )
        managed.save_llm_output(
            make_llm_output(qwen35_4b_tok, completion), managed_prompt
        )
        control.save_llm_output(
            make_llm_output(qwen35_4b_tok, completion), control_prompt
        )
        observation = (
            f"page {turn}: " + (f"verbose product detail {turn} " * 180)
            + f"\nClickable elements: ['item-{turn}', 'next']"
        )
        managed.save_env_output({"content": observation})
        control.save_env_output({"content": observation})

    assert managed_prompt_ids[:5] == control_prompt_ids[:5]
    assert managed_prompt_ids[5] != control_prompt_ids[5]
    assert (
        managed.decision_snapshots[5].raw_prompt_hash
        == control.decision_snapshots[5].raw_prompt_hash
        == control.decision_snapshots[5].prompt_hash
    )
    managed_stats = managed.decision_snapshots[5].context_stats
    control_stats = control.decision_snapshots[5].context_stats
    assert managed_stats["compressed_turns"] == 1
    assert managed_stats["clipped_observations"] >= 1
    assert managed_stats["managed_prompt_tokens"] < managed_stats["raw_prompt_tokens"]
    assert control_stats["compressed_turns"] == 0
    assert control_stats["clipped_observations"] == 0
    assert control_stats["managed_prompt_tokens"] == control_stats["raw_prompt_tokens"]
    assert managed_stats["managed_prompt_tokens"] <= 22528
    assert control_stats["managed_prompt_tokens"] <= 22528


def test_snapshot_completion_is_never_silently_truncated(qwen35_tok):
    cfg = make_config(
        strip_think=False,
        context_management={
            "enabled": True,
            "max_prompt_tokens": 256,
            "recent_turns": 1,
            "min_recent_turns": 1,
            "snapshot_training": True,
            "snapshot_selection": "last",
        },
        max_prompt_length=256,
        max_response_length=8,
    )
    cmt = Linear_CMT(cfg, qwen35_tok)
    cmt.save_init_input([
        {"role": "system", "content": "goal"},
        {"role": "user", "content": "observation"},
    ])
    prompt = cmt.prepare_next_llm_context()
    cmt.save_llm_output(make_llm_output(qwen35_tok, LLM_CONTENTS[0]), prompt)
    cmt.data_id = cmt.rollout_id = cmt.task_id = "0"
    cmt.reward = Reward(outcome=0.0, success_rate=0.0)
    with pytest.raises(RuntimeError, match="never silently truncate"):
        cmt.group_tokenize()


def test_snapshot_metadata_and_rollout_logprobs_reach_dataproto(qwen35_tok):
    from agentevolver.module.env_manager.env_manager import ParallelEnvManager
    from agentevolver.module.trainer.ae_ray_trainer import (
        _validation_rollout_audit,
    )

    cfg = make_config(
        strip_think=False,
        context_management={
            "enabled": True,
            "max_prompt_tokens": 512,
            "recent_turns": 1,
            "min_recent_turns": 1,
            "snapshot_training": True,
            "snapshot_selection": "first",
        },
        max_prompt_length=512,
    )
    cmt = Linear_CMT(cfg, qwen35_tok)
    run_episode(cmt, qwen35_tok, LLM_CONTENTS[:2], ENV_OBS[:3])

    sample = cmt.group_tokenize()[0]
    manager = ParallelEnvManager.__new__(ParallelEnvManager)
    manager.config = cfg
    manager.tokenizer = qwen35_tok
    manager.pad_token_id = qwen35_tok.pad_token_id
    sample.extras = manager.get_extra(cmt)
    batch = manager.samples_to_dataproto([sample])

    response_len = len(sample.response_ids)
    assert batch.batch["rollout_log_probs"].shape == (1, response_len)
    assert batch.batch["rollout_log_probs_mask"].sum().item() == response_len
    assert batch.batch["rollout_log_probs"][0].tolist() == pytest.approx(
        cmt.decision_snapshots[0].completion_log_probs
    )
    assert batch.batch["context_managed_prompt_tokens"][0].item() == len(
        sample.prompt_ids
    )
    assert batch.non_tensor_batch["extras"][0]["prompt_hash"] == (
        cmt.decision_snapshots[0].prompt_hash
    )
    assert batch.batch["context_decision_count"].tolist() == [2]
    assert len(batch.non_tensor_batch["decision_context_stats"][0]) == 2
    decision_audit = batch.non_tensor_batch["extras"][0]["decision_audit"]
    assert len(decision_audit) == 2
    assert decision_audit[0]["prompt_hash"] == cmt.decision_snapshots[0].prompt_hash
    assert decision_audit[0]["raw_prompt_hash"] == (
        cmt.decision_snapshots[0].raw_prompt_hash
    )
    assert decision_audit[0]["completion_tokens"] == len(
        cmt.decision_snapshots[0].completion_token_ids
    )
    assert _validation_rollout_audit(batch)["context_decision_count"] == [2]
    # Old completion producers did not include finish_reason. Such samples stay
    # trainable and are explicitly observable as unknown, never as length.
    assert batch.batch["finish_reason_code"][0].item() == -1
    assert not batch.batch["truncated_by_length"][0].item()


@pytest.mark.parametrize(
    ("finish_reason", "expected_length"),
    [("stop", False), ("length", True)],
)
def test_completion_callback_preserves_finish_contract_and_raw_tokens(
    finish_reason, expected_length
):
    class FakeMessage:
        def model_dump(self, **_kwargs):
            return {"role": "assistant", "content": "partial completion"}

    wire_tokens = [
        SimpleNamespace(
            token="token_id:101",
            logprob=-0.25,
            bytes=list(b"a"),
        ),
        SimpleNamespace(
            token="token_id:202",
            logprob=-0.5,
            bytes=list(b"b"),
        ),
    ]
    choice = SimpleNamespace(
        message=FakeMessage(),
        finish_reason=finish_reason,
        stop_reason=None,
        logprobs=SimpleNamespace(content=wire_tokens),
    )
    completion = SimpleNamespace(id="completion-0", choices=[choice])
    callback = SimpleCompletionCallback.__new__(SimpleCompletionCallback)
    callback.config = OmegaConf.create(
        {"env_service": {"env_params": {"action_format": "react"}}}
    )
    messages = [{"role": "user", "content": "observation"}]

    asyncio.run(callback(messages, completion, {}))

    result = messages[-1]
    assert result["finish_reason"] == finish_reason
    assert result["truncated_by_length"] is expected_length
    assert [token.token_id for token in result["tokens"]] == [101, 202]
    assert [token.logprob for token in result["tokens"]] == [-0.25, -0.5]
    # In particular, the length path does not append a fabricated EOS token.
    assert len(result["tokens"]) == len(wire_tokens)


def test_legacy_length_token_path_does_not_fabricate_eos(qwen35_tok):
    cfg = make_config(
        strip_think=False,
        context_management={
            "enabled": False,
            "snapshot_training": False,
        },
        max_prompt_length=512,
    )
    cmt = Linear_CMT(cfg, qwen35_tok)
    cmt.save_init_input(
        [
            {"role": "system", "content": "choose an action"},
            {"role": "user", "content": "observation"},
        ]
    )
    prompt = cmt.prepare_next_llm_context()
    raw = make_finished_llm_output(
        qwen35_tok, "unfinished reasoning", "length"
    )
    ext_msg = cmt.save_llm_output(raw, prompt)

    sampled_ids = [token.token_id for token in raw["tokens"]]
    assert ext_msg.token_arr[-len(sampled_ids) :] == sampled_ids
    assert ext_msg.token_arr[-1] != qwen35_tok.eos_token_id


def test_agent_flow_length_completion_terminates_without_env_step():
    from agentevolver.module.agent_flow.agent_flow import AgentFlow

    class FakeContext:
        def __init__(self):
            self.metadata = {}
            self.discarded = False
            self.is_terminated = False
            self.generated_token_cnt = 0
            self.reward = None
            self.full_context = []

        def save_init_input(self, *_args, **_kwargs):
            return None

        def prepare_next_llm_context(self):
            return [{"role": "user", "content": "observation"}]

        def check_context_token_num_safe(self, _messages):
            return True

        def save_llm_output(self, output, **_kwargs):
            self.generated_token_cnt = len(output["tokens"])

        def compute_madness(self):
            return 0.0

        def reward_patch(self, reward):
            return reward

        def remove_last_context(self):
            return None

        def generate_log(self, **_kwargs):
            return None

    class FakeEnv:
        def __init__(self):
            self.step_calls = 0

        def step(self, *_args, **_kwargs):
            self.step_calls += 1
            raise AssertionError("a length completion must not reach env.step")

        def get_tools_info(self, *_args, **_kwargs):
            return {"success_rate": 1.0}

    class FakeExpWorker:
        def manage_rollout_context(self, init_messages, traj_exp_config):
            return init_messages, traj_exp_config

    class FakeRewardCalculator:
        def calculate_reward(self, *_args, **_kwargs):
            return {"score": 1.0, "reason": "would otherwise look successful"}

    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "thinking_mode": "native_qwen35",
                    "use_qwen3": True,
                    "length_truncation_reward": -0.1,
                }
            }
        }
    )
    flow = AgentFlow.__new__(AgentFlow)
    flow.config = cfg
    flow.max_steps = 3
    flow.tokenizer = SimpleNamespace()
    flow.exp_worker = FakeExpWorker()
    flow._reward_calculator = FakeRewardCalculator()
    flow.sparse = True
    flow.sciworld_success_threshold = 0.0
    flow.console_debug_mode = False
    flow.llm_chat_fn = lambda *_args, **_kwargs: {
        "role": "assistant",
        "content": "unfinished",
        "tokens": [SimpleNamespace(token_id=i) for i in range(10240)],
        "finish_reason": "length",
        "truncated_by_length": True,
    }
    context = FakeContext()
    env = FakeEnv()
    traj_exp_config = SimpleNamespace(
        query=None,
        train_mode="discard",
        add_exp=False,
        experience_list=[],
    )

    result = flow.execute(
        context_manager=context,
        init_messages=[{"role": "user", "content": "task"}],
        env=env,
        instance_id="instance",
        tmux={"step": [0], "token": [0]},
        stop=[False],
        thread_index=0,
        task_id="task",
        traj_exp_config=traj_exp_config,
    )

    assert env.step_calls == 0
    assert result.metadata["episode_end_reason"] == "length_truncation"
    assert result.metadata["length_truncation_step"] == 0
    assert result.reward.outcome == pytest.approx(-0.1)
    assert result.reward.success_rate == 0.0


def _make_agent_flow_action_harness(world_interaction, *, step_error=None):
    from agentevolver.module.agent_flow.agent_flow import AgentFlow

    class FakeContext:
        def __init__(self):
            self.metadata = {}
            self.discarded = False
            self.is_terminated = False
            self.generated_token_cnt = 0
            self.reward = None
            self.full_context = []
            self.saved_env_outputs = []
            self.reward_patch_calls = 0

        def save_init_input(self, *_args, **_kwargs):
            return None

        def prepare_next_llm_context(self):
            return [{"role": "user", "content": "observation"}]

        def check_context_token_num_safe(self, _messages):
            return True

        def save_llm_output(self, output, **_kwargs):
            self.generated_token_cnt = len(output["tokens"])

        def prepare_world_interaction(self):
            return world_interaction

        def save_env_output(self, state, **_kwargs):
            self.saved_env_outputs.append(dict(state))

        def compute_madness(self):
            return 0.0

        def reward_patch(self, reward):
            self.reward_patch_calls += 1
            # A malformed action must remain a hard failure even if a custom
            # patch would otherwise make it look successful.
            reward.outcome = 1.0
            reward.success_rate = 1.0
            return reward

        def remove_last_context(self):
            return None

        def generate_log(self, **_kwargs):
            return None

    class FakeEnv:
        def __init__(self):
            self.step_calls = 0
            self.info_calls = 0
            self.actions = []

        def step(self, _instance_id, action):
            self.step_calls += 1
            self.actions.append(action)
            if step_error is not None:
                raise step_error
            return {
                "state": [{"role": "user", "content": "observation"}],
                "reward": 0.0,
                "is_terminated": True,
            }

        def get_tools_info(self, *_args, **_kwargs):
            self.info_calls += 1
            return {"success_rate": 1.0}

    class FakeExpWorker:
        def manage_rollout_context(self, init_messages, traj_exp_config):
            return init_messages, traj_exp_config

    class FakeRewardCalculator:
        def __init__(self):
            self.calls = 0

        def calculate_reward(self, *_args, **_kwargs):
            self.calls += 1
            return {"score": 1.0, "reason": "would otherwise look successful"}

    cfg = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "thinking_mode": "native_qwen35",
                    "use_qwen3": True,
                }
            }
        }
    )
    reward_calculator = FakeRewardCalculator()
    flow = AgentFlow.__new__(AgentFlow)
    flow.config = cfg
    flow.max_steps = 1
    flow.tokenizer = SimpleNamespace()
    flow.exp_worker = FakeExpWorker()
    flow._reward_calculator = reward_calculator
    flow.sparse = True
    flow.sciworld_success_threshold = 0.0
    flow.console_debug_mode = False
    flow.llm_chat_fn = lambda *_args, **_kwargs: {
        "role": "assistant",
        "content": "reasoning only\n</think>\n\n",
        "tokens": [SimpleNamespace(token_id=1)],
        "finish_reason": "stop",
        "truncated_by_length": False,
    }
    context = FakeContext()
    env = FakeEnv()
    traj_exp_config = SimpleNamespace(
        query=None,
        train_mode="discard",
        add_exp=False,
        experience_list=[],
    )
    return flow, context, env, reward_calculator, traj_exp_config


def _execute_agent_flow_harness(flow, context, env, traj_exp_config):
    return flow.execute(
        context_manager=context,
        init_messages=[{"role": "user", "content": "task"}],
        env=env,
        instance_id="instance",
        tmux={"step": [0], "token": [0]},
        stop=[False],
        thread_index=0,
        task_id="task",
        traj_exp_config=traj_exp_config,
    )


def test_agent_flow_empty_post_think_is_local_malformed_failure():
    flow, context, env, reward_calculator, traj_exp_config = (
        _make_agent_flow_action_harness("   \n")
    )

    result = _execute_agent_flow_harness(
        flow, context, env, traj_exp_config
    )

    assert env.step_calls == 0
    assert env.info_calls == 0
    assert reward_calculator.calls == 0
    assert result.is_terminated is True
    assert result.metadata["episode_end_reason"] == "malformed_action"
    assert result.metadata["malformed_action"] is True
    assert result.metadata["malformed_action_step"] == 0
    assert "error" not in result.metadata
    assert result.reward.outcome == 0.0
    assert result.reward.success_rate == 0.0
    assert context.reward_patch_calls == 1
    assert context.saved_env_outputs == [
        {
            "content": (
                "Malformed action: the model produced no environment-facing "
                "content after reasoning."
            ),
            "role": "user",
        }
    ]


def test_agent_flow_normal_action_semantics_are_unchanged():
    action = "<action>\nlook\n</action>"
    flow, context, env, reward_calculator, traj_exp_config = (
        _make_agent_flow_action_harness(action)
    )

    result = _execute_agent_flow_harness(
        flow, context, env, traj_exp_config
    )

    assert env.step_calls == 1
    assert env.actions == [{"content": action, "role": "assistant"}]
    assert env.info_calls == 1
    assert reward_calculator.calls == 1
    assert result.metadata["episode_end_reason"] == "env_terminated"
    assert "malformed_action" not in result.metadata
    assert result.reward.outcome == 1.0
    assert result.reward.success_rate == 1.0


def _http_error(status_code):
    response = requests.Response()
    response.status_code = status_code
    response.url = "http://env.test/step"
    return requests.exceptions.HTTPError(
        f"{status_code} server error", response=response
    )


@pytest.mark.parametrize(
    "step_error",
    [
        requests.exceptions.Timeout("env timeout"),
        requests.exceptions.ConnectionError("env connection failed"),
        _http_error(503),
    ],
    ids=["timeout", "connection", "http_5xx"],
)
def test_agent_flow_rethrows_env_infrastructure_errors(step_error):
    flow, context, env, reward_calculator, traj_exp_config = (
        _make_agent_flow_action_harness(
            "<action>\nlook\n</action>", step_error=step_error
        )
    )

    with pytest.raises(type(step_error)) as exc_info:
        _execute_agent_flow_harness(flow, context, env, traj_exp_config)

    assert exc_info.value is step_error
    assert env.step_calls == 1
    assert env.info_calls == 0
    assert reward_calculator.calls == 0
    assert context.saved_env_outputs == []
    assert context.reward_patch_calls == 0


@pytest.mark.parametrize(
    ("snapshot_selection", "expected_step", "expected_reason", "expected_length"),
    [
        ("first", 0, "stop", False),
        ("last", 1, "length", True),
    ],
)
def test_finish_reason_stays_aligned_after_snapshot_selection(
    qwen35_tok,
    snapshot_selection,
    expected_step,
    expected_reason,
    expected_length,
):
    from agentevolver.module.env_manager.env_manager import ParallelEnvManager

    cfg, cmt = run_finish_reason_episode(
        qwen35_tok, snapshot_selection, data_id="0"
    )
    sample = cmt.group_tokenize()[0]
    chosen = cmt.decision_snapshots[expected_step]
    manager = ParallelEnvManager.__new__(ParallelEnvManager)
    manager.config = cfg
    manager.tokenizer = qwen35_tok
    manager.pad_token_id = qwen35_tok.pad_token_id
    sample.extras = manager.get_extra(cmt)
    batch = manager.samples_to_dataproto([sample])

    assert cmt.metadata["decision_finish_reasons"] == ["stop", "length"]
    assert cmt.metadata["decision_truncated_by_length"] == [False, True]
    assert cmt.metadata["length_truncated_decision_count"] == 1
    assert sample.minor_index_id == expected_step
    assert sample.response_ids == chosen.completion_token_ids
    assert sample.extras["finish_reason"] == expected_reason
    assert sample.extras["truncated_by_length"] is expected_length
    assert batch.non_tensor_batch["finish_reasons"].tolist() == [
        expected_reason
    ]
    assert batch.batch["finish_reason_code"].tolist() == [expected_step]
    assert batch.batch["truncated_by_length"].tolist() == [expected_length]
    assert batch.batch["decision_count"].tolist() == [2]
    assert batch.batch["length_truncated_decision_count"].tolist() == [1]
    assert batch.batch["has_length_truncated_decision"].tolist() == [True]
    assert batch.batch["rollout_log_probs"][0, : len(sample.response_ids)].tolist() == pytest.approx(
        chosen.completion_log_probs
    )
    if expected_length:
        assert qwen35_tok.eos_token_id not in sample.response_ids


def test_finish_flags_survive_sequence_padding_and_balance_reorder(qwen35_tok):
    from agentevolver.module.env_manager.env_manager import ParallelEnvManager

    cfg, stop_cmt = run_finish_reason_episode(
        qwen35_tok, "first", data_id="0"
    )
    _, length_cmt = run_finish_reason_episode(
        qwen35_tok, "last", data_id="1"
    )
    manager = ParallelEnvManager.__new__(ParallelEnvManager)
    manager.config = cfg
    manager.tokenizer = qwen35_tok
    manager.pad_token_id = qwen35_tok.pad_token_id
    samples = manager.trajectories_to_samples([stop_cmt, length_cmt])
    assert len(samples[0].response_ids) != len(samples[1].response_ids)

    batch = manager.samples_to_dataproto(samples)
    assert batch.batch["responses"].shape[0] == 2
    assert batch.batch["truncated_by_length"].tolist() == [False, True]
    assert batch.non_tensor_batch["finish_reasons"].tolist() == [
        "stop",
        "length",
    ]

    # The trainer's balance path ultimately uses DataProto.reorder. Tensor and
    # non-tensor termination state must follow the same permutation.
    batch.reorder(torch.tensor([1, 0], dtype=torch.long))
    assert batch.batch["finish_reason_code"].tolist() == [1, 0]
    assert batch.batch["truncated_by_length"].tolist() == [True, False]
    assert batch.batch["length_truncated_decision_count"].tolist() == [1, 1]
    assert batch.non_tensor_batch["finish_reasons"].tolist() == [
        "length",
        "stop",
    ]
    assert batch.non_tensor_batch["decision_finish_reasons"].tolist() == [
        ["stop", "length"],
        ["stop", "length"],
    ]


def test_production_4b_contract_over_30_turns(qwen35_4b_tok):
    """Production tokenizer property test: deterministic, bounded, immutable."""
    cfg = make_config(
        strip_think=False,
        context_management={
            "enabled": True,
            "max_prompt_tokens": 6144,
            "recent_turns": 2,
            "min_recent_turns": 1,
            "history_observation_max_tokens": 160,
            "allow_current_observation_truncation": False,
            "reasoning_history_tokens": 0,
            "snapshot_training": True,
            "snapshot_selection": "token_weighted",
            "snapshot_selection_seed": 2025,
        },
        max_prompt_length=6144,
        max_response_length=2048,
    )
    cmt = Linear_CMT(cfg, qwen35_4b_tok)
    cmt.save_init_input([
        {"role": "system", "content": "Follow the goal and output an action."},
        {"role": "user", "content": "目标: put café mug in cabinet ☕"},
        {"role": "assistant", "content": "I will follow the requested format."},
    ])
    raw_observations = []
    for turn in range(30):
        prompt = cmt.prepare_next_llm_context()
        assert len(chat_template_ids(
            qwen35_4b_tok, prompt, add_generation_prompt=True
        )) <= 6144
        output = (
            f"reasoning turn {turn} with unicode café\n</think>\n\n"
            f"<action>\nlook at object {turn}\n</action>"
        )
        cmt.save_llm_output(make_llm_output(qwen35_4b_tok, output), prompt)
        observation = (
            f"observation {turn}: " + ("room detail " * 45)
            + "\nAVAILABLE ACTIONS: look, go north, take mug"
        )
        raw_observations.append(observation)
        cmt.save_env_output({"content": observation})

    assert len(cmt.decision_snapshots) == 30
    assert all(
        snapshot.prompt_hash
        == cmt.context_policy.ids_hash(snapshot.prompt_token_ids)
        for snapshot in cmt.decision_snapshots
    )
    assert [
        message.content for message in cmt.full_context if message.author == "env"
    ] == raw_observations

    first_build = cmt.context_policy.build(cmt.full_context)
    second_build = cmt.context_policy.build(cmt.full_context)
    assert first_build == second_build
    assert len(first_build.prompt_token_ids) <= 6144
    assert first_build.stats["managed_prompt_tokens"] < first_build.stats[
        "raw_prompt_tokens"
    ]

    cmt.data_id = cmt.rollout_id = "0"
    cmt.task_id = "production_contract"
    cmt.reward = Reward(outcome=1.0, success_rate=1.0)
    sample = cmt.group_tokenize()[0]
    chosen = cmt.decision_snapshots[sample.minor_index_id]
    assert sample.prompt_ids == chosen.prompt_token_ids
    assert sample.response_ids == chosen.completion_token_ids


def test_token_arr_matches_chat_template(qwen35_tok):
    """(i) concat(token_arr) == apply_chat_template(full message list):
    history think stripped, final turn intact."""
    cmt = Linear_CMT(make_config(strip_think=True), qwen35_tok)
    run_episode(cmt, qwen35_tok, LLM_CONTENTS, ENV_OBS)

    sample = cmt.group_tokenize()[0]

    # reference render: raw messages, trailing env obs dropped (as tokenize_steps does)
    msgs = cmt.prepare_previous_context(mod="raw")
    assert msgs[-1]["role"] == "user"
    msgs = msgs[:-1]
    ref_text = qwen35_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ref_ids = qwen35_tok(ref_text, return_tensors="pt", padding=False)["input_ids"][0].tolist()

    assert sample.input_ids == ref_ids, (
        f"token_arr concat != chat template render\n"
        f"got:      {qwen35_tok.decode(sample.input_ids)!r}\n"
        f"expected: {ref_text!r}"
    )
    # history turns stripped, final turn intact
    decoded = qwen35_tok.decode(sample.input_ids)
    assert "reasoning A" not in decoded and "reasoning B" not in decoded and "reasoning C" not in decoded
    assert "<think>\nreasoning D final\n</think>" in decoded
    assert decoded.count("<action>\n") == 4  # one per turn (system prompt's inline mention excluded)


def test_three_message_initialization_is_not_duplicated(qwen35_tok):
    cmt = Linear_CMT(make_config(strip_think=True), qwen35_tok)
    initial = [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "task goal"},
        {"role": "user", "content": "current observation"},
    ]
    cmt.save_init_input(initial)
    prompt = cmt.prepare_next_llm_context()
    cmt.save_llm_output(make_llm_output(qwen35_tok, LLM_CONTENTS[0]), prompt)
    cmt.save_env_output({"content": "trailing observation"})
    cmt.data_id = cmt.rollout_id = cmt.task_id = "0"
    cmt.reward = Reward(outcome=0.0, success_rate=0.0)
    sample = cmt.group_tokenize()[0]

    reference = chat_template_ids(
        qwen35_tok,
        cmt.prepare_previous_context(mod="raw")[:-1],
        add_generation_prompt=False,
    )
    assert sample.input_ids == reference


def test_loss_mask_blackouts_headers(qwen35_tok):
    """(ii) generation-prompt / assistant headers are blacked out in the loss mask."""
    cmt = Linear_CMT(make_config(strip_think=True), qwen35_tok)
    run_episode(cmt, qwen35_tok, LLM_CONTENTS, ENV_OBS)
    sample = cmt.group_tokenize()[0]

    ids, mask = sample.input_ids, sample.loss_mask
    gen_prompt = [248045, 74455, 198, 248068, 198]        # <|im_start|>assistant\n<think>\n
    plain_header = [248045, 74455, 198]                   # <|im_start|>assistant\n
    im_end = qwen35_tok.eos_token_id

    # locate every assistant span start
    header_starts = [i for i in range(len(ids) - 2) if ids[i:i + 3] == plain_header]
    assert len(header_starts) == 4
    # first three are stripped history turns: 3-token header blacked out, content trained
    for start in header_starts[:-1]:
        assert ids[start:start + 5] != gen_prompt, "history turn should not contain <think> header"
        assert mask[start:start + 3] == [0, 0, 0]
        assert mask[start + 3] == 1, "history action content must stay in the loss"
    # final turn: full 5-token generation prompt blacked out, think content trained
    start = header_starts[-1]
    assert ids[start:start + 5] == gen_prompt
    assert mask[start:start + 5] == [0, 0, 0, 0, 0]
    assert mask[start + 5] == 1, "reasoning tokens of the final turn must stay in the loss"
    # eos of the final turn kept in the loss, trailing newline blacked out
    eos_pos = len(ids) - 1 - ids[::-1].index(im_end)
    assert mask[eos_pos] == 1 and all(m == 0 for m in mask[eos_pos + 1:])
    # non-llm messages (system / user / env) carry no loss
    for i, m in enumerate(mask[:header_starts[0]]):
        assert m == 0


def test_action_extraction_ignores_think(qwen35_tok):
    """(iii) '</action>'/'<action>' inside the think block must not confuse extraction."""
    cfg = make_config(strip_think=True)
    cmt = Linear_CMT(cfg, qwen35_tok)

    tricky = ("I could do <action>\nfake move\n</action> but I won't\n</think>\n\n"
              "<action>\nreal move\n</action>")
    # compression extracts the real (post-think) action
    assert cmt._compress_llm_message(tricky) == "<action>\nreal move\n</action>"

    # prepare_world_interaction only exposes the post-think part to the env
    cmt.save_init_input([{"role": "system", "content": "sys"}, {"role": "user", "content": "obs"}])
    cmt.save_llm_output(make_llm_output(qwen35_tok, tricky), cmt.prepare_next_llm_context())
    world = cmt.prepare_world_interaction()
    assert "</think>" not in world and "fake move" not in world and "real move" in world

    # _normalize_llm_output_content: '</action>' inside think must not suppress the repair
    out = {"role": "assistant", "content": "fake </action> in think\n</think>\n\n<action>\nreal",
           "stop_reason": "</action>"}
    fixed = Linear_CMT._normalize_llm_output_content(cmt, out)
    assert fixed.endswith("\n</action>") and "real" in fixed

    # a response with '</think>' but no action in the post part is passed through
    # unchanged (existing format-error path downstream)
    no_action = "thoughts only\n</think>\n\nnothing actionable"
    assert cmt._compress_llm_message(no_action) == "\n\nnothing actionable"


def test_strip_history_think_idempotent(qwen35_tok):
    """Calling the strip twice must not re-tokenize or change anything."""
    cmt = Linear_CMT(make_config(strip_think=True), qwen35_tok)
    run_episode(cmt, qwen35_tok, LLM_CONTENTS[:2], ENV_OBS[:3])
    snapshot = [list(m.token_arr) for m in cmt.full_context]
    assert cmt._strip_history_think() == 0  # already stripped inside save_env_output
    assert [list(m.token_arr) for m in cmt.full_context] == snapshot


def test_teacher_convert_strips_history_think(qwen35_tok):
    """A1-teacher: convert_offpolicy_to_cmt applies the same history <think>
    stripping as on-policy rollouts (identical formatting for DR3)."""
    from agentevolver.module.env_manager.env_manager import ParallelEnvManager
    from agentevolver.schema.trajectory import Trajectory

    steps = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": ENV_OBS[0]},
        {"role": "assistant", "content": LLM_CONTENTS[0]},
        {"role": "user", "content": ENV_OBS[1]},
        {"role": "assistant", "content": LLM_CONTENTS[1]},
    ]
    traj = Trajectory(
        data_id="0", rollout_id="0", steps=steps, query="q",
        is_terminated=True, reward=Reward(outcome=1.0, success_rate=1.0),
        metadata={"task_id": "t0", "is_teacher": True, "has_log_prob": False},
    )
    traj.task_id = "t0"
    config = make_config(strip_think=True)
    # convert_offpolicy_to_cmt does not touch self -> call unbound
    cmt = ParallelEnvManager.convert_offpolicy_to_cmt(None, [traj], config, qwen35_tok)[0]

    contents = [m.content_for_future for m in cmt.full_context if m.role == "assistant"]
    assert contents[0] == "<action>\ngo north\n</action>"           # history: stripped
    assert contents[1] == LLM_CONTENTS[1]                           # final: intact
    # tokenization matches the chat-template render of the raw messages
    sample = cmt.group_tokenize()[0]
    ref_text = qwen35_tok.apply_chat_template(steps, tokenize=False, add_generation_prompt=False)
    ref_ids = qwen35_tok(ref_text, return_tensors="pt", padding=False)["input_ids"][0].tolist()
    assert sample.input_ids == ref_ids


def test_nonthinking_qwen25_regression(qwen25_tok):
    """strip flag off + Qwen2.5 tokenizer: concat(token_arr) still equals the
    chat template render (legacy non-thinking behavior preserved)."""
    cmt = Linear_CMT(make_config(strip_think=False), qwen25_tok)
    contents = ["<action>\ngo north\n</action>", "<action>\nopen door\n</action>"]
    run_episode(cmt, qwen25_tok, contents, ENV_OBS[:3])
    sample = cmt.group_tokenize()[0]

    msgs = cmt.prepare_previous_context(mod="raw")[:-1]
    ref_text = qwen25_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    ref_ids = qwen25_tok(ref_text, return_tensors="pt", padding=False)["input_ids"][0].tolist()
    assert sample.input_ids == ref_ids

    # header extraction unchanged for Qwen2.5: plain '<|im_start|>assistant\n'
    header = extract_assistant_header_tokens(qwen25_tok)
    assert header == qwen25_tok("<|im_start|>assistant\n")["input_ids"]
    # headers blacked out
    starts = [i for i in range(len(sample.input_ids) - len(header) + 1)
              if sample.input_ids[i:i + len(header)] == header]
    assert len(starts) == 2
    for s in starts:
        assert sample.loss_mask[s:s + len(header)] == [0] * len(header)
        assert sample.loss_mask[s + len(header)] == 1


def test_sampled_turn_always_keeps_forced_think_prefix(qwen35_tok):
    """Every sampled assistant turn must tokenize to the generation prompt's prefix.

    The prompt forces '<|im_start|>assistant\\n<think>\\n'; a turn whose tokens start
    anywhere else is scored under a context the policy never saw. Turns truncated
    before '</think>' used to lose those two tokens, which fed back into the model
    emitting its own '<think>' and never closing it.
    """
    from agentevolver.module.context_manager.cmt_base import (
        auto_tokenize_message,
        extract_assistant_header_tokens,
    )

    header = extract_assistant_header_tokens(qwen35_tok)
    sampled = [
        "<think>\nreasoning\n</think>\n<action>\ngo\n</action>",   # normal
        "<think>\nreasoning that never closes",                     # truncated mid-think
        "<think>\n",                                                # degenerate, empty
    ]
    for content in sampled:
        ids = auto_tokenize_message(qwen35_tok, "assistant", content)
        assert ids[: len(header)] == header, f"prefix lost for {content!r}"

    # history turns are think-stripped and must render verbatim (no forced prefix)
    history = auto_tokenize_message(qwen35_tok, "assistant", "<action>\ngo\n</action>")
    assert history[: len(header)] != header
