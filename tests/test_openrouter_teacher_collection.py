import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.collect_openrouter_teacher_trajectories import (
    ATTEMPT_SCHEMA_VERSION,
    EXPECTED_TOKENIZER_HASHES,
    SCHEMA_VERSION,
    DecisionTraceRecorder,
    JsonlJournal,
    collection_config,
    compose_student_config,
    expected_curriculum,
    load_and_validate_task_file,
    parse_api_key_from_python,
    scan_attempt_ledger,
    scan_success_output,
    validate_student_contract,
)
from agentevolver.module.teacher.openai_teacher_llm import OpenAITeacherLLM


ROOT = Path(__file__).resolve().parents[1]
ALF_CONFIG = ROOT / (
    "config/duet_paper_experiments_configs/iclr2027/alfworld/"
    "alfworld_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100.yaml"
)
WS_CONFIG = ROOT / (
    "config/duet_paper_experiments_configs/iclr2027/webshop/"
    "webshop_qwen35_4b_grpo_32k_baseline_b16_gpu47_seed2025_s100.yaml"
)


@pytest.mark.parametrize(
    ("environment", "task_file", "expected_sha"),
    [
        (
            "alfworld",
            ROOT / "data/alfworld/task_ids_train1600_seed2026.txt",
            "38373eb25d63affb26f672dbfec83820731b586cc22c1db832a04303b7b58c39",
        ),
        (
            "webshop",
            ROOT / "data/webshop/task_ids_train1600_seed2026.txt",
            "bd235d350a18e2a69bc80281e650f3320d1000e54b622abeb4df800d791a31ac",
        ),
    ],
)
def test_fixed_1600_curriculum_exact(environment, task_file, expected_sha):
    task_ids, manifest = load_and_validate_task_file(
        task_file,
        env_name=environment,
        task_seed=2026,
        expected_count=1600,
    )
    assert len(task_ids) == len(set(task_ids)) == 1600
    assert manifest["ordered_newline_sha256"] == expected_sha
    assert expected_curriculum(environment, 2026, 1600)["task_ids"] == task_ids


def test_legacy_800_is_rejected_as_current_curriculum():
    with pytest.raises(RuntimeError, match="exactly 1600 unique"):
        load_and_validate_task_file(
            ROOT / "data/alfworld/task_ids_800_seed2026.txt",
            env_name="alfworld",
            task_seed=2026,
            expected_count=1600,
        )


@pytest.mark.parametrize(
    ("path", "environment", "recent_turns", "history_cap"),
    [(ALF_CONFIG, "alfworld", 2, 160), (WS_CONFIG, "webshop", 4, 512)],
)
def test_collection_config_is_student_projection(
    path, environment, recent_turns, history_cap
):
    student = compose_student_config(path)
    contract = validate_student_contract(student)
    projected = collection_config(student, "http://127.0.0.1:19999/")

    assert contract["environment"] == environment
    assert contract["prompt_length"] == 22528
    assert contract["response_length"] == 10240
    assert contract["max_model_len"] == 32768
    assert contract["context_management"]["recent_turns"] == recent_turns
    assert (
        contract["context_management"]["history_observation_max_tokens"]
        == history_cap
    )
    assert student.actor_rollout_ref.rollout.context_management.snapshot_training
    assert not projected.actor_rollout_ref.rollout.context_management.snapshot_training
    assert projected.env_service.env_url == "http://127.0.0.1:19999"
    assert projected.env_service.env_params == student.env_service.env_params


def test_api_key_python_source_is_parsed_but_never_executed(tmp_path):
    source = tmp_path / "credential_source.py"
    source.write_text(
        "raise RuntimeError('must not execute')\n"
        "client = object(api_key='sk-test-literal-safe-value')\n",
        encoding="utf-8",
    )
    assert parse_api_key_from_python(source) == "sk-test-literal-safe-value"


def test_api_key_python_source_rejects_ambiguity(tmp_path):
    source = tmp_path / "ambiguous.py"
    source.write_text(
        "a = Client(api_key='sk-first-test-value')\n"
        "b = Client(api_key='sk-second-test-value')\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="exactly one literal"):
        parse_api_key_from_python(source)


class _FakeTokenizer:
    chat_template = "fake-template"

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        rendered = json.dumps(messages, sort_keys=True) + str(
            kwargs.get("add_generation_prompt", False)
        )
        return [ord(char) for char in rendered] if tokenize else rendered

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in str(text)]


class _FakeTeacher:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.content, {
            "finish_reason": "stop",
            "native_finish_reason": None,
            "usage": {"prompt_tokens": 7, "completion_tokens": 5},
            "latency_ms": 12.0,
            "retry_count": 0,
        }


def test_trace_recorder_freezes_exact_context_and_enforces_qwen_limit():
    tokenizer = _FakeTokenizer()
    teacher = _FakeTeacher("<think>x</think><action>go</action>")
    recorder = DecisionTraceRecorder(
        teacher_llm=teacher,
        tokenizer=tokenizer,
        response_token_limit=10,
        temperature=0.9,
        top_p=1.0,
        store_prompt_messages=True,
    )
    messages = [{"role": "user", "content": "state"}]
    output = recorder.chat(messages)
    assert output["finish_reason"] == "length"
    assert output["truncated_by_length"] is True
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    context_result = SimpleNamespace(
        messages=messages,
        prompt_token_ids=prompt_ids,
        raw_prompt_hash="raw-hash",
        stats={"raw_prompt_tokens": 11, "managed_prompt_tokens": 9},
    )
    recorder.observe(
        step_index=0,
        prompt_messages=messages,
        context_result=context_result,
        llm_output=output,
    )
    assert recorder.trace[0]["prompt_messages"] == messages
    assert recorder.trace[0]["length_source"] == "qwen35_retokenization"
    assert teacher.calls[0][1]["max_tokens"] == 10
    assert teacher.calls[0][1]["top_p"] == 1.0


def test_resume_scans_rollout_ids_not_task_ids(tmp_path):
    output = tmp_path / "teacher.jsonl"
    contract = "contract"
    allowed = {"7"}
    records = [
        {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": contract,
            "task_id": "7",
            "rollout_id": "alfworld:7:deepseek-v4-flash:0",
            "success": True,
        },
        {
            "schema_version": SCHEMA_VERSION,
            "contract_sha256": contract,
            "task_id": "7",
            "rollout_id": "alfworld:7:deepseek-v4-flash:1",
            "success": True,
        },
    ]
    output.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    completed = scan_success_output(
        output, contract_sha256=contract, allowed_tasks=allowed
    )
    assert set(completed) == {record["rollout_id"] for record in records}

    attempts = tmp_path / "teacher.attempts.jsonl"
    journal = JsonlJournal(attempts)
    journal.append(
        {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "contract_sha256": contract,
            "event": "attempt_finished",
            "rollout_id": records[0]["rollout_id"],
            "attempt_index": 3,
        }
    )
    assert scan_attempt_ledger(attempts, contract)[records[0]["rollout_id"]] == 4


def test_openai_teacher_metadata_and_single_sdk_retry_layer(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            message = SimpleNamespace(
                content="<action>go</action>",
                reasoning="reason",
                reasoning_content=None,
            )
            choice = SimpleNamespace(
                message=message,
                finish_reason="stop",
                native_finish_reason=None,
                model_extra={"native_finish_reason": "stop"},
                logprobs=None,
            )
            usage = SimpleNamespace(
                model_dump=lambda **_: {
                    "prompt_tokens": 9,
                    "completion_tokens": 4,
                    "total_tokens": 13,
                    "cost": 0.01,
                }
            )
            return SimpleNamespace(
                choices=[choice],
                id="response-id",
                model="returned-model",
                usage=usage,
                model_extra={},
            )

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    teacher = OpenAITeacherLLM(
        model_name="deepseek/deepseek-v4-flash",
        api_base="https://example.invalid/v1",
        api_key="sk-test-only-value",
        max_retries=5,
    )
    content, metadata = teacher(
        [{"role": "user", "content": "x"}], max_tokens=10240, top_p=1.0
    )
    assert captured["client"]["max_retries"] == 0
    assert captured["request"]["model"] == "deepseek/deepseek-v4-flash"
    assert captured["request"]["max_tokens"] == 10240
    assert captured["request"]["top_p"] == 1.0
    assert content.startswith("<think>\nreason\n</think>")
    assert metadata["finish_reason"] == "stop"
    assert metadata["native_finish_reason"] == "stop"
    assert metadata["usage"]["cost"] == 0.01
    assert metadata["response_id"] == "response-id"
    assert EXPECTED_TOKENIZER_HASHES["chat_template.jinja"] == (
        "1bdb2478ddd74a9d051a91e202e370625156ebac9fb68783644340656c54fc00"
    )
