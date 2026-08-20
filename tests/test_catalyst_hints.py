"""T6:hint 清洗管线回归 + 注入模板 pin + 注入/剥除往返。

真实 fixture(tests/fixtures/catalyst/hint_fixtures.json)来自试点原始数据:
  * alfworld 1215 — 基线(纯 think 拼接);
  * webshop 3186 — flash 草稿块剥除 + 5000 字符截断;
  * webshop 3249 — 粘连句界修复(".The" → ". The")。
期望值即试点 hints 文件的 raw 字段(逐字节)。
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from agentevolver.module.exp_manager.catalyst import (
    HINT_MARKER,
    HINT_MAX_CHARS,
    HINT_TEMPLATE_SHA256_PIN,
    build_hint_from_v2_record,
    hint_template_parts,
    inject_hint_into_init_messages,
    load_hint_template,
    strip_hint_messages,
)

FIXTURES = Path(__file__).parent / "fixtures" / "catalyst" / "hint_fixtures.json"


def test_hint_template_matches_pilot_pin():
    template = load_hint_template()
    assert (
        hashlib.sha256(template.encode("utf-8")).hexdigest()
        == HINT_TEMPLATE_SHA256_PIN
    )
    prefix, suffix = hint_template_parts()
    assert HINT_MARKER in prefix
    assert template == prefix + "{hint}" + suffix


@pytest.mark.parametrize(
    "fixture", json.loads(FIXTURES.read_text(encoding="utf-8")),
    ids=lambda f: f"{f['environment']}_{f['task_id']}",
)
def test_pilot_fixture_byte_identical(fixture):
    built = build_hint_from_v2_record(fixture["record"])
    assert built == fixture["expected_raw"]


def test_flash_draft_block_removed():
    record = {
        "decision_trace": [
            {
                "completion_content": (
                    "<think>\nLet me search.\nresponse<action>\nsearch[bed]\n"
                    "</action>\nLooking at results now.\n</think>\n"
                    "<action>\nsearch[bed]\n</action>"
                )
            }
        ]
    }
    assert build_hint_from_v2_record(record) == "Let me search.\n\nLooking at results now."


def test_glued_boundary_fixed():
    record = {
        "decision_trace": [
            {"completion_content": "<think>Go back.The user wants white.</think>"}
        ]
    }
    assert build_hint_from_v2_record(record) == "Go back. The user wants white."


def test_cap_at_5000_chars():
    record = {
        "decision_trace": [
            {"completion_content": "<think>" + "x" * 9000 + "</think>"}
        ]
    }
    built = build_hint_from_v2_record(record)
    assert len(built) == HINT_MAX_CHARS


def test_missing_think_returns_none():
    record = {
        "decision_trace": [
            {"completion_content": "<action>\ngo north\n</action>"}
        ]
    }
    assert build_hint_from_v2_record(record) is None
    assert build_hint_from_v2_record({"decision_trace": []}) is None


def test_inject_then_strip_roundtrip():
    init_messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Task: do a thing."},
    ]
    original = copy.deepcopy(init_messages)
    hinted = inject_hint_into_init_messages(init_messages, "check the sink")
    # 注入不改原对象(deepcopy 语义,试点同款)
    assert init_messages == original
    # 注入目标 = 首条 user 消息末尾,模板逐字节
    template = load_hint_template()
    assert hinted[2]["content"] == original[2]["content"] + template.format(
        hint="check the sink"
    )
    assert hinted[0] == original[0] and hinted[1] == original[1]
    # 剥除往返:恢复原文,且断言恰剥 1 条
    strip_hint_messages(hinted)
    assert hinted == original


def test_strip_asserts_exactly_one():
    with pytest.raises(ValueError):
        strip_hint_messages([{"role": "user", "content": "no hint here"}])
    doubled = inject_hint_into_init_messages(
        [{"role": "user", "content": "a"}], "h1"
    ) + inject_hint_into_init_messages([{"role": "user", "content": "b"}], "h2")
    with pytest.raises(ValueError):
        strip_hint_messages(doubled)


def test_inject_requires_user_message():
    with pytest.raises(RuntimeError):
        inject_hint_into_init_messages(
            [{"role": "system", "content": "sys"}], "h"
        )
    with pytest.raises(RuntimeError):
        inject_hint_into_init_messages(
            [{"role": "user", "content": "task"}], "   "
        )
