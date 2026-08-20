# -*- coding: utf-8 -*-
"""stable_truncation 的幂等契约测试。

两层验证:
  1. 真实 tokenizer(Qwen3.5-4B)+ 真实语料:语料节选自 deepsearch p400
     line 72(rollout deepsearch:musique_train_19198:student:0)decision 1 的
     BM25 观察——正是军规校验 50 条 prompt sha 失配的触发文本,含两处
     泰卢固语 'జ్యోతి'(tokenizers 0.21.2 与 0.22.2 对 'తి' 分词不同,
     版本漂移根因的回归锚点)。验证任意/全部切点截断后 100% 幂等。
  2. 病态假 tokenizer:确定性地触发有界回退、空白边界兜底、不稳定标记
     三条路径(真实 BPE 上难以稳定复现这些分支)。

运行(duet 环境 python):
    /projects_vol/gp_wangwy/qisheng/duet_h200/conda/envs/duet/bin/python \
        -m pytest tests/test_stable_truncation.py -v
"""

import os

import pytest

from agentevolver.module.context_manager.stable_truncation import (
    StableTruncation,
    stable_token_truncate,
    stable_token_truncate_suffix,
)

# ---------------------------------------------------------------------------
# 真实语料(p400 line72 obs0 节选;含两处 జ్యోతి)
# ---------------------------------------------------------------------------
REAL_BM25_EXCERPT = (
    "Search results for 'Jyothi performer':\n\n"
    "[Doc 1] Jyothi (1976 film)\n"
    'Jyothi (1976 film) Jyothi (Telugu: జ్యోతి) is a 1976 Telugu film directed '
    'by K. Raghavendra Rao. It is based on the story of an innocent girl '
    '""Jyothi."" Jayasudha won the Nandi award as Best Actress for her '
    "performance in the title role.\n\n"
    "[Doc 3] Jyothi (1976 film)\n"
    "The secret behind this unexpected marriage forms the rest of the gripping "
    "family story. Jyothi (1976 film) Jyothi (Telugu: జ్యోతి) is a 1976 Telugu "
    "film direct"
)

# 候选 tokenizer 路径(不同服务器布局;都缺失则跳过真实语料层)
_TOKENIZER_CANDIDATES = [
    os.environ.get("STABLE_TRUNC_TOKENIZER", ""),
    "/projects_vol/gp_wangwy/models/Qwen3.5-4B",
    "/data/shared_models/Qwen3.5-4B-think",
]


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    for path in _TOKENIZER_CANDIDATES:
        if path and os.path.isdir(path):
            return AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, local_files_only=True
            )
    pytest.skip("Qwen3.5 tokenizer not found on this host")


def _assert_contract(result: StableTruncation, tokenizer, max_tokens: int) -> None:
    """stable=True 时的硬契约:文字重分词逐 id 等于返回切片,且不超预算。"""
    assert len(result.token_ids) <= max_tokens
    if result.stable:
        assert (
            list(tokenizer.encode(result.text, add_special_tokens=False))
            == result.token_ids
        )


# ---------------------------------------------------------------------------
# 第 1 层:真实 tokenizer + 真实失配语料
# ---------------------------------------------------------------------------
class TestRealTokenizerRealCorpus:
    def test_telugu_word_roundtrip_pair(self, qwen_tokenizer):
        """回归锚点:切点落在 జ్యోతి 一词之后,返回对必须自洽。

        注意该词的 token 数依 tokenizers 库版本而异(0.21.2: 'తి' 一个
        token;0.22.2: 'త'+'ి' 两个)——这正是 p400 军规失配的根因。
        本模块不消除版本漂移,只保证当前进程内返回对严格自洽。
        """
        ids = qwen_tokenizer.encode(
            "Jyothi (Telugu: జ్యోతి) is a 1976 Telugu film",
            add_special_tokens=False,
        )
        for n in range(1, len(ids) + 1):
            result = stable_token_truncate(qwen_tokenizer, ids, n)
            _assert_contract(result, qwen_tokenizer, n)
            assert result.stable, f"unexpected unstable cut at n={n}"

    def test_all_cut_points_prefix(self, qwen_tokenizer):
        """全切点性质测试:任意预算下截断结果 100% 幂等(或显式标记)。"""
        ids = qwen_tokenizer.encode(REAL_BM25_EXCERPT, add_special_tokens=False)
        assert len(ids) > 100  # 语料足够长,覆盖 Doc 边界/引号/泰卢固语区
        unstable = []
        for n in range(1, len(ids) + 1):
            result = stable_token_truncate(qwen_tokenizer, ids, n)
            _assert_contract(result, qwen_tokenizer, n)
            if not result.stable:
                unstable.append(n)
        # 该语料在 tokenizers 0.21.2 / 0.22.2 下实测全切点幂等;若未来版本
        # 出现不稳定切点,契约仍由 _assert_contract 保住,这里显式暴露。
        assert unstable == [], f"unstable cut points appeared: {unstable[:10]}"

    def test_all_cut_points_suffix(self, qwen_tokenizer):
        """后缀版(服务 _clip_text 的 tail 段)同样全切点幂等。"""
        ids = qwen_tokenizer.encode(REAL_BM25_EXCERPT, add_special_tokens=False)
        unstable = []
        for m in range(1, len(ids) + 1):
            result = stable_token_truncate_suffix(qwen_tokenizer, ids, m)
            _assert_contract(result, qwen_tokenizer, m)
            if not result.stable:
                unstable.append(m)
        assert unstable == [], f"unstable suffix cuts appeared: {unstable[:10]}"

    def test_budget_zero_and_full(self, qwen_tokenizer):
        ids = qwen_tokenizer.encode(REAL_BM25_EXCERPT, add_special_tokens=False)
        assert stable_token_truncate(qwen_tokenizer, ids, 0) == ("", [], True)
        full = stable_token_truncate(qwen_tokenizer, ids, len(ids) + 100)
        assert full.stable and full.token_ids == list(ids)


# ---------------------------------------------------------------------------
# 第 2 层:病态假 tokenizer,确定性触发回退/兜底/不稳定三条路径
# ---------------------------------------------------------------------------
class PathologicalTokenizer:
    """字符级 tokenizer,附加两条破坏前后缀幂等的合并规则:

    * 末尾的 'a' 连跑(长度>=2)合并为单 token 100000+len;
    * 开头的 'b' 连跑(长度>=2)合并为单 token 200000+len。

    因此 decode(切片) 一旦以 'a' 连跑收尾(或以 'b' 连跑开头),重分词就
    与原切片不同——模拟 BPE 切在词中的再合并现象,且完全确定。
    """

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002
        del add_special_tokens
        head_b = 0
        while head_b < len(text) and text[head_b] == "b":
            head_b += 1
        tail_a = 0
        while tail_a < len(text) - head_b and text[len(text) - 1 - tail_a] == "a":
            tail_a += 1
        ids = []
        if head_b >= 2:
            ids.append(200000 + head_b)
            body_start = head_b
        else:
            body_start = 0
        body_end = len(text) - tail_a if tail_a >= 2 else len(text)
        ids.extend(ord(ch) for ch in text[body_start:body_end])
        if tail_a >= 2:
            ids.append(100000 + tail_a)
        return ids

    def decode(self, ids):
        parts = []
        for token_id in ids:
            if token_id >= 200000:
                parts.append("b" * (token_id - 200000))
            elif token_id >= 100000:
                parts.append("a" * (token_id - 100000))
            else:
                parts.append(chr(token_id))
        return "".join(parts)


class TestPathologicalPaths:
    def setup_method(self):
        self.tok = PathologicalTokenizer()

    def test_bounded_backoff_finds_stable_point(self):
        # 原文以 'z' 结尾 → 整串是逐字符 id;前缀切在 a 连跑内则不幂等,
        # 回退应一路走到 'za'(n=2)处稳定。
        text = "z" + "a" * 10 + "z"
        ids = self.tok.encode(text)
        result = stable_token_truncate(self.tok, ids, 8, max_backoff=32)
        assert result.stable
        assert result.text == "za"
        assert result.token_ids == ids[:2]

    def test_whitespace_fallback(self):
        # 回退窗口(8 步)耗尽仍在 a 连跑内 → 兜底退到空白边界 'hello'。
        text = "hello " + "a" * 40 + "z"
        ids = self.tok.encode(text)
        result = stable_token_truncate(self.tok, ids, 30, max_backoff=8)
        assert result.stable
        assert result.text == "hello"
        assert result.token_ids == self.tok.encode("hello")

    def test_unstable_marker(self):
        # 无空白可兜底 → 接受原切片并明确标记不稳定。
        text = "a" * 40 + "z"
        ids = self.tok.encode(text)
        result = stable_token_truncate(self.tok, ids, 30, max_backoff=4)
        assert not result.stable
        assert result.token_ids == ids[:30]
        # 不稳定标记语义:重分词确实与切片不同(这就是被标记的原因)
        assert self.tok.encode(result.text) != result.token_ids

    def test_suffix_backoff(self):
        # 后缀切片以 'b' 连跑开头则不幂等,回退到 'bz'(m=2)处稳定。
        text = "z" + "b" * 10 + "z"
        ids = self.tok.encode(text)
        result = stable_token_truncate_suffix(self.tok, ids, 8, max_backoff=32)
        assert result.stable
        assert result.text == "bz"
        assert result.token_ids == ids[-2:]

    def test_suffix_whitespace_fallback_and_unstable(self):
        # 兜底:向后找到空白,保留 'world'
        text = "z" + "b" * 40 + " world"
        ids = self.tok.encode(text)
        result = stable_token_truncate_suffix(self.tok, ids, 30, max_backoff=4)
        assert result.stable
        assert result.text == "world"
        # 无空白 → 不稳定标记
        text2 = "z" + "b" * 40 + "z"
        ids2 = self.tok.encode(text2)
        result2 = stable_token_truncate_suffix(self.tok, ids2, 30, max_backoff=4)
        assert not result2.stable
        assert result2.token_ids == ids2[-30:]

    def test_budget_edge_cases(self):
        ids = self.tok.encode("plain text")
        assert stable_token_truncate(self.tok, ids, 0) == ("", [], True)
        assert stable_token_truncate(self.tok, ids, -3) == ("", [], True)
        full = stable_token_truncate(self.tok, ids, 999)
        assert full.stable and full.token_ids == ids
