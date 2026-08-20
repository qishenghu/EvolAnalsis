# -*- coding: utf-8 -*-
"""截断稳定化:让"token 切片 → decode 成文字 → 未来 retokenize"的往返可控。

背景(2026-08 deepsearch p400 军规校验排查):
  * `StructuredContextPolicy._clip_text`(context_policy.py L159-180)对超长
    观察按 token 切片后 decode 成文字存入历史;该文字未来会被 chat template
    重新分词。BPE 对任意切点不保证 tokenize(decode(ids[:n])) == ids[:n],
    切在词中/多字节字符中时,重分词结果可能与原切片不同(数量与 id 都可能变)。
  * 注意:p400 那 50 条 prompt sha 失配的**真实根因不是本现象**,而是采集器
    (duet 环境, tokenizers 0.21.2)与 vLLM/校验(vllm2 环境, tokenizers
    0.22.2)对稀有文字(如泰卢固语组合序列 'తి')的分词差异——同一份
    tokenizer.json,不同库版本给出 4 vs 5 个 token。本模块管不了版本漂移,
    只把截断路径的"往返幂等"钉死,消除同类问题的另一个潜在来源。

契约:
  stable_token_truncate / stable_token_truncate_suffix 返回 (text, token_ids,
  stable),保证 stable=True 时在**当前进程的 tokenizer 下**严格满足
      encode(text, add_special_tokens=False) == token_ids 且
      len(token_ids) <= max_tokens。
  找不到稳定切点时(有界回退 + 空白边界兜底都失败)接受原切片并返回
  stable=False,由调用方决定是否记录/隔离。
"""

from __future__ import annotations

from typing import Any, List, NamedTuple, Sequence

# 有界回退步数默认值:切点不稳定几乎总是差 1-2 个 token 的再合并,32 步
# 已远超实际需要,同时为病态输入(如长同字符 run)保住 O(回退步数×编码) 上界。
DEFAULT_MAX_BACKOFF = 32

# 兜底切分认可的"空白"字符:与 BPE 预分词的 word boundary 一致的保守子集。
_WHITESPACE = (" ", "\n", "\t", "\r")


class StableTruncation(NamedTuple):
    """截断结果三元组;text 与 token_ids 在 stable=True 时严格互为编码/解码。"""

    text: str
    token_ids: List[int]
    stable: bool


def _encode(tokenizer: Any, text: str) -> List[int]:
    """与 context_policy._encode_text 同一调用形状(不加特殊 token)。"""
    return list(tokenizer.encode(text, add_special_tokens=False))


def _roundtrip_ok(tokenizer: Any, text: str, token_ids: Sequence[int]) -> bool:
    """幂等校验:该文字重分词后必须逐 id 等于给定切片。"""
    return _encode(tokenizer, text) == list(token_ids)


def _whitespace_fallback_prefix(
    tokenizer: Any, sliced_text: str, max_tokens: int
) -> StableTruncation | None:
    """兜底:回退到最近的空白字符边界,丢掉尾部残词后再验一次。"""
    cut = max(sliced_text.rfind(ch) for ch in _WHITESPACE)
    if cut <= 0:
        return None
    text = sliced_text[:cut]
    ids = _encode(tokenizer, text)
    # 空白边界的候选不再是原切片前缀,幂等要求改为自洽:decode(encode(text))
    # 再编码仍是同一串(排除本身就编码不稳定的病态文字),且不超预算。
    if len(ids) <= max_tokens and _roundtrip_ok(tokenizer, tokenizer.decode(ids), ids):
        return StableTruncation(text=text, token_ids=ids, stable=True)
    return None


def _whitespace_fallback_suffix(
    tokenizer: Any, sliced_text: str, max_tokens: int
) -> StableTruncation | None:
    """后缀版兜底:向后找最近空白,丢掉开头残词后再验一次。"""
    candidates = [sliced_text.find(ch) for ch in _WHITESPACE]
    candidates = [c for c in candidates if c >= 0]
    if not candidates:
        return None
    cut = min(candidates) + 1  # 空白本身留在被丢弃侧
    if cut >= len(sliced_text):
        return None
    text = sliced_text[cut:]
    ids = _encode(tokenizer, text)
    if len(ids) <= max_tokens and _roundtrip_ok(tokenizer, tokenizer.decode(ids), ids):
        return StableTruncation(text=text, token_ids=ids, stable=True)
    return None


def stable_token_truncate(
    tokenizer: Any,
    token_ids: Sequence[int],
    max_tokens: int,
    *,
    max_backoff: int = DEFAULT_MAX_BACKOFF,
) -> StableTruncation:
    """取 token 前缀切片并保证 decode→retokenize 幂等。

    从 max_tokens 起逐步回退(最多 max_backoff 步),找第一个满足
    encode(decode(ids[:n])) == ids[:n] 的 n;回退耗尽则退到最近空白字符
    边界再验一次;仍不稳定则接受 max_tokens 原切片并以 stable=False 标记。
    """
    ids = [int(t) for t in token_ids]
    if max_tokens <= 0:
        return StableTruncation(text="", token_ids=[], stable=True)
    limit = min(len(ids), max_tokens)

    for n in range(limit, max(0, limit - max_backoff), -1):
        candidate = ids[:n]
        text = tokenizer.decode(candidate)
        if _roundtrip_ok(tokenizer, text, candidate):
            return StableTruncation(text=text, token_ids=candidate, stable=True)

    sliced_text = tokenizer.decode(ids[:limit])
    fallback = _whitespace_fallback_prefix(tokenizer, sliced_text, max_tokens)
    if fallback is not None:
        return fallback

    # 接受不稳定切片:调用方拿到的 token_ids 仍是原切片,便于对账/隔离。
    return StableTruncation(text=sliced_text, token_ids=ids[:limit], stable=False)


def stable_token_truncate_suffix(
    tokenizer: Any,
    token_ids: Sequence[int],
    max_tokens: int,
    *,
    max_backoff: int = DEFAULT_MAX_BACKOFF,
) -> StableTruncation:
    """后缀版:取 token 尾部切片并保证 decode→retokenize 幂等。

    与前缀版对称,服务于 _clip_text 的 head+marker+tail 组合里的 tail 段:
    从 max_tokens 起逐步缩短(最多 max_backoff 步),找第一个满足
    encode(decode(ids[-m:])) == ids[-m:] 的 m;兜底与不稳定语义同前缀版。
    """
    ids = [int(t) for t in token_ids]
    if max_tokens <= 0:
        return StableTruncation(text="", token_ids=[], stable=True)
    limit = min(len(ids), max_tokens)

    for m in range(limit, max(0, limit - max_backoff), -1):
        candidate = ids[-m:]
        text = tokenizer.decode(candidate)
        if _roundtrip_ok(tokenizer, text, candidate):
            return StableTruncation(text=text, token_ids=candidate, stable=True)

    sliced_text = tokenizer.decode(ids[-limit:])
    fallback = _whitespace_fallback_suffix(tokenizer, sliced_text, max_tokens)
    if fallback is not None:
        return fallback

    return StableTruncation(text=sliced_text, token_ids=ids[-limit:], stable=False)
