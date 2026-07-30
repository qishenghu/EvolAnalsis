"""For a set of (state_messages, teacher_action) pairs, compute π_θ(a_teacher | s) under a given model.

We use the model's chat template to convert state_messages to a prompt, then measure the
log-probability of the teacher assistant turn string appended after the prompt.

Key probability modes:
  (a) P(full teacher response) — think + action + closing tokens
  (b) P(action-only) — only the `<action>click[...]</action>` substring inside the response
  (c) P(option-token-only) — only the `click[<option_name>]` substring (what we care about for
      the support-gap question: the option identifier itself)

We compute all three for each sample, plus the probability of the *first token* of the action
substring — this is the "leading edge" probability the theory cares about (at the moment the
model decides to emit `click[...]` the key commitment is the first token).

Outputs: JSONL with one line per sample, fields:
  task_id, rollout_id, action, page
  prompt_len
  resp_len                # length of full assistant response including <think>
  p_resp, logp_resp       # geometric mean token prob of response
  p_action_span, logp_action_span    # prob of <action>...</action> span
  p_click_span, logp_click_span      # prob of click[...] span
  p_first_click_token, logp_first    # prob of first click[ token
  p_per_click_token_min              # min per-token prob in click[...] span (bottleneck)
"""
import argparse
import json
import os
import re
import math
from typing import List, Dict, Any, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# An assistant turn in our data looks like:
#   <think>...</think>\n<action>\nclick[155- yellow]\n</action>
# We want to measure prob of various spans.

def pick_spans(assistant_text: str) -> Dict[str, Tuple[int, int]]:
    """Return char-span offsets (start, end) for 'action_span' and 'click_span' inside assistant_text."""
    spans = {}
    m = re.search(r"<action>\s*(.*?)\s*</action>", assistant_text, re.DOTALL)
    if m:
        spans["action_span"] = (m.start(), m.end())
        click_text = m.group(1).strip()
        # locate click_text char span inside assistant_text
        ct_start = assistant_text.find(click_text, m.start(1))
        if ct_start >= 0:
            spans["click_span"] = (ct_start, ct_start + len(click_text))
    return spans


def compute_probs_for_sample(
    model, tokenizer, device, state_messages, teacher_action_str: str,
):
    """Compute various probability statistics for the teacher turn.

    We construct the full conversation (state + teacher_turn), render with chat template (no generation prompt),
    and separately render state with `add_generation_prompt=True`. The difference gives us the
    character boundary between prompt and assistant content.
    """
    # Build assistant content. Our teacher turn was a full assistant message (with <think> and <action>).
    # The 'action' string we stored is the inner content of <action> (e.g. 'click[red]'). We need the
    # full teacher assistant content instead. Reconstruct: the actual assistant message we want is
    # the one at position len(state_messages) in the trajectory. The caller passes the full teacher
    # assistant content via the argument.
    assistant_content = teacher_action_str

    prompt = tokenizer.apply_chat_template(
        state_messages, tokenize=False, add_generation_prompt=True
    )
    full = tokenizer.apply_chat_template(
        state_messages + [{"role": "assistant", "content": assistant_content}],
        tokenize=False, add_generation_prompt=False
    )

    # The assistant content starts right after the prompt in `full` (chat templates usually append
    # the assistant block inside <|im_start|>assistant\n ... <|im_end|>).
    assert full.startswith(prompt), (
        f"Expected full to start with prompt. \nPROMPT END: ...{prompt[-100:]}\nFULL AT PROMPT END: {full[len(prompt)-100:len(prompt)+100]}"
    )
    assistant_block = full[len(prompt):]  # e.g. '<think>...</think>\n<action>click[red]</action><|im_end|>\n'

    # Find spans inside assistant_block
    spans_char = pick_spans(assistant_block)
    # Also span for the whole assistant response (minus trailing <|im_end|>/whitespace)
    end_of_response = len(assistant_block)
    # Trim trailing <|im_end|> etc:
    eos_marker = "<|im_end|>"
    eo_idx = assistant_block.rfind(eos_marker)
    if eo_idx >= 0:
        response_span = (0, eo_idx)
    else:
        response_span = (0, end_of_response)

    # Tokenize full and prompt to get token boundaries
    full_enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
    prompt_enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    n_prompt_tokens = len(prompt_enc["input_ids"])
    full_ids = full_enc["input_ids"]
    offsets = full_enc["offset_mapping"]  # list of (char_start, char_end) in `full`

    # Convert assistant-block char spans to absolute char offsets in `full`
    base = len(prompt)
    def to_full_span(cs):
        return (cs[0] + base, cs[1] + base)

    spans_full_char = {}
    if "action_span" in spans_char:
        spans_full_char["action_span"] = to_full_span(spans_char["action_span"])
    if "click_span" in spans_char:
        spans_full_char["click_span"] = to_full_span(spans_char["click_span"])
    spans_full_char["response_span"] = to_full_span(response_span)

    # Convert char spans to token index ranges: token is 'inside' if any of its chars overlap the span
    def token_range_for_span(span_start, span_end):
        start_tok = None
        end_tok = None
        for i, (a, b) in enumerate(offsets):
            if a == b:
                continue
            if b <= span_start:
                continue
            if a >= span_end:
                break
            if start_tok is None:
                start_tok = i
            end_tok = i + 1
        return start_tok, end_tok

    token_spans = {}
    for k, (cs, ce) in spans_full_char.items():
        ts = token_range_for_span(cs, ce)
        if ts[0] is not None:
            token_spans[k] = ts

    # Forward pass on CPU/GPU with teacher forcing
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[0]  # (T, V)
    # For token at position t (>=1), the prediction distribution was logits[t-1]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_logp = log_probs[:-1].gather(1, input_ids[0, 1:].unsqueeze(-1)).squeeze(-1)  # (T-1,)
    # token_logp[t-1] = logp of token at position t

    # Build per-token logprob, where entry t corresponds to token at position t (we use [t-1] slot)
    def span_logp(start_tok, end_tok):
        # token positions [start_tok, end_tok) in input_ids
        # their logp are stored at indices [start_tok-1, end_tok-1) in token_logp
        if start_tok is None or start_tok <= 0 or end_tok <= start_tok:
            return None
        s = start_tok - 1
        e = end_tok - 1
        return token_logp[s:e].tolist()

    results = {}
    for k in ("response_span", "action_span", "click_span"):
        if k not in token_spans:
            results[f"{k}_logps"] = None
            continue
        lps = span_logp(*token_spans[k])
        results[f"{k}_logps"] = lps

    # Probability of the first token after the prompt (the "commitment" token).
    # That's the token at position n_prompt_tokens in input_ids; its logp at index n_prompt_tokens-1.
    first_resp_idx = n_prompt_tokens
    if first_resp_idx - 1 < token_logp.shape[0]:
        results["first_response_token_logp"] = token_logp[first_resp_idx - 1].item()
        results["first_response_token_str"] = tokenizer.decode([full_ids[first_resp_idx]])
    else:
        results["first_response_token_logp"] = None
        results["first_response_token_str"] = None

    # Probability of first click-span token
    if "click_span" in token_spans:
        st, en = token_spans["click_span"]
        if st - 1 >= 0 and st - 1 < token_logp.shape[0]:
            results["first_click_token_logp"] = token_logp[st - 1].item()
            results["first_click_token_str"] = tokenizer.decode([full_ids[st]])
            # tokens in click span
            results["click_tokens"] = [tokenizer.decode([full_ids[i]]) for i in range(st, en)]
    # Click span-level stats
    if results.get("click_span_logps"):
        lps = results["click_span_logps"]
        results["click_span_mean_logp"] = float(sum(lps) / len(lps))
        results["click_span_min_logp"] = float(min(lps))
        # Probability of the full click[...] sequence (conditioned on prompt only):
        # This is the product of per-token probs (each conditioned on previous), which equals
        # p(click_span | prompt) = exp(sum(logps)) provided the click_span starts right at the
        # first assistant token (which is usually inside <think>, so NOT true in general).
        # So instead we use the mean logp as a per-token fluency proxy.
        results["click_span_prod_logp"] = float(sum(lps))
        results["click_span_len"] = len(lps)

    if results.get("action_span_logps"):
        lps = results["action_span_logps"]
        results["action_span_mean_logp"] = float(sum(lps) / len(lps))
        results["action_span_min_logp"] = float(min(lps))
        results["action_span_len"] = len(lps)

    if results.get("response_span_logps"):
        lps = results["response_span_logps"]
        results["response_span_mean_logp"] = float(sum(lps) / len(lps))
        results["response_span_len"] = len(lps)

    # We also want the probability of the model *starting* to emit '<action>' at all.
    # Heuristic: find the first '<action>' literal in assistant_block and grab logp of its first token.
    # This measures "support of emitting an action at all."
    try:
        action_tag_start_in_block = assistant_block.index("<action>")
        tag_char_start = base + action_tag_start_in_block
        # find token whose (start,end) contains tag_char_start
        tag_tok_idx = None
        for i, (a, b) in enumerate(offsets):
            if a == b: continue
            if a <= tag_char_start < b:
                tag_tok_idx = i; break
            if a >= tag_char_start:
                tag_tok_idx = i; break
        if tag_tok_idx is not None and tag_tok_idx - 1 >= 0 and tag_tok_idx - 1 < token_logp.shape[0]:
            results["action_tag_first_token_logp"] = token_logp[tag_tok_idx - 1].item()
    except ValueError:
        pass

    return results


def load_model(model_path, device):
    print(f"Loading {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--samples", required=True, help="JSONL of state-action pairs")
    p.add_argument("--output", required=True)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tag", default="")
    args = p.parse_args()

    with open(args.samples) as f:
        samples = [json.loads(l) for l in f]
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples from {args.samples}")

    model, tok = load_model(args.model, args.device)

    # Load the full teacher trajectories dict so we can recover the exact teacher assistant_content
    # for each sample (our extraction saved state_messages + inner action string, not the wrapped
    # content). Rebuild by re-loading pickle and indexing.
    import pickle
    TEACHER_PATH = "/data/home/qisheng/EvolAnalsis/data/teacher_trajectories/qwen72b/webshop_qwen72b_filtered.pkl"
    with open(TEACHER_PATH, "rb") as f:
        teacher = pickle.load(f)
    by_rollout = {t["rollout_id"]: t for t in teacher}

    fout = open(args.output, "w")
    n_done = 0
    import time
    t0 = time.time()
    for sample in samples:
        t = by_rollout.get(sample["rollout_id"])
        if t is None:
            continue
        # Find the specific assistant message whose content contains the stored action string
        messages = t["messages"]
        state_len = len(sample["state_messages"])
        if state_len >= len(messages):
            continue
        teacher_turn = messages[state_len]
        if teacher_turn.get("role") != "assistant":
            # Fall back: search
            found = False
            for j in range(state_len, len(messages)):
                if messages[j].get("role") == "assistant" and sample["action"] in messages[j].get("content", ""):
                    teacher_turn = messages[j]; found = True; break
            if not found:
                continue
        assistant_content = teacher_turn["content"]
        try:
            result = compute_probs_for_sample(
                model, tok, args.device,
                state_messages=sample["state_messages"],
                teacher_action_str=assistant_content,
            )
        except Exception as e:
            print(f"Error on sample {sample.get('rollout_id')}: {e}")
            continue

        rec = {
            "tag": args.tag,
            "rollout_id": sample["rollout_id"],
            "task_id": sample["task_id"],
            "action": sample["action"],
            "page": sample["page"],
            "assistant_content_first80": assistant_content[:80],
            **{k: v for k, v in result.items() if k not in (
                "response_span_logps", "action_span_logps", "click_span_logps"
            )},
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        n_done += 1
        if n_done % 20 == 0:
            dt = time.time() - t0
            print(f"[{args.tag}] {n_done}/{len(samples)} done in {dt:.1f}s ({dt/n_done:.2f}s/sample)")
    fout.close()
    print(f"Wrote {n_done} results to {args.output}")


if __name__ == "__main__":
    main()
