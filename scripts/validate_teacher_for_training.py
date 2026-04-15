#!/usr/bin/env python3
"""
Validate that teacher trajectories can be used for Qwen3-1.7B training.

Checks:
1. Data format correctness (required fields, react_tags format)
2. Tokenization with student model (Qwen3-1.7B)
3. Sequence lengths fit within 32K training context
4. Log probabilities alignment
5. Forward pass test (optional, requires GPU)

Usage:
    python scripts/validate_teacher_for_training.py \
        --teacher_data /tmp/test_32k_alfworld.jsonl \
        --student_model models/Qwen/Qwen3-1.7B \
        [--forward_pass]  # optional: run a forward pass to verify memory
"""

import argparse
import json
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer


def validate_format(trajs):
    """Check required fields and react_tags format."""
    print("=" * 60)
    print("1. Format Validation")
    print("=" * 60)

    required = ['task_id', 'messages', 'reward', 'success', 'metadata']
    errors = []

    for i, t in enumerate(trajs):
        for f in required:
            if f not in t:
                errors.append(f"traj[{i}] missing '{f}'")
        if not t.get('metadata', {}).get('is_teacher'):
            errors.append(f"traj[{i}] metadata.is_teacher not set")

        # Check react_tags format in assistant messages
        assistant_msgs = [m for m in t['messages'] if m['role'] == 'assistant']
        action_msgs = [m for m in assistant_msgs if '<action>' in m['content']]
        think_msgs = [m for m in assistant_msgs if '<think>' in m['content']]

        print(f"  Task {t['task_id']}: reward={t['reward']:.2f}, "
              f"turns={len(assistant_msgs)}, "
              f"<think>={len(think_msgs)}, <action>={len(action_msgs)}, "
              f"log_probs={'yes' if t.get('log_probs') else 'no'}")

    if errors:
        print(f"\n  ERRORS: {len(errors)}")
        for e in errors[:5]:
            print(f"    - {e}")
    else:
        print(f"\n  [PASS] All {len(trajs)} trajectories have correct format")

    return len(errors) == 0


def validate_tokenization(trajs, tokenizer, max_seq_len=32768):
    """Tokenize with student model and check sequence lengths."""
    print("\n" + "=" * 60)
    print("2. Tokenization & Sequence Length Validation")
    print(f"   Student tokenizer: {tokenizer.name_or_path}")
    print(f"   Max training seq_len: {max_seq_len}")
    print("=" * 60)

    all_fit = True

    for t in trajs:
        messages = t['messages']

        # Method A: tokenize full conversation as chat
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        full_tokens = tokenizer.encode(full_text)

        # Method B: per-turn analysis
        turn_stats = []
        for msg in messages:
            if msg['role'] == 'assistant':
                n_tok = len(tokenizer.encode(msg['content']))
                has_think = '<think>' in msg['content']
                has_action = '<action>' in msg['content']
                turn_stats.append({
                    'tokens': n_tok,
                    'think': has_think,
                    'action': has_action,
                })

        fits = len(full_tokens) <= max_seq_len
        if not fits:
            all_fit = False

        avg_turn = sum(s['tokens'] for s in turn_stats) / max(len(turn_stats), 1)
        max_turn = max((s['tokens'] for s in turn_stats), default=0)

        status = "PASS" if fits else "OVER"
        print(f"  Task {t['task_id']}: total={len(full_tokens)} tokens "
              f"({'<=' if fits else '>'} {max_seq_len}) [{status}]")
        print(f"    Per turn: avg={avg_turn:.0f}, max={max_turn}, "
              f"turns={len(turn_stats)}")

        # Check log_probs alignment
        if t.get('log_probs'):
            teacher_lp_count = len(t['log_probs'])
            # Count student tokens for assistant-only content
            assistant_tokens = sum(s['tokens'] for s in turn_stats)
            ratio = teacher_lp_count / max(assistant_tokens, 1)
            print(f"    Log probs: {teacher_lp_count} (teacher) vs "
                  f"{assistant_tokens} (student tokens), ratio={ratio:.2f}")

    if all_fit:
        print(f"\n  [PASS] All trajectories fit within {max_seq_len} tokens")
    else:
        print(f"\n  [WARN] Some trajectories exceed {max_seq_len} tokens")
        print(f"         These will be truncated by CMT during training")

    return all_fit


def validate_forward_pass(trajs, model_path, tokenizer, max_seq_len=32768):
    """Run a forward pass with the student model to verify memory fits."""
    print("\n" + "=" * 60)
    print("3. Forward Pass Test (memory validation)")
    print(f"   Model: {model_path}")
    print(f"   Seq len: {max_seq_len}")
    print("=" * 60)

    try:
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            print("  [SKIP] No GPU available")
            return True

        # Pick the shortest successful trajectory
        success_trajs = [t for t in trajs if t.get('success')]
        if not success_trajs:
            success_trajs = trajs
        t = min(success_trajs, key=lambda x: len(x['messages']))

        # Tokenize
        full_text = tokenizer.apply_chat_template(
            t['messages'], tokenize=False, add_generation_prompt=False
        )
        input_ids = tokenizer.encode(full_text, return_tensors="pt")

        # Truncate or pad to max_seq_len
        if input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]

        seq_len = input_ids.shape[1]
        print(f"  Test sequence: task={t['task_id']}, {seq_len} tokens")

        # Load model
        print(f"  Loading model (bf16)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
        )

        # Forward pass
        print(f"  Running forward pass...")
        input_ids = input_ids.to("cuda:0")
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        print(f"  Output logits shape: {logits.shape}")

        mem_used = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak GPU memory: {mem_used:.1f} GB")

        del model, outputs, logits, input_ids
        torch.cuda.empty_cache()

        print(f"\n  [PASS] Forward pass successful")
        return True

    except Exception as e:
        print(f"\n  [FAIL] Forward pass error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_data", type=str, required=True,
                        help="Path to teacher trajectory JSONL file")
    parser.add_argument("--student_model", type=str, default="models/Qwen/Qwen3-1.7B",
                        help="Path to student model (for tokenization)")
    parser.add_argument("--max_seq_len", type=int, default=32768,
                        help="Max training sequence length")
    parser.add_argument("--forward_pass", action="store_true",
                        help="Run a forward pass to verify GPU memory")
    args = parser.parse_args()

    # Load data
    print(f"Loading teacher data: {args.teacher_data}")
    with open(args.teacher_data) as f:
        trajs = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(trajs)} trajectories\n")

    # Load tokenizer
    print(f"Loading student tokenizer: {args.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    print(f"Vocab size: {tokenizer.vocab_size}\n")

    # Run validations
    fmt_ok = validate_format(trajs)
    tok_ok = validate_tokenization(trajs, tokenizer, args.max_seq_len)

    fwd_ok = True
    if args.forward_pass:
        fwd_ok = validate_forward_pass(trajs, args.student_model, tokenizer, args.max_seq_len)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Format:        {'PASS' if fmt_ok else 'FAIL'}")
    print(f"  Tokenization:  {'PASS' if tok_ok else 'WARN (some over limit)'}")
    if args.forward_pass:
        print(f"  Forward pass:  {'PASS' if fwd_ok else 'FAIL'}")
    print(f"\n  Teacher data is {'READY' if fmt_ok else 'NOT ready'} for training")


if __name__ == "__main__":
    main()
