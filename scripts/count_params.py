import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.model import count_params, count_flops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--d-ff", type=int, default=1344)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=24414)
    args = ap.parse_args()

    vocab_size, context_length, d_model = args.vocab_size, args.context_length, args.d_model
    config = (vocab_size, context_length, d_model, args.num_layers, args.num_heads, args.d_ff)

    total = count_params(*config)
    no_blocks = count_params(vocab_size, context_length, d_model, 0, args.num_heads, args.d_ff)
    blocks = total - no_blocks
    embedding = vocab_size * d_model
    head = total - count_params(*config, tied=True)

    fwd_flops, train_flops = count_flops(*config, context_length)
    head_flops = 2 * vocab_size * d_model

    tokens_per_step = args.batch_size * context_length
    tokens = args.steps * tokens_per_step

    print(f"Total parameters: {total:,}")
    print(f"Parameters in transformer blocks {blocks:,}, embedding {embedding:,}, head {head:,}, final norm {total - blocks - embedding - head:,}")
    print(f"forward FlOPs per token: {fwd_flops:,} (head {head_flops / fwd_flops:.0%})")
    print(f"training FlOPs per token: {train_flops:.3e}")
    print(f"tokens D: {tokens:.1e}, steps {args.steps:,}, batch {args.batch_size} x {context_length} = {tokens_per_step:,} tokens")
    print(f"training FlOPs: {tokens * train_flops:.2e}")


if __name__ == "__main__":
    main()
