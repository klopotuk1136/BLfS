#!/usr/bin/env python3
"""Train the BPE tokenizer and record the Part 1 numbers.

    python scripts/train_tokenizer.py --input data/tinystories_train.txt \
        --valid data/tinystories_valid.txt --owt data/owt_sample.txt --out tok
"""
import argparse
import json
import os
import time
import psutil

from lm.tokenizer import Tokenizer, train_and_save_bpe

def peak_memory_gb() -> float:
    memory = psutil.Process().memory_info().peak_wset
    memory_gb = memory / 1e+9
    return memory_gb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="saves")
    ap.add_argument("--vocab-size", type=int, default=10_000)
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = ap.parse_args()

    t0 = time.perf_counter()
    vocab, merges = train_and_save_bpe(args.input, args.vocab_size, args.special, args.out)
    print(f"whole train: {time.perf_counter() - t0:.1f} s")
    print(f"peak memory: {peak_memory_gb():.2f} GB")

    longest_token = sorted(vocab.values(), key=len, reverse=True)[0]
    first_merges = merges[:20]

    print(f"Longest token is {longest_token}")
    print(f"first merges are {first_merges}")


if __name__ == "__main__":
    main()