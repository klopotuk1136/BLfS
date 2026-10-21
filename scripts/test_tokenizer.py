import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lm.tokenizer import Tokenizer


def read_text(path, max_mb=None):
    with open(path, "rb") as f:
        if max_mb:
            data = f.read(int(max_mb * 1e+6))
            data = data[: data.rfind(b"\n") + 1]
        else:
            data = f.read()
    return data.decode("utf-8")


def evaluate(tokenizer, text):
    n_bytes = len(text.encode("utf-8"))
    t0 = time.perf_counter()
    ids = tokenizer.encode(text)
    seconds = time.perf_counter() - t0
    return n_bytes, len(ids), seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default="saves/vocab_full.pkl")
    ap.add_argument("--merges", default="saves/merges_full.pkl")
    ap.add_argument("--valid", required=True)
    ap.add_argument("--owt", required=True)
    ap.add_argument("--owt-mb-limit", type=float, default=10)
    args = ap.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])

    n_bytes, n_tokens, seconds = evaluate(tokenizer, read_text(args.valid))
    print("Evaluating on TinyStories valid")
    print(f"{n_bytes / 1e6:.1f} MB, {n_tokens} tokens")
    print(f"bytes per token: {n_bytes / n_tokens:.3f}")
    print(f"time to tokenise: {seconds:.1f} s")
    print(f"throughput: {n_bytes / 1e+6 / seconds:.3f} MB/s")
    print("========================================================")

    n_bytes, n_tokens, seconds = evaluate(tokenizer, read_text(args.owt, args.owt_mb_limit))
    print("Evaluating on OWT")
    print(f"{n_bytes / 1e6:.1f} MB, {n_tokens} tokens")
    print(f"bytes per token: {n_bytes / n_tokens:.3f}")
    print(f"time to tokenise: {seconds:.1f} s")
    print(f"throughput: {n_bytes / 1e+6 / seconds:.3f} MB/s")


if __name__ == "__main__":
    main()
