import time
from tokenizer import count_pretokens, SPECIAL_TOKENS

with open("data/tinystories_train.txt", encoding="utf-8") as f:
    sample = f.read(20_000_000)
with open("data/ts_20mb.txt", "w", encoding="utf-8") as f:
    f.write(sample)


t0 = time.perf_counter()
counts = count_pretokens(sample, SPECIAL_TOKENS)
print(f"pre-tokenization: {time.perf_counter() - t0:.1f} s, {len(counts)} distinct pre-tokens")