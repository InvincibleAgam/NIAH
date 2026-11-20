"""Export the NIAH question sets used in the TokenBank experiments as JSONL (text + token ids).

Every set is rebuilt with niah_task.build() and exactly the arguments the experiments used, so the questions are
identical to the ones the models saw. Usage:
  python export_niah.py --code DIR --train-books DIR --test-books DIR --out DIR [--qwen-tokenizer NAME]
"""
import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True, help="folder containing niah_task.py")
ap.add_argument("--train-books", required=True, help="folder of PG-19 training books (*.txt)")
ap.add_argument("--test-books", required=True, help="folder of PG-19 test books (*.txt)")
ap.add_argument("--manifest", default=None, help="optional PG-19 manifest with book titles")
ap.add_argument("--out", required=True)
ap.add_argument("--qwen-tokenizer", default=None, help="also export the key-first test set as built with this tokenizer")
args = ap.parse_args()
sys.path.insert(0, args.code)
import niah_task  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
titles = {}
if args.manifest and Path(args.manifest).exists():
    man = json.loads(Path(args.manifest).read_text())
    for split in ("test", "train"):
        titles[split] = {m["file"]: m.get("title") for m in man.get(split, [])}


def book_hash(folder, name):
    return hashlib.md5((Path(folder) / name).read_bytes()).hexdigest()


def export(name, examples, tok, tok_name, fmt, books_dir, split, keep_ids=True):
    path = out / f"{name}.jsonl.gz"
    hashes = {}
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for e in examples:
            t = e.tokens
            q_start = e.needle_end + e.distance
            if e.book not in hashes:
                hashes[e.book] = book_hash(books_dir, e.book)
            rec = {
                "id": f"{name}/{e.uid}",
                "format": fmt,
                "distance": e.distance,
                "mentions": e.mentions,
                "book_file": e.book,
                "book_title": titles.get(split, {}).get(e.book),
                "text": tok.decode(t),
                "needle": tok.decode(t[e.needle_start:e.needle_end]),
                "question": tok.decode(t[q_start:]),
                "answer": tok.decode([e.answer]).strip(),
                "candidates": [tok.decode([c]).strip() for c in e.candidates],
                "tokenizer": tok_name,
                "num_tokens": len(t),
                "needle_token_start": e.needle_start,
                "needle_token_end": e.needle_end,
            }
            if keep_ids:
                rec.update(input_ids=t, answer_id=e.answer, candidate_ids=list(e.candidates))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{name}: {len(examples)} questions, {sum(len(e.tokens) for e in examples):,} tokens -> {path}", flush=True)
    return {"file": path.name, "questions": len(examples), "tokenizer": tok_name, "format": fmt,
            "book_md5": hashes}


gpt2 = AutoTokenizer.from_pretrained("gpt2")
gpt2.model_max_length = int(1e9)
TEST_DIST = [16, 256, 1024, 4096]
TRAIN_DIST = [16, 256, 512, 1024]
info = {}

# 1) Key-first test set: every zero-shot NIAH evaluation (GPT-2 and Qwen) and the NIAH-only GPT-2 runs.
ex = niah_task.build(gpt2, args.test_books, distances=TEST_DIST, per_cell=100, seed=0, fmt="key_first")
info["test_key_first_gpt2"] = export("test_key_first_gpt2", ex, gpt2, "gpt2", "key_first", args.test_books, "test")

# 2) Key-first training set (GPT-2 trained on NIAH only: train_niah_gpt2.py, seed 0, 1,000 per cell = 8,000).
ex = niah_task.build(gpt2, args.train_books, distances=TRAIN_DIST, per_cell=1000, seed=1000,
                     value_words=niah_task.TRAIN_VALUE_WORDS, key_words=niah_task.TRAIN_KEY_WORDS,
                     fmt="key_first", n_values=niah_task.N_VALUES)
info["train_key_first_gpt2"] = export("train_key_first_gpt2", ex, gpt2, "gpt2", "key_first", args.train_books, "train")

# 3-6) Value-first sets (folded TokenBank, train_niah_folded.py --seed 42; repair stage --seed 7).
all_values, all_keys = niah_task.vocab_pools(gpt2, exclude=niah_task.VALUE_WORDS + niah_task.KEY_WORDS)
train_values = tuple(sorted(random.Random(0).sample(all_values, 100)))
ex = niah_task.build(gpt2, args.train_books, distances=TRAIN_DIST, per_cell=750, seed=1042,
                     value_words=train_values, key_words=all_keys, fmt="value_first", n_values=None)
info["train_value_first_gpt2"] = export("train_value_first_gpt2", ex, gpt2, "gpt2", "value_first", args.train_books, "train")
ex = niah_task.build(gpt2, args.train_books, distances=TRAIN_DIST, per_cell=187, seed=1007,
                     value_words=train_values, key_words=all_keys, fmt="value_first", n_values=None)
info["train_value_first_repair_gpt2"] = export("train_value_first_repair_gpt2", ex, gpt2, "gpt2", "value_first",
                                               args.train_books, "train")
ex = niah_task.build(gpt2, args.test_books, distances=TEST_DIST, per_cell=100, seed=0, value_words=train_values,
                     key_words=niah_task.KEY_WORDS, fmt="value_first", n_values=None)
info["test_value_first_seen_gpt2"] = export("test_value_first_seen_gpt2", ex, gpt2, "gpt2", "value_first",
                                            args.test_books, "test")
ex = niah_task.build(gpt2, args.test_books, distances=TEST_DIST, per_cell=100, seed=0, fmt="value_first")
info["test_value_first_unseen_gpt2"] = export("test_value_first_unseen_gpt2", ex, gpt2, "gpt2", "value_first",
                                              args.test_books, "test")

# 7) Key-first test set as built with the Qwen tokenizer (the Qwen3-4B evaluations).
if args.qwen_tokenizer:
    qwen = AutoTokenizer.from_pretrained(args.qwen_tokenizer)
    qwen.model_max_length = int(1e9)
    ex = niah_task.build(qwen, args.test_books, distances=TEST_DIST, per_cell=100, seed=0, fmt="key_first")
    info["test_key_first_qwen3"] = export("test_key_first_qwen3", ex, qwen, args.qwen_tokenizer, "key_first",
                                          args.test_books, "test")

(out / "word_pools.json").write_text(json.dumps({
    "test_value_words": list(niah_task.VALUE_WORDS), "test_key_words": list(niah_task.KEY_WORDS),
    "train_value_words_key_first": list(niah_task.TRAIN_VALUE_WORDS),
    "train_key_words_key_first": list(niah_task.TRAIN_KEY_WORDS),
    "train_value_words_value_first": list(train_values),
    "template_words_excluded": list(niah_task.TEMPLATE_WORDS)}, indent=1))
(out / "sets.json").write_text(json.dumps(info, indent=1))
print("DONE", flush=True)
