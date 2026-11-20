"""Passkey-in-a-book NIAH examples, built token-exactly for one tokenizer.

Each example is

    [book text, P tokens] [needle] [book text, d tokens] [query]

needle (mentions=1):  " The secret code is KEY VALUE."
needle (mentions=2):  " The secret code is KEY VALUE. I repeat, the secret code is KEY VALUE."
query:                "\n\nQuestion: What is the secret code? Answer: The secret code is KEY"

KEY is a single-token city name that does not occur anywhere else in the example; VALUE is one of
20 single-token nouns. The model must put VALUE first among the 20 candidates at the last position
(chance = 5%). P is chosen so the total length is a multiple of 64: the final 64-token window then
starts exactly 64 tokens before the end, so a needle is "inside the window" only when d is small.
Book text comes from PG-19 test books, which no memory was trained on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from pathlib import Path

WINDOW = 64
VALUE_WORDS = ("apple", "river", "table", "horse", "window", "garden", "silver", "candle", "mountain", "bread",
               "doctor", "pencil", "ocean", "tiger", "violin", "carpet", "lemon", "rocket", "blanket", "castle",
               "mirror", "basket", "button", "camera", "dragon", "pillow", "rabbit", "hammer", "orange", "piano")
KEY_WORDS = ("Paris", "London", "Tokyo", "Berlin", "Madrid", "Rome", "Moscow", "Cairo", "Sydney", "Toronto",
             "Chicago", "Boston", "Dallas", "Denver", "Seattle", "Miami", "Atlanta", "Phoenix", "Vienna", "Prague",
             "Dublin", "Oslo", "Lisbon", "Athens", "Warsaw", "Bangkok", "Seoul", "Beijing", "Mumbai", "Delhi",
             "Lagos", "Nairobi", "Lima", "Havana", "Manila", "Jakarta", "Istanbul", "Zurich", "Geneva", "Munich",
             "Hamburg", "Milan", "Venice", "Florence", "Naples", "Houston", "Detroit", "Portland", "Austin", "Ottawa")
N_VALUES = 20
# Words used by the needle and question templates; never used as code words or answers.
TEMPLATE_WORDS = ("The", "the", "secret", "code", "repeat", "Note", "belongs", "Again", "again", "Question", "What",
                  "Which", "word", "that", "Answer", "is", "to")
# Training pools for NIAH-trained memories: no word is shared with the test pools above.
TRAIN_VALUE_WORDS = ("chair", "bottle", "flower", "cloud", "glove", "wheel", "anchor", "feather", "kettle", "ladder",
                     "pepper", "ribbon", "saddle", "shovel", "spoon", "stone", "tower", "wagon", "whistle", "bucket",
                     "cherry", "coffee", "desert", "engine", "forest", "guitar", "island", "jacket", "lamp", "needle")
TRAIN_KEY_WORDS = ("Texas", "Ohio", "Florida", "Kenya", "Brazil", "Canada", "Spain", "Italy", "France", "Germany",
                   "Japan", "China", "India", "Egypt", "Peru", "Chile", "Cuba", "Iran", "Israel", "Korea", "Vietnam",
                   "Nigeria", "Poland", "Sweden", "Norway", "Finland", "Greece", "Turkey", "Mexico", "Arizona", "Nevada",
                   "Oregon", "Utah", "Kansas", "Iowa", "Alaska", "Hawaii", "Virginia", "Carolina", "Michigan", "Ireland",
                   "Scotland", "Wales", "Russia", "Ukraine", "Austria", "Denmark", "Belgium", "Portugal", "Argentina")
assert not set(TRAIN_VALUE_WORDS) & set(VALUE_WORDS) and not set(TRAIN_KEY_WORDS) & set(KEY_WORDS)


@dataclass
class Example:
    uid: str
    tokens: list
    answer: int            # token id of VALUE
    candidates: list       # token ids of the 20 values
    key: int
    distance: int          # book tokens between the needle and the query
    mentions: int
    needle_start: int
    needle_end: int
    book: str


def vocab_pools(tokenizer, exclude=()):
    """Large training pools taken from the tokenizer's own vocabulary: single-token lowercase words (answers)
    and single-token capitalised words (code words), 4-9 letters, excluding every word in `exclude`."""
    import re
    lower, upper = [], []
    banned = {w.lower() for w in tuple(exclude) + TEMPLATE_WORDS}
    for token, index in tokenizer.get_vocab().items():
        text = tokenizer.convert_tokens_to_string([token])
        if not text.startswith(" ") or text[1:].lower() in banned:
            continue
        word = text[1:]
        if re.fullmatch(r"[a-z]{4,9}", word):
            lower.append(word)
        elif re.fullmatch(r"[A-Z][a-z]{3,8}", word):
            upper.append(word)
    return tuple(sorted(lower)), tuple(sorted(upper))


def _single_tokens(tokenizer, words):
    out = []
    for word in words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            out.append((word, ids[0]))
    return out


def build(tokenizer, books_dir, *, distances, mentions=(1, 2), per_cell=100, seed=0, prefix=256, skip=1000,
          value_words=VALUE_WORDS, key_words=KEY_WORDS, fmt="key_first", n_values=N_VALUES):
    """fmt="key_first":   " The secret code is KEY VALUE."  ... "The secret code is KEY" -> VALUE
    fmt="value_first": " Note: VALUE belongs to KEY."    ... "The word that belongs to KEY is" -> VALUE
    n_values=None draws answers from the whole value pool; each example then gets its own 20 candidates
    (the answer plus 19 others), so chance stays 5%."""
    if fmt not in ("key_first", "value_first"):
        raise ValueError("fmt must be key_first or value_first")
    values = _single_tokens(tokenizer, value_words)
    values = values[:n_values] if n_values else values
    keys = _single_tokens(tokenizer, key_words)
    if len(values) < N_VALUES or len(keys) < 20:
        raise ValueError(f"tokenizer gives only {len(values)} single-token values and {len(keys)} keys")
    fixed_candidates = [token for _, token in values] if n_values else None
    books = sorted(Path(books_dir).glob("*.txt"))
    cache = {}
    longest = max(distances) + prefix + WINDOW + 200

    def book_tokens(path):
        if path not in cache:
            text = path.read_text(encoding="utf-8")
            cache[path] = tokenizer.encode(text[: (skip + longest) * 8], add_special_tokens=False)
        return cache[path]

    rng = random.Random(seed)
    examples = []
    for m in mentions:
        for d in distances:
            made = 0
            while made < per_cell:
                r = random.Random(rng.randrange(1 << 30))
                path = r.choice(books)
                ids = book_tokens(path)
                if len(ids) < skip + longest:
                    continue
                start = r.randrange(skip, len(ids) - longest)
                hay = ids[start : start + longest]
                present = set(hay)
                usable = [(w, t) for w, t in keys if t not in present]
                if not usable:
                    continue
                key_word, key = r.choice(usable)
                value_word, value = r.choice(values)
                if value == key:
                    continue
                if fmt == "key_first":
                    sentence = f" The secret code is {key_word} {value_word}."
                    needle_text = sentence if m == 1 else sentence + f" I repeat, the secret code is {key_word} {value_word}."
                    query_text = f"\n\nQuestion: What is the secret code? Answer: The secret code is {key_word}"
                else:
                    sentence = f" Note: {value_word} belongs to {key_word}."
                    needle_text = sentence if m == 1 else sentence + f" Again, {value_word} belongs to {key_word}."
                    query_text = (f"\n\nQuestion: Which word belongs to {key_word}? "
                                  f"Answer: The word that belongs to {key_word} is")
                needle = tokenizer.encode(needle_text, add_special_tokens=False)
                query = tokenizer.encode(query_text, add_special_tokens=False)
                if needle.count(key) != m or needle.count(value) < m or (fmt == "key_first" and query[-1] != key):
                    if fixed_candidates is None:
                        continue            # large vocabulary pools: skip the rare word that merges with its neighbours
                    raise ValueError(f"key/value not single tokens in context: {needle_text!r}")
                if fixed_candidates is None:
                    others = [t for _, t in r.sample(values, 21) if t != value][:19]
                    candidates = sorted(others + [value])
                else:
                    candidates = fixed_candidates
                body = len(needle) + d + len(query)
                p = prefix + (-(prefix + body)) % WINDOW          # total length is a multiple of 64
                tokens = hay[:p] + needle + hay[p : p + d] + query
                assert len(tokens) % WINDOW == 0 and tokens.count(key) == m + query.count(key)
                examples.append(Example(
                    uid=f"m{m}_d{d}_{made}", tokens=tokens, answer=value, candidates=list(candidates), key=key,
                    distance=d, mentions=m, needle_start=p, needle_end=p + len(needle), book=path.name))
                made += 1
    return examples


def score(logits_last, example):
    """logits_last: 1-D tensor over the vocabulary at the final position."""
    import torch
    cand = torch.tensor(example.candidates, device=logits_last.device)
    chosen = example.candidates[int(logits_last[cand].argmax())]
    logp = torch.log_softmax(logits_last.float(), -1)
    return {
        "correct": chosen == example.answer,
        "top1_vocab": int(logits_last.argmax()) == example.answer,
        "logp_answer": float(logp[example.answer]),
        "rank": int((logits_last[cand] > logits_last[example.answer]).sum()) + 1,
    }


def row(example, method, result, **extra):
    record = {k: v for k, v in asdict(example).items() if k not in ("tokens", "candidates")}
    record.update(method=method, length=len(example.tokens), in_window=example.needle_start >= len(example.tokens) - WINDOW)
    record.update(result)
    record.update(extra)
    return record
