# NIAH-Dataset

The Needle-in-a-Haystack (NIAH) question sets used in the TokenBank experiments. Each question hides one short fact (the
*needle*) inside real book text (the *haystack*) and then asks for it after `d` more tokens of book text. The model
must pick the right answer word out of 20 candidates, so chance is 5%.

All sets were regenerated with the original generator (`code/niah_task.py`) and the exact arguments and seeds used in
the experiments. The test sets were checked against the per-question result files written during the experiments:
**800/800 identical** for every test set.

## Formats

**Key-first** (`"The secret code is KEY VALUE."`):

```
[book text] " The secret code is Ottawa silver." [d tokens of book text]
"\n\nQuestion: What is the secret code? Answer: The secret code is Ottawa"   -> answer: "silver"
```

**Value-first** (`"Note: VALUE belongs to KEY."`), where the answer comes before its key in the needle:

```
[book text] " Note: eaten belongs to Ottawa." [d tokens of book text]
"\n\nQuestion: Which word belongs to Ottawa? Answer: The word that belongs to Ottawa is"   -> answer: "eaten"
```

**Construction rules:**
- **Mentions:** with 2 mentions, a second sentence repeats the fact: " I repeat, the secret code is KEY VALUE." for key-first, or " Again, VALUE belongs to KEY." for value-first.
- **Single tokens:** keys and values are single tokens of the tokenizer.
- **Unique key:** the key appears nowhere else in that stretch of book text.
- **Length:** each question is a multiple of 64 tokens long, so a 64-token sliding window ends exactly at the question.
- **Haystack:** each haystack is drawn from a random position in a randomly chosen book, skipping the first 1,000 tokens (the Project Gutenberg front matter).

## Files (`data/`)

| File | Questions | Distances `d` (tokens) | Mentions | Answer words | Key words | Books | Used for |
|---|---|---|---|---|---|---|---|
| `test_key_first_gpt2.jsonl.gz` | 800 | 16, 256, 1,024, 4,096 | 1, 2 | 20 fixed nouns | 50 city names | PG-19 test | All zero-shot NIAH evaluations; evaluation of the NIAH-only GPT-2 models |
| `test_key_first_qwen3.jsonl.gz` | 800 | same | 1, 2 | same | same | PG-19 test | The same set built with the Qwen3 tokenizer (Qwen3-4B evaluations) |
| `train_key_first_gpt2.jsonl.gz` | 8,000 | 16, 256, 512, 1,024 | 1, 2 | 20 training nouns | 50 training place names | PG-19 train | Training GPT-2 on NIAH only |
| `train_value_first_gpt2.jsonl.gz` | 6,000 | 16, 256, 512, 1,024 | 1, 2 | 100 training words | 8,323 vocabulary words | PG-19 train | Training the folded TokenBank (main NIAH result) |
| `train_value_first_repair_gpt2.jsonl.gz` | 1,496 | 16, 256, 512, 1,024 | 1, 2 | 100 training words | 8,323 vocabulary words | PG-19 train | Second training stage of the folded TokenBank |
| `test_value_first_seen_gpt2.jsonl.gz` | 800 | 16, 256, 1,024, 4,096 | 1, 2 | the 100 training words | 50 city names | PG-19 test | Folded TokenBank test, answer words seen in training |
| `test_value_first_unseen_gpt2.jsonl.gz` | 800 | 16, 256, 1,024, 4,096 | 1, 2 | 20 new nouns | 50 city names | PG-19 test | Folded TokenBank test, answer words never seen in training |

Each test set has 100 questions per (distance, mentions) cell. With `d = 16` the needle is still inside the last
64-token window, which serves as a sanity check; `d ≥ 256` is "beyond the window".

**Other files:**
- `data/word_pools.json`: every word pool. Answer and key words used in testing never appear in training.
- `data/sets.json`: for each set, the md5 of every source book.

**Value-first candidates.** In the value-first sets, the 20 candidates for each question are the answer plus 19 other
single GPT-2 tokens from the same pool. Some of these are word pieces (for example `sche`, `narr`).

## Record fields

One JSON object per line:

| Field | Meaning |
|---|---|
| `id` | `<set>/m<mentions>_d<distance>_<index>` |
| `format` | `key_first` or `value_first` |
| `distance` | tokens of book text between the needle and the question |
| `mentions` | 1 or 2 |
| `book_file` | source book file (`000.txt` …) |
| `text` | the full model input as text: haystack + needle + haystack + question |
| `needle`, `question` | the needle sentence(s) and the question, as text |
| `answer` | the correct answer word |
| `candidates` | the 20 candidate answer words (chance = 5%) |
| `tokenizer` | the tokenizer the set was built with (`gpt2` or `Qwen/Qwen3-4B-Instruct-2507`) |
| `num_tokens` | input length in tokens (a multiple of 64) |
| `needle_token_start`, `needle_token_end` | needle position in `input_ids` |
| `input_ids`, `answer_id`, `candidate_ids` | the exact token ids the models saw |

**Scoring:** the model's next-token logits at the end of the input are compared over `candidate_ids`. The question is
correct if `answer_id` scores highest.

```python
import gzip, json
questions = [json.loads(line) for line in gzip.open("data/test_value_first_seen_gpt2.jsonl.gz", "rt")]
q = questions[0]
print(q["needle"], "|", q["question"], "->", q["answer"])
```

## Source books

| Folder | Contents |
|---|---|
| Test | The first 100 books of the PG-19 test split, from `emozilla/pg19` (a Parquet copy of DeepMind's PG-19 with the same splits), streamed in order and saved as `000.txt` … `099.txt` |
| Train | 64 PG-19 training-split books, streamed from `emozilla/pg19` with seed 42 and shuffle buffer 8, cleaned and truncated by the TokenBank training code |

## Regenerating

```bash
python code/export_niah.py --code code --train-books TRAIN_BOOKS_DIR --test-books TEST_BOOKS_DIR \
    --out data --qwen-tokenizer Qwen/Qwen3-4B-Instruct-2507
```

`code/niah_task.py` is the generator used in the experiments. Its `build()` arguments for each set are listed in
`code/export_niah.py`, including seeds, distances, questions per cell and word pools.

## Licence and attribution

The haystack text comes from PG-19, which consists of Project Gutenberg books published before 1919 and in the public
domain. If you use this dataset, please also cite PG-19:

> Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, Timothy P. Lillicrap.
> *Compressive Transformers for Long-Range Sequence Modelling.* ICLR 2020.
