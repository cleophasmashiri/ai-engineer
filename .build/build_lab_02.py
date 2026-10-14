"""SPOILERS — builds lab_02_tokenization.ipynb (contains solutions)."""
from pathlib import Path
from nbbuild import Notebook

COURSE = Path(__file__).resolve().parent.parent / "generative-ai-llm-architecture-data-preparation"
nb = Notebook("lab_02_tokenization", "Lab 2 — Implementing Tokenization")

nb.md(r'''
# Lab 2 — Implementing Tokenization

*Adapted from the AI Engineer course "Generative AI and LLMs: Architecture and Data Preparation".*

Before a model sees text, a **tokenizer** chops it into units and maps each unit to an integer id. The choice of
unit affects vocabulary size, sequence length, how unknown words are handled, and so what the model can learn.

| # | Concept | You will build |
|---|---|---|
| 1 | Word-based tokenization | `word_tokenize_simple()` |
| 2 | Character-based tokenization | `char_tokenize()` |
| 3 | The out-of-vocabulary (OOV) problem | `vocab_and_oov()` |
| 4 | Subword tokenization: **BPE from scratch** | `most_frequent_pair()`, `merge_pair()` |
| 5 | Training a BPE tokenizer | `train_bpe()` |
| 6 | WordPiece `##` vs SentencePiece `▁` | `merge_subwords()` |
| 7 | Real libraries: NLTK, spaCy, BERT, XLNet | `make_tokenizers()` |
| 8 | Frequency analysis | `token_frequencies()` |
| 9 | Timing with `datetime` | `time_tokenizer()` |
| 10 | **The lab exercise**: comparative analysis | `compare_tokenizers()` |

**Grading:** pass an exercise's tests → **+points**. Click *Show solution* before passing → **−points** (the same
amount). Revealing after you pass is free. Progress is saved in `.progress/`.
''')
nb.setup()
nb.code(r'''
import re
from collections import Counter

# The text from the original lab exercise
text = """
Going through the world of tokenization has been like walking through a huge maze made of words, symbols, and meanings. Each turn shows a bit more about the cool ways computers learn to understand our language. And while I'm still finding my way through it, the journey’s been enlightening and, honestly, a bunch of fun.
Eager to see where this learning path takes me next!"
"""
print(len(text), "characters")
''')

nb.md(r'''
---
## 1. The three granularities
''')
nb.figure("l2_granularities", "The trade-off: vocabulary size vs sequence length vs meaning per token.")

nb.md(r'''
### 1a. Word-based

The naive version is `text.split()`, but then punctuation sticks to words (`"words,"` ≠ `"words"`). A regular
expression does better: `\w+` matches runs of letters/digits/underscore, and `[^\w\s]` matches any single
character that is neither a word character nor whitespace (that is, punctuation).
''')
nb.code(r'''
print(text.split()[:12])
print(re.findall(r"\w+", "Hello, world! It's 2024."))       # words only
print(re.findall(r"[^\w\s]", "Hello, world! It's 2024."))   # punctuation only
''')
nb.exercise("ex1", "Regex word tokenizer", 10,
    prompt=r'''
Write `word_tokenize_simple(text)` that returns words **and** punctuation as separate tokens, in order:
every run of `\w` characters is a token, and every other non-whitespace character is its own token.
Combine the two patterns above with `|` in a single `re.findall`.
''',
    guide=r'''
![regex scan](figures/l2_regex.svg)

| regex piece | meaning |
|---|---|
| `r"..."` | a *raw* string: backslashes stay literal, so `\w` reaches the regex engine intact |
| `\w` | one "word" character: letter, digit or `_` |
| `\s` | one whitespace character (space, tab, newline) |
| `+` | one or more of the previous item: `\w+` = a whole run of word characters |
| `[^ ... ]` | any ONE character **not** in the brackets: `[^\w\s]` = not a word char and not whitespace |
| `A\|B` | try `A`; if it doesn't match here, try `B` |

**`re.findall(pattern, text)`** returns a list of every non-overlapping match, left to right.

**🪜 Steps:** your whole function is one `return re.findall(...)` with the two patterns joined by `|`.
''',
    guide_code=r'''
print(re.findall(r"\d+", "a1 b22 c333"))          # runs of digits
print(re.findall(r"[aeiou]", "banana"))           # single characters from a set
print(re.findall(r"[^a-z ]", "hi, you!"))         # NOT a-z and NOT space → punctuation
print(re.findall(r"\d+|[a-z]", "ab12c"))          # alternation: digit-runs OR single letters
''',
    stub=r'''
def word_tokenize_simple(text):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_splits_punctuation():
    assert word_tokenize_simple("Hello, world!") == ["Hello", ",", "world", "!"]

def test_contraction_is_split_naively():
    got = word_tokenize_simple("I'm still here, honestly!")
    assert got == ["I", "'", "m", "still", "here", ",", "honestly", "!"], got

def test_on_lab_text():
    toks = word_tokenize_simple(text)
    assert len(toks) == 79, f"expected 79 tokens, got {len(toks)}"
    assert toks[:4] == ["Going", "through", "the", "world"]
''',
    solution=r'''
def word_tokenize_simple(text):
    return re.findall(r"\w+|[^\w\s]", text)
''')

nb.md(r'''
### 1b. Character-based

Every character is a token. The vocabulary is tiny (an alphabet plus punctuation), there are no unknown
words, and sequences become very long.
''')
nb.exercise("ex2", "Character tokenizer", 10,
    prompt=r'''
Write `char_tokenize(text, keep_spaces=False)` → list of characters. When `keep_spaces` is False, drop **all
whitespace** characters (spaces, newlines, tabs: use `str.isspace()`).
''',
    guide=r'''
![char tokenize](figures/l2_chars.svg)

- **Looping over a string** gives one character at a time: `for c in "hi"` → `"h"`, `"i"`.
- **A list comprehension with a filter:** `[x for x in items if condition]`.
- **`c.isspace()`** is `True` for `" "`, `"\n"`, `"\t"`, …
- The keep-condition in words: *keep `c` if `keep_spaces` is on, OR `c` is not whitespace*.
''',
    guide_code=r'''
print([c for c in "a1 b2" if c.isdigit()])
print(" ".isspace(), "\n".isspace(), "x".isspace())
keep_all = False
print([c for c in "a b" if keep_all or c != " "])
''',
    stub=r'''
def char_tokenize(text, keep_spaces=False):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_drops_whitespace_by_default():
    assert char_tokenize("hi there\n!") == ["h", "i", "t", "h", "e", "r", "e", "!"]

def test_keep_spaces():
    assert char_tokenize("a b", keep_spaces=True) == ["a", " ", "b"]

def test_sequence_length_explodes():
    n_chars, n_words = len(char_tokenize(text)), len(word_tokenize_simple(text))
    assert n_chars > 3 * n_words, f"{n_chars} chars vs {n_words} words"
''',
    solution=r'''
def char_tokenize(text, keep_spaces=False):
    return [c for c in text if keep_spaces or not c.isspace()]
''')
nb.code(r'''
try:
    for name, toks in [("word", word_tokenize_simple(text)), ("char", char_tokenize(text))]:
        print(f"{name:>5}: {len(toks):4d} tokens, vocabulary {len(set(toks)):3d}")
except NotImplementedError:
    print("Finish Exercises 1–2 first.")
''')

nb.md(r'''
---
## 2. The out-of-vocabulary (OOV) problem

A word-level model can only embed words it saw in training. Everything else becomes `[UNK]`, and all
information about it is lost. The **OOV rate** is the fraction of test tokens missing from the training
vocabulary. Character vocabularies almost never have OOVs, which is what motivates subwords: get the
coverage of characters with the efficiency of words.
''')
nb.exercise("ex3", "Vocabulary size and OOV rate", 10,
    prompt=r'''
Write `vocab_and_oov(train_tokens, test_tokens)` that returns a tuple `(vocab_size, oov_rate)`:
- `vocab_size`: number of **distinct** training tokens
- `oov_rate`: fraction of test tokens (counting repeats) not in the training vocabulary, as a float in [0, 1]
  (return `0.0` for an empty test list)
''',
    guide=r'''
![OOV](figures/l2_oov.svg)

- **`set(tokens)`** keeps the distinct tokens, and `len(set(...))` is the vocabulary size.
- **Membership in a set** (`t in vocab`) is fast (O(1)), so build the set once.
- **Counting matches:** `sum(1 for t in xs if condition)`.
- **Guard the division:** if `test_tokens` is empty, return `0.0` before dividing.
- Return a **tuple**: `return size, rate`.
''',
    guide_code=r'''
xs = ["a", "b", "a", "c"]
print(set(xs), len(set(xs)))
print(sum(1 for t in xs if t != "a"), "of", len(xs), "→", sum(1 for t in xs if t != "a") / len(xs))
''',
    stub=r'''
def vocab_and_oov(train_tokens, test_tokens):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_simple_counts():
    assert vocab_and_oov(["a", "b", "a"], ["a", "c", "c", "b"]) == (2, 0.5)

def test_empty_test_set():
    assert vocab_and_oov(["a"], []) == (1, 0.0)

def test_word_vs_char_oov():
    train, test = text[:250], "Unbelievably, tokenizers handle zymurgy!"
    _, word_oov = vocab_and_oov(word_tokenize_simple(train), word_tokenize_simple(test))
    _, char_oov = vocab_and_oov(char_tokenize(train), char_tokenize(test))
    assert word_oov > 0.5 and char_oov < word_oov, (word_oov, char_oov)
''',
    solution=r'''
def vocab_and_oov(train_tokens, test_tokens):
    vocab = set(train_tokens)
    if not test_tokens:
        return len(vocab), 0.0
    missing = sum(1 for t in test_tokens if t not in vocab)
    return len(vocab), missing / len(test_tokens)
''')

nb.md(r'''
---
## 3. Subword tokenization: Byte-Pair Encoding (BPE) from scratch

BPE (used by GPT-2/3/4, Llama, RoBERTa) **learns** its vocabulary from data:

1. Start with every word split into characters, plus an end-of-word marker `</w>`.
2. Count every **adjacent pair** of symbols across the corpus (weighted by word frequency).
3. **Merge** the most frequent pair into one new symbol everywhere.
4. Repeat for N merges. Each merge adds one vocabulary entry.

Frequent words end up as single tokens, and rare words stay split into reusable pieces.
''')
nb.figure("l2_bpe_pairs", "Step 2 of BPE: count adjacent pairs, weighted by how often each word occurs. Here (e, s) wins and becomes the new symbol \"es\".")
nb.md(r'''
We represent the corpus as `word_freqs`: a dict mapping a **tuple of symbols** to its count, e.g.
`{('l','o','w','</w>'): 5, ...}`.
''')
nb.code(r'''
def corpus_to_word_freqs(words):
    """['low', 'low', 'lower'] -> {('l','o','w','</w>'): 2, ('l','o','w','e','r','</w>'): 1}"""
    return dict(Counter(tuple(w) + ("</w>",) for w in words))

toy_words = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 + ["widest"] * 3
toy = corpus_to_word_freqs(toy_words)
for k, v in toy.items():
    print(v, k)
''')
nb.exercise("ex4", "BPE: count pairs and merge", 20,
    prompt=r'''
Write two functions:

- `most_frequent_pair(word_freqs)` → the adjacent symbol pair `(a, b)` with the highest **frequency-weighted**
  count. Break ties by choosing the **lexicographically smallest** pair (so results are deterministic).
  Return `None` if no word has 2+ symbols.
- `merge_pair(pair, word_freqs)` → a **new** dict where every occurrence of `a` immediately followed by `b`
  is replaced by the single symbol `a + b`. Scan each word left to right, and don't modify the input.
''',
    guide=r'''
**Part 1: `most_frequent_pair`.** See the pair-counting figure above.
- **`zip(word, word[1:])`** pairs every symbol with its right neighbour.
- **`Counter()`** starts every count at 0: `pairs[(a, b)] += freq` adds the word's frequency for each occurrence.
- **Max with a tie-break:** `min(pairs, key=lambda p: (-pairs[p], p))`. Negating the count makes the *largest*
  count the *smallest* key, and the second tuple element `p` breaks ties alphabetically.
- If `pairs` is empty (every word is a single symbol), return `None`.

**Part 2: `merge_pair`.**

![merge pair](figures/l2_bpe_merge.svg)

- Build a **new** dict `out`. For each `(word, freq)`, walk with an index `i` (a `while` loop, because you sometimes
  jump by 2).
- Two different words can become the same tuple after merging, so add the counts:
  `out[key] = out.get(key, 0) + freq`.
''',
    guide_code=r'''
w = ("n", "e", "w")
print(list(zip(w, w[1:])))                        # adjacent pairs

c = Counter()
c[("x", "y")] += 5; c[("x", "y")] += 3; c[("a", "b")] += 8
print(c, "→ winner:", min(c, key=lambda p: (-c[p], p)))   # tie at 8 → ('a','b') wins alphabetically

i, out = 0, []
s = ["a", "b", "c"]
while i < len(s):
    out.append(s[i]); i += 1
print(tuple(out), {"k": 1}.get("missing", 0))
''',
    stub=r'''
def most_frequent_pair(word_freqs):
    # YOUR CODE HERE
    raise NotImplementedError

def merge_pair(pair, word_freqs):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_first_pair_is_e_s():
    assert most_frequent_pair(toy) == ("e", "s"), most_frequent_pair(toy)

def test_tie_break_lexicographic():
    assert most_frequent_pair({("b", "a"): 1, ("a", "b"): 1}) == ("a", "b")
    assert most_frequent_pair({("x",): 3}) is None

def test_merge():
    merged = merge_pair(("e", "s"), toy)
    assert merged[("n", "e", "w", "es", "t", "</w>")] == 6
    assert merged[("w", "i", "d", "es", "t", "</w>")] == 3
    assert merged[("l", "o", "w", "</w>")] == 5
    assert ("n", "e", "w", "e", "s", "t", "</w>") in toy, "don't mutate the input"

def test_merge_repeated_pair_in_one_word():
    assert merge_pair(("a", "a"), {("a", "a", "a", "a"): 1}) == {("aa", "aa"): 1}
''',
    solution=r'''
def most_frequent_pair(word_freqs):
    pairs = Counter()
    for word, freq in word_freqs.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += freq
    if not pairs:
        return None
    return min(pairs, key=lambda p: (-pairs[p], p))

def merge_pair(pair, word_freqs):
    a, b = pair
    out = {}
    for word, freq in word_freqs.items():
        new, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new.append(a + b); i += 2
            else:
                new.append(word[i]); i += 1
        out[tuple(new)] = out.get(tuple(new), 0) + freq
    return out
''')
nb.exercise("ex5", "Train a BPE tokenizer", 20,
    prompt=r'''
Write `train_bpe(words, num_merges)` that starts from `corpus_to_word_freqs(words)` and repeats
*find most frequent pair → merge* up to `num_merges` times (stop early if no pair remains).
Return a tuple `(merges, word_freqs)`: the list of merged pairs **in order**, and the final corpus.
''',
    guide=r'''
![train bpe](figures/l2_bpe_train.svg)

- Reuse `corpus_to_word_freqs`, your `most_frequent_pair` and your `merge_pair`.
- `for _ in range(num_merges):` and `break` as soon as the pair is `None`.
- Record each pair with `merges.append(pair)` **before** you merge it, then reassign `wf = merge_pair(pair, wf)`.
- Return `(merges, wf)`.
''',
    stub=r'''
def train_bpe(words, num_merges):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_classic_merges():
    merges, _ = train_bpe(toy_words, 4)
    assert merges == [("e", "s"), ("es", "t"), ("est", "</w>"), ("l", "o")], merges

def test_final_corpus():
    _, wf = train_bpe(toy_words, 10)
    assert wf[("low</w>",)] == 5 and wf[("newest</w>",)] == 6, wf

def test_stops_when_nothing_to_merge():
    merges, wf = train_bpe(["ab"], 50)
    assert merges == [("a", "b"), ("ab", "</w>")] and wf == {("ab</w>",): 1}
''',
    solution=r'''
def train_bpe(words, num_merges):
    wf = corpus_to_word_freqs(words)
    merges = []
    for _ in range(num_merges):
        pair = most_frequent_pair(wf)
        if pair is None:
            break
        merges.append(pair)
        wf = merge_pair(pair, wf)
    return merges, wf
''')
nb.code(r'''
try:
    merges, wf = train_bpe(word_tokenize_simple(text.lower()), 40)
    print("first 15 merges:", merges[:15])
    print("\nlab text after 40 merges (first 12 words):")
    for w in list(wf)[:12]:
        print("  ", " ".join(w))
except NotImplementedError:
    print("Finish Exercises 4–5 first.")
''')
nb.md(r'''
Frequent words like `the</w>` and `through</w>` become a single token quickly, while rarer words stay in pieces.
Production BPE tokenizers do exactly this with ~50k–200k merges over terabytes of text, usually at the
**byte** level so that *any* string (emoji, code, any language) can be encoded with no `[UNK]` at all.
''')

nb.md(r'''
---
## 4. WordPiece (`##`) vs SentencePiece (`▁`)

Real tokenizers mark word boundaries in different ways:

| tokenizer | used by | marker | example: `"Unbelievably fun"` |
|---|---|---|---|
| **WordPiece** | BERT, DistilBERT | `##` = *continues the previous word* | `un ##bel ##ie ##va ##bly fun` |
| **SentencePiece** (Unigram/BPE) | XLNet, T5, Llama, ALBERT | `▁` = *a space preceded this piece* | `▁Un be liev ably ▁fun` |

SentencePiece treats the space as an ordinary symbol, so decoding is lossless (`▁`→space). WordPiece
drops spaces around punctuation, so `"I'm"` decodes as `"i ' m"`.
''')
nb.code(r'''
from transformers import BertTokenizer, XLNetTokenizer
bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")   # WordPiece, lower-cases
xlnet_tok = XLNetTokenizer.from_pretrained("xlnet-base-cased")  # SentencePiece (Unigram)

s = "I'm learning tokenization!"
print("BERT :", bert_tok.tokenize(s), "→ ids", bert_tok.encode(s))   # encode adds [CLS] ... [SEP]
print("XLNet:", xlnet_tok.tokenize(s), "→ ids", xlnet_tok.encode(s)) # XLNet appends <sep> <cls>
''')
nb.exercise("ex6", "Reassemble words from subword pieces", 15,
    prompt=r'''
Write `merge_subwords(tokens, scheme)` that rebuilds whole words from subword tokens:
- `scheme="wordpiece"`: a token starting with `##` is glued (without the `##`) onto the previous word, and every
  other token starts a new word.
- `scheme="sentencepiece"`: a token starting with `▁` starts a new word (without the `▁`), and every other token
  is glued onto the previous word (or starts the first word if there is none yet).
- Any other scheme → `raise ValueError`.
''',
    guide=r'''
![subword markers](figures/l2_subwords.svg)

- **`t.startswith("##")`** / **`t.startswith("▁")`** test the marker. **`t[2:]`** / **`t[1:]`** strip it.
- **`words[-1] += piece`** glues onto the last word you've built so far.
- Guard `words[-1]` when `words` is still empty: a continuation piece at the very start becomes its own word.
- **`raise ValueError(f"unknown scheme {scheme!r}")`** for anything that isn't one of the two names.
''',
    guide_code=r'''
words = ["un"]
t = "##bel"
if t.startswith("##"):
    words[-1] += t[2:]
print(words, "▁fun"[1:], "▁fun".startswith("▁"))
try:
    raise ValueError("demo")
except ValueError as e:
    print("caught:", e)
''',
    stub=r'''
def merge_subwords(tokens, scheme):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_wordpiece():
    toks = ['un', '##bel', '##ie', '##va', '##bly', ',', 'the', 'journey']
    assert merge_subwords(toks, "wordpiece") == ["unbelievably", ",", "the", "journey"]

def test_sentencepiece():
    toks = ['▁I', "'", 'm', '▁learning', '▁token', 'ization', '!']
    assert merge_subwords(toks, "sentencepiece") == ["I'm", "learning", "tokenization!"]

def test_with_real_tokenizers():
    s = "Unbelievably, the journey has been enlightening"
    assert merge_subwords(bert_tok.tokenize(s), "wordpiece") == ["unbelievably", ",", "the", "journey", "has", "been", "enlightening"]
    assert " ".join(merge_subwords(xlnet_tok.tokenize(s), "sentencepiece")) == s

def test_unknown_scheme():
    try:
        merge_subwords(["a"], "bpe?")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
''',
    solution=r'''
def merge_subwords(tokens, scheme):
    words = []
    if scheme == "wordpiece":
        for t in tokens:
            if t.startswith("##") and words:
                words[-1] += t[2:]
            else:
                words.append(t)
    elif scheme == "sentencepiece":
        for t in tokens:
            if t.startswith("▁"):
                words.append(t[1:])
            elif words:
                words[-1] += t
            else:
                words.append(t)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return words
''')

nb.md(r'''
---
## 5. The four libraries from the original lab

| library | type | notes |
|---|---|---|
| **NLTK** `word_tokenize` | rule-based word (Penn Treebank) | splits `I'm` → `I`, `'m`; needs the `punkt_tab` data |
| **spaCy** | rule-based word + exceptions | language-specific rules; `spacy.blank("en")` = tokenizer only (no model download) |
| **BertTokenizer** | WordPiece subword | 30,522-token vocab, lower-cased (`bert-base-uncased`) |
| **XLNetTokenizer** | SentencePiece subword | 32,000-token vocab, case-preserving |
''')
nb.code(r'''
import nltk, spacy
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import word_tokenize
nlp_en = spacy.blank("en")    # (spacy.load("en_core_web_sm") works too, same tokenizer + a full pipeline)

s = "Unbelievably, the journey’s been enlightening."
print("NLTK :", word_tokenize(s))
print("spaCy:", [t.text for t in nlp_en(s)])
''')
nb.md("Look at `journey’s` with a curly apostrophe: NLTK and spaCy disagree. Rule-based tokenizers can differ on edge cases like this.")
nb.exercise("ex7", "One interface for four tokenizers", 15,
    prompt=r'''
Write `make_tokenizers()` returning a dict with the keys `"nltk"`, `"spacy"`, `"bert"`, `"xlnet"`. Each value
is a function `str → list[str]` (use `word_tokenize`, `nlp_en`, `bert_tok.tokenize`, `xlnet_tok.tokenize`).
For spaCy, return the token **texts**, not `Token` objects.
''',
    guide=r'''
![one interface](figures/l2_interface.svg)

- **Functions are values:** you can store them in a dict without calling them (`word_tokenize`, not `word_tokenize(...)`).
- **Bound methods are functions too:** `bert_tok.tokenize` remembers which tokenizer it belongs to.
- **spaCy returns a `Doc`**, so wrap it in a `lambda` that returns `[t.text for t in nlp_en(s)]`.
''',
    guide_code=r'''
ops = {"upper": str.upper, "length": lambda s: len(s)}
print(ops["upper"]("hi"), ops["length"]("hello"))
doc = nlp_en("Hi there!")
print(type(doc).__name__, [type(t).__name__ for t in doc][:1], [t.text for t in doc])
''',
    stub=r'''
def make_tokenizers():
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
S = "I'm learning tokenization!"

def test_keys():
    assert set(make_tokenizers()) == {"nltk", "spacy", "bert", "xlnet"}

def test_word_level():
    t = make_tokenizers()
    assert t["nltk"](S) == ["I", "'m", "learning", "tokenization", "!"]
    assert t["spacy"](S) == ["I", "'m", "learning", "tokenization", "!"]
    assert all(isinstance(x, str) for x in t["spacy"](S)), "return strings, not spaCy Tokens"

def test_subword_level():
    t = make_tokenizers()
    assert t["bert"](S) == ["i", "'", "m", "learning", "token", "##ization", "!"]
    assert t["xlnet"](S) == ["▁I", "'", "m", "▁learning", "▁token", "ization", "!"]
''',
    solution=r'''
def make_tokenizers():
    return {
        "nltk": word_tokenize,
        "spacy": lambda s: [t.text for t in nlp_en(s)],
        "bert": bert_tok.tokenize,
        "xlnet": xlnet_tok.tokenize,
    }
''')

nb.md(r'''
---
## 6. Frequency analysis

Counting tokens shows what a tokenizer does to a text: which words dominate, how much punctuation there is,
and how many pieces subword tokenizers create. `collections.Counter` is the tool for this.
''')
nb.code(r'''
c = Counter(["the", "cat", "the", "dog", "the", "cat"])
print(c.most_common(2))
print(sorted(c.items(), key=lambda kv: (-kv[1], kv[0])))   # deterministic: count desc, then alphabetical
''')
nb.exercise("ex8", "Token frequencies", 10,
    prompt=r'''
Write `token_frequencies(tokens, top_n=None, lowercase=False, drop_punct=False)` returning a list of
`(token, count)` sorted by **count descending, then token ascending**.
- `lowercase`: lower-case tokens before counting
- `drop_punct`: drop tokens made **only** of non-word characters (no `\w` in them, e.g. `","`, `"!"`, `"’"`)
- `top_n`: if given, keep only the first `top_n` entries
''',
    guide=r'''
![frequencies](figures/l2_freq.svg)

- Apply the optional filters **first** (lower-case, drop punctuation), then count with `Counter(tokens)`.
- **`re.search(r"\w", t)`** returns a match object (truthy) if `t` contains *any* word character, else `None`.
  `"##ing"` and `"▁the"` contain letters, so they're kept. `","` and `"’"` are dropped.
- **`sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))`** sorts by count descending, then token A→Z.
- **Slicing with `None`:** `items[:None]` returns the whole list, so `items[:top_n]` handles both cases.
''',
    guide_code=r'''
print(bool(re.search(r"\w", "##ing")), bool(re.search(r"\w", ",")))
items = list(Counter("abracadabra").items())
print(sorted(items, key=lambda kv: (-kv[1], kv[0])))
print(items[:None] == items, items[:2])
''',
    stub=r'''
def token_frequencies(tokens, top_n=None, lowercase=False, drop_punct=False):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
T = ["The", "cat", ",", "the", "dog", "!", "the", "cat", ","]

def test_basic_sort():
    assert token_frequencies(T) == [(",", 2), ("cat", 2), ("the", 2), ("!", 1), ("The", 1), ("dog", 1)]

def test_lowercase_drop_punct_top_n():
    assert token_frequencies(T, top_n=2, lowercase=True, drop_punct=True) == [("the", 3), ("cat", 2)]

def test_drop_keeps_subword_markers():
    assert token_frequencies(["##ing", "▁the", "’", "!"], drop_punct=True) == [("##ing", 1), ("▁the", 1)]
''',
    solution=r'''
def token_frequencies(tokens, top_n=None, lowercase=False, drop_punct=False):
    if lowercase:
        tokens = [t.lower() for t in tokens]
    if drop_punct:
        tokens = [t for t in tokens if re.search(r"\w", t)]
    items = sorted(Counter(tokens).items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:top_n] if top_n is not None else items
''')

nb.md(r'''
---
## 7. Measuring speed with `datetime`

The original lab asks you to time each tokenizer with `datetime`. `datetime.now()` returns a timestamp, subtracting
two gives a `timedelta`, and `.total_seconds()` converts it to a float. A single run is noisy, so average over
several repeats. (For serious benchmarks use `time.perf_counter()` or `timeit`.)
''')
nb.code(r'''
from datetime import datetime
start = datetime.now()
sum(range(1_000_000))
elapsed = datetime.now() - start
print(type(elapsed).__name__, elapsed, "→", elapsed.total_seconds(), "s")
''')
nb.exercise("ex9", "Time a tokenizer", 10,
    prompt=r'''
Write `time_tokenizer(fn, text, repeats=5)` that calls `fn(text)` `repeats` times, measures the **total** time with
`datetime.now()`, and returns a tuple `(tokens, avg_seconds)`: the tokens from the last call and the
average seconds per call (a float).
''',
    guide=r'''
![timing](figures/l2_timing.svg)

- `start = datetime.now()` → loop `repeats` times calling `fn(text)` → `elapsed = datetime.now() - start`.
- `elapsed` is a `timedelta`, and `.total_seconds()` gives a float.
- Return `(tokens, seconds / repeats)`, where `tokens` is the result of the last call.
''',
    stub=r'''
def time_tokenizer(fn, text, repeats=5):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
import time as _time

def test_returns_tokens_and_float():
    toks, secs = time_tokenizer(str.split, "a b c", repeats=3)
    assert toks == ["a", "b", "c"] and isinstance(secs, float) and secs >= 0

def test_average_not_total():
    calls = []
    def slow(s):
        calls.append(1); _time.sleep(0.02); return [s]
    _, secs = time_tokenizer(slow, "x", repeats=4)
    assert len(calls) == 4, f"fn called {len(calls)} times"
    assert 0.015 < secs < 0.06, f"expected ~0.02 s per call, got {secs:.4f}"
''',
    solution=r'''
def time_tokenizer(fn, text, repeats=5):
    start = datetime.now()
    for _ in range(repeats):
        tokens = fn(text)
    elapsed = (datetime.now() - start).total_seconds()
    return tokens, elapsed / repeats
''')

nb.md(r'''
---
## 8. The lab exercise: comparative tokenization and performance analysis

Now combine everything: run all four libraries on the lab text, then compare token counts, vocabulary,
top frequencies and speed.
''')
nb.exercise("ex10", "Compare the tokenizers", 15,
    prompt=r'''
Write `compare_tokenizers(text, tokenizers, top_n=5, repeats=5)` where `tokenizers` is a dict like
`make_tokenizers()` returns. For each `(name, fn)` **in sorted name order**, build a dict:

| key | value |
|---|---|
| `"name"` | the tokenizer name |
| `"n_tokens"` | number of tokens |
| `"n_unique"` | number of distinct tokens |
| `"top"` | `token_frequencies(tokens, top_n=top_n, lowercase=True, drop_punct=True)` |
| `"seconds"` | average seconds from `time_tokenizer(fn, text, repeats)` |

Return the list of dicts.
''',
    guide=r'''
![compare](figures/l2_compare.svg)

- **`sorted(a_dict)`** returns the dict's **keys**, sorted, which is the order the tests expect.
- `tokenizers[name]` is a function, so pass it straight to your `time_tokenizer`.
- Build one dict per tokenizer with exactly the five keys, `append` it, and return the list.
''',
    guide_code=r'''
d = {"b": len, "a": str.upper}
for name in sorted(d):
    print(name, d[name]("xyz"))
''',
    stub=r'''
def compare_tokenizers(text, tokenizers, top_n=5, repeats=5):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_structure_and_order():
    rows = compare_tokenizers(text, make_tokenizers(), repeats=1)
    assert [r["name"] for r in rows] == ["bert", "nltk", "spacy", "xlnet"]
    assert all(set(r) == {"name", "n_tokens", "n_unique", "top", "seconds"} for r in rows)

def test_counts_on_lab_text():
    rows = {r["name"]: r for r in compare_tokenizers(text, make_tokenizers(), repeats=1)}
    assert rows["nltk"]["n_tokens"] == len(word_tokenize(text))
    assert rows["bert"]["n_tokens"] > rows["nltk"]["n_tokens"], "subword tokenizers should produce more tokens"
    assert rows["nltk"]["top"][:2] == [("a", 3), ("and", 3)], rows["nltk"]["top"]

def test_with_toy_tokenizer():
    rows = compare_tokenizers("b a b", {"ws": str.split}, top_n=1, repeats=2)
    r = rows[0]
    assert (r["name"], r["n_tokens"], r["n_unique"], r["top"]) == ("ws", 3, 2, [("b", 2)]), r
    assert isinstance(r["seconds"], float)
''',
    solution=r'''
def compare_tokenizers(text, tokenizers, top_n=5, repeats=5):
    rows = []
    for name in sorted(tokenizers):
        tokens, secs = time_tokenizer(tokenizers[name], text, repeats)
        rows.append({
            "name": name,
            "n_tokens": len(tokens),
            "n_unique": len(set(tokens)),
            "top": token_frequencies(tokens, top_n=top_n, lowercase=True, drop_punct=True),
            "seconds": secs,
        })
    return rows
''')
nb.code(r'''
try:
    rows = compare_tokenizers(text, make_tokenizers())
    print(f"{'library':<8}{'tokens':>8}{'unique':>8}{'ms/call':>10}   top-5 (lower-cased, no punctuation)")
    for r in rows:
        top = ", ".join(f"{t}×{c}" for t, c in r["top"])
        print(f"{r['name']:<8}{r['n_tokens']:>8}{r['n_unique']:>8}{r['seconds'] * 1000:>10.2f}   {top}")
except NotImplementedError:
    print("Finish Exercises 7–10 first.")
''')
nb.md(r'''
**Questions to think about** (no grading, but jot down answers):
1. Why do BERT and XLNet produce *more* tokens than NLTK/spaCy for the same text?
2. Which words did BERT split into `##` pieces? Why those?
3. Why does XLNet's top list show `▁the` rather than `the`?
4. Which library was fastest? Why might a Rust-backed "fast" tokenizer beat a pure-Python one?
5. If you were training a translation model, which granularity would you choose, and why?

---
## Recap

- **Word** tokenization: meaningful but huge vocabularies and OOVs. **Character**: no OOVs but long sequences.
- **Subword** (BPE / WordPiece / SentencePiece) gets the best of both, and all modern LLMs use it.
- BPE *learns* merges from pair frequencies, and you implemented it.
- `##` (WordPiece) marks continuation, `▁` (SentencePiece) marks a preceding space.
- Tokenizers disagree on edge cases, so always use **the tokenizer the model was trained with**.

Next lab: turning these tokens into padded, batched tensors with a PyTorch `DataLoader`.

### Your score
''')
nb.code("lab.scoreboard()")

nb.write(COURSE, "lab_02_tokenization.ipynb")
