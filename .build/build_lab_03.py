"""SPOILERS — builds lab_03_nlp_data_loader.ipynb (contains solutions)."""
from pathlib import Path
from nbbuild import Notebook

COURSE = Path(__file__).resolve().parent.parent / "generative-ai-llm-architecture-data-preparation"
nb = Notebook("lab_03_nlp_data_loader", "Lab 3 — Creating an NLP Data Loader")

nb.md(r'''
# Lab 3 — Creating an NLP Data Loader

*Adapted from the AI Engineer course "Generative AI and LLMs: Architecture and Data Preparation".*

You're an AI engineer on a **French → English translation** project. Models train on *batches* of tensors,
but sentences are strings of different lengths. This lab builds the pipeline that turns raw sentences into
padded, batched tensors:
''')
nb.figure("l3_pipeline", "Everything before the model is data preparation, and the DataLoader's collate_fn is where it happens.")
nb.md(r'''
| # | Concept | You will build |
|---|---|---|
| 1 | `Dataset`: indexable access to samples | `FrenchDataset` |
| 2 | Tokenization with spaCy (French rules) | `tokenize_fr()` |
| 3 | Vocabulary with special tokens | `build_vocab()` |
| 4 | Numericalization, `<unk>`, `<bos>`/`<eos>` | `numericalize()` |
| 5 | Padding variable-length sequences | `pad_batch()` |
| 6 | The `collate_fn` hook | `collate_fn()` |
| 7 | Measuring padding waste | `padding_stats()` |
| 8 | **The lab exercise**: sort by length, batch size 4 | `make_loader()` |
| 9 | Bucketing: sorted *and* shuffled | `BucketBatchSampler` |
| 10 | Source/target pairs for translation | `collate_pairs()` |

> **Why not `torchtext`?** The original lab uses `torchtext.vocab.build_vocab_from_iterator` and
> `get_tokenizer`. torchtext is **deprecated** and no longer works with current PyTorch. Building the vocab
> yourself takes about 10 lines and shows you exactly what it does. These are the same concepts the course exam
> asks about.

**Grading:** pass an exercise's tests → **+points**. Click *Show solution* before passing → **−points**.
Revealing after you pass is free. Progress is saved in `.progress/`.
''')
nb.setup()
nb.code(r'''
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# The French corpus from the original lab
corpus = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !",
]
print(len(corpus), "sentences")
''')

nb.md(r'''
---
## 1. `Dataset`: a uniform way to get sample *i*

PyTorch splits data handling into two pieces:

- **`Dataset`**: *what* the data is. It implements `__len__` and `__getitem__(i)` (a "map-style" dataset).
- **`DataLoader`**: *how* to iterate. It handles batching, shuffling, parallel workers, and a **`collate_fn`** that
  merges a list of samples into one batch.
''')
nb.figure("l3_dataset_loader", "The Dataset answers dataset[i]; the DataLoader picks indices, fetches samples, and hands the list to collate_fn.")
nb.exercise("ex1", "A custom Dataset", 10,
    prompt=r'''
Implement `FrenchDataset(sentences)`, a subclass of `torch.utils.data.Dataset` that stores the list and
returns the raw sentence string for `dataset[i]`. `len(dataset)` must work.
''',
    guide=r'''
- **Subclassing:** `class FrenchDataset(Dataset):` inherits from PyTorch's base class.
- **`__init__(self, sentences)`** stores the data on `self` (`list(sentences)` makes a copy).
- **`__len__(self)`** returns how many samples there are, so `len(ds)` works.
- **`__getitem__(self, idx)`** returns sample number `idx`, so `ds[idx]` works.
- That's all a map-style dataset needs. The figure above shows how the DataLoader uses it.
''',
    guide_code=r'''
class Squares(Dataset):                    # same pattern, different data
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        return idx * idx

ds = Squares(6)
print(len(ds), ds[3], [ds[i] for i in range(len(ds))])
for batch in DataLoader(ds, batch_size=4):
    print("batch:", batch)                 # ints are auto-collated into a tensor
''',
    stub=r'''
class FrenchDataset(Dataset):
    # YOUR CODE HERE
    pass
''',
    tests=r'''
def test_len_and_getitem():
    ds = FrenchDataset(corpus)
    assert len(ds) == 30
    assert ds[6] == "Je t'adore."

def test_is_a_torch_dataset():
    assert isinstance(FrenchDataset(["a"]), Dataset)

def test_default_dataloader_batches_strings():
    batch = next(iter(DataLoader(FrenchDataset(corpus), batch_size=3)))
    assert batch == corpus[:3], batch
''',
    solution=r'''
class FrenchDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = list(sentences)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
''')
nb.md(r'''
Without a `collate_fn`, the DataLoader returns a batch of strings as a plain list. Models need tensors, so the
rest of this lab builds the collate function.
''')

nb.md(r'''
---
## 2. Tokenizing French

French has elisions (`C'est`, `J'aime`, `aujourd'hui`) and hyphenated forms (`Est-ce`, `week-end`). spaCy ships
language-specific rules. `spacy.blank("fr")` gives the French **tokenizer only**, with no model download.
(`spacy.load("fr_core_news_sm")` uses the same tokenizer plus a tagger/parser.)
''')
nb.code(r'''
import spacy
nlp_fr = spacy.blank("fr")
for s in ["C'est un autre exemple de phrase.", "Il fait beau aujourd'hui.", "Est-ce que tu as des projets pour le week-end ?"]:
    print([t.text for t in nlp_fr(s)])
''')
nb.md("Look at the output: `C'` is split off as an elided article, `aujourd'hui` stays whole (it's a lexical exception), and `week-end` is split.")
nb.exercise("ex2", "French tokenizer", 5,
    prompt=r'''
Write `tokenize_fr(sentence)` → list of token **strings** using `nlp_fr`. Don't lower-case.
''',
    guide=r'''
![spaCy French](figures/l3_spacy_fr.svg)

- `nlp_fr(sentence)` returns a `Doc`. Iterating it gives `Token` objects, and each token's **`.text`** is its string.
- One list comprehension does it.
''',
    guide_code=r'''
doc = nlp_fr("Bonjour !")
for tok in doc:
    print(repr(tok), type(tok).__name__, "→", repr(tok.text), type(tok.text).__name__)
''',
    stub=r'''
def tokenize_fr(sentence):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_elision():
    assert tokenize_fr("C'est un autre exemple de phrase.") == ["C'", "est", "un", "autre", "exemple", "de", "phrase", "."]

def test_returns_strings():
    toks = tokenize_fr("Je t'adore.")
    assert toks == ["Je", "t'", "adore", "."] and all(type(t) is str for t in toks), toks
''',
    solution=r'''
def tokenize_fr(sentence):
    return [tok.text for tok in nlp_fr(sentence)]
''')

nb.md(r'''
---
## 3. Building a vocabulary

A vocabulary is a two-way mapping between tokens and integer ids:
- **`stoi`** ("string to index"): `dict[str, int]`
- **`itos`** ("index to string"): `list[str]`, where `itos[i]` is the token with id `i`

**Special tokens** go first so their ids are fixed and known:

| token | purpose |
|---|---|
| `<unk>` | any token not in the vocabulary |
| `<pad>` | filler that makes sequences in a batch the same length |
| `<bos>` | *beginning of sequence*: tells a decoder to start generating |
| `<eos>` | *end of sequence*: tells it to stop |

Ordering the rest by frequency (most common first) is conventional. `min_freq` drops rare tokens, which then
become `<unk>`. That keeps the vocabulary (and the model's embedding table) small.
''')
nb.code(r'''
SPECIALS = ["<unk>", "<pad>", "<bos>", "<eos>"]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
''')
nb.exercise("ex3", "Build the vocabulary", 15,
    prompt=r'''
Write `build_vocab(token_lists, specials=SPECIALS, min_freq=1)` → `(stoi, itos)`:
1. `itos` starts with `specials`, in the given order.
2. Then every token whose total count is `>= min_freq` (and that isn't already a special), sorted by
   **count descending, then alphabetically**.
3. `stoi` is the inverse mapping of `itos`.
''',
    guide=r'''
![vocab](figures/l3_vocab.svg)

- **Count across nested lists:** `Counter(tok for toks in token_lists for tok in toks)`.
- **Filter:** keep `t` if `counts[t] >= min_freq` **and** `t not in specials`.
- **Sort** with `key=lambda t: (-counts[t], t)`, so the count goes down and ties go A→Z.
- **`itos = list(specials) + words`**, and **`stoi = {t: i for i, t in enumerate(itos)}`**.
- Return `(stoi, itos)`.
''',
    guide_code=r'''
baskets = [["apple", "pear"], ["apple", "fig"], ["apple", "pear", "kiwi"]]
counts = Counter(f for b in baskets for f in b)
print(counts)
kept = [t for t in counts if counts[t] >= 2]
kept.sort(key=lambda t: (-counts[t], t))
itos_demo = ["<x>"] + kept
print(itos_demo, {t: i for i, t in enumerate(itos_demo)})
''',
    stub=r'''
def build_vocab(token_lists, specials=SPECIALS, min_freq=1):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
TOY = [["le", "chat", "dort"], ["le", "chien", "dort"], ["le", "chat"]]

def test_specials_first():
    stoi, itos = build_vocab(TOY)
    assert itos[:4] == SPECIALS and stoi["<pad>"] == PAD_IDX

def test_frequency_then_alpha_order():
    _, itos = build_vocab(TOY)
    assert itos[4:] == ["le", "chat", "dort", "chien"], itos

def test_min_freq_and_inverse():
    stoi, itos = build_vocab(TOY, min_freq=2)
    assert itos == SPECIALS + ["le", "chat", "dort"], itos
    assert all(stoi[t] == i for i, t in enumerate(itos))

def test_on_corpus():
    stoi, itos = build_vocab([tokenize_fr(s) for s in corpus])
    assert len(itos) == len(stoi) == 127, len(itos)
    assert itos[4:7] == [".", "Je", "!"], itos[4:7]
''',
    solution=r'''
def build_vocab(token_lists, specials=SPECIALS, min_freq=1):
    counts = Counter(tok for toks in token_lists for tok in toks)
    words = [t for t, c in counts.items() if c >= min_freq and t not in specials]
    words.sort(key=lambda t: (-counts[t], t))
    itos = list(specials) + words
    stoi = {t: i for i, t in enumerate(itos)}
    return stoi, itos
''')
nb.code(r'''
try:
    stoi, itos = build_vocab([tokenize_fr(s) for s in corpus])
    print("vocab size:", len(itos))
    print("first 20:", itos[:20])
except NotImplementedError:
    print("Finish Exercises 2–3 first.")
''')

nb.md(r'''
---
## 4. Numericalization

Map each token to its id, and give unknown tokens `<unk>`. For sequence-to-sequence models we usually wrap
every sentence in `<bos>` … `<eos>`. Models expect `torch.long` (int64) ids, because these index into an
embedding table.
''')
nb.exercise("ex4", "Numericalize", 10,
    prompt=r'''
Write `numericalize(tokens, stoi, add_bos_eos=True)` → a 1-D `torch.long` tensor of ids. Unknown tokens map to
`stoi["<unk>"]`. If `add_bos_eos`, prepend `stoi["<bos>"]` and append `stoi["<eos>"]`.
''',
    guide=r'''
![numericalize](figures/l3_numericalize.svg)

- **`stoi.get(t, unk)`** returns `unk` when `t` is missing (instead of raising `KeyError`). Fetch `unk = stoi["<unk>"]` once.
- Wrap with `[stoi["<bos>"]] + ids + [stoi["<eos>"]]` when `add_bos_eos` is true.
- **`torch.tensor(ids, dtype=torch.long)`** gives a 1-D LongTensor. An empty `ids` list still works.
''',
    guide_code=r'''
d = {"a": 1}
print(d.get("a", 0), d.get("zzz", 0))
t = torch.tensor([7] + [1, 2] + [9], dtype=torch.long)
print(t, t.dtype, t.shape)
''',
    stub=r'''
def numericalize(tokens, stoi, add_bos_eos=True):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
V = {"<unk>": 0, "<pad>": 1, "<bos>": 2, "<eos>": 3, "je": 4, "suis": 5}

def test_with_bos_eos():
    t = numericalize(["je", "suis"], V)
    assert t.tolist() == [2, 4, 5, 3] and t.dtype == torch.long, (t, t.dtype)

def test_unknown_and_no_specials():
    assert numericalize(["je", "danse"], V, add_bos_eos=False).tolist() == [4, 0]

def test_is_1d_tensor():
    assert isinstance(numericalize([], V), torch.Tensor) and numericalize([], V).shape == (2,)
''',
    solution=r'''
def numericalize(tokens, stoi, add_bos_eos=True):
    unk = stoi["<unk>"]
    ids = [stoi.get(t, unk) for t in tokens]
    if add_bos_eos:
        ids = [stoi["<bos>"]] + ids + [stoi["<eos>"]]
    return torch.tensor(ids, dtype=torch.long)
''')

nb.md(r'''
---
## 5. Padding

A batch tensor must be rectangular, so shorter sequences are filled with `<pad>` up to the **longest sequence in
the batch** (not the longest in the dataset). This is *dynamic padding*. We also keep the true **lengths**,
because models use them to build attention masks or pack sequences, so the padding is ignored.
''')
nb.code(r'''
from torch.nn.utils.rnn import pad_sequence
seqs = [torch.tensor([2, 9, 3]), torch.tensor([2, 7, 8, 6, 3]), torch.tensor([2, 3])]
print(pad_sequence(seqs, batch_first=True, padding_value=PAD_IDX))
print("default (batch_first=False) is (T, B):", pad_sequence(seqs, padding_value=PAD_IDX).shape)
''')
nb.md(r'''
`batch_first=True` → shape `(batch, time)`, which is what Transformers and most modern code expect. The default
`(time, batch)` is a legacy of RNN APIs.
''')
nb.exercise("ex5", "Pad a batch", 10,
    prompt=r'''
Write `pad_batch(seqs, pad_idx=PAD_IDX)` → `(padded, lengths)`:
- `padded`: `(batch, max_len)` long tensor, right-padded with `pad_idx`
- `lengths`: 1-D long tensor of the original lengths

(You may use `pad_sequence` or write the loop yourself.)
''',
    guide=r'''
![padding](figures/l3_padding.svg)

- **`pad_sequence(seqs, batch_first=True, padding_value=pad_idx)`** does the padding (demonstrated above).
- **Lengths:** `len(s)` of each original tensor, collected into `torch.tensor([...], dtype=torch.long)`.
- Return the tuple `(padded, lengths)`.
''',
    guide_code=r'''
s = torch.tensor([4, 5, 6])
print(len(s), s.shape[0])
print(torch.tensor([len(x) for x in [s, s[:1]]], dtype=torch.long))
''',
    stub=r'''
def pad_batch(seqs, pad_idx=PAD_IDX):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_shapes_and_values():
    padded, lengths = pad_batch([torch.tensor([5, 6, 7]), torch.tensor([8])])
    assert padded.tolist() == [[5, 6, 7], [8, 1, 1]], padded
    assert lengths.tolist() == [3, 1] and lengths.dtype == torch.long

def test_custom_pad_idx():
    padded, _ = pad_batch([torch.tensor([5]), torch.tensor([6, 7])], pad_idx=0)
    assert padded.tolist() == [[5, 0], [6, 7]]
''',
    solution=r'''
def pad_batch(seqs, pad_idx=PAD_IDX):
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    return padded, lengths
''')

nb.md(r'''
---
## 6. The `collate_fn` hook

`DataLoader` calls `collate_fn(list_of_samples)` for every batch. This is where the preprocessing goes:
**tokenize → numericalize → pad**. The Dataset stays simple (raw strings), and all batch-level logic lives
in one function.
''')
nb.exercise("ex6", "The collate function", 10,
    prompt=r'''
Write `collate_fn(batch)` where `batch` is a list of raw French sentences. Use the global `stoi`, your
`tokenize_fr`, `numericalize` (with `<bos>`/`<eos>`) and `pad_batch`. Return `(padded, lengths)`.
''',
    guide=r'''
![collate](figures/l3_collate.svg)

- `batch` is a **list of raw strings**, because that's what your Dataset's `__getitem__` returns.
- Compose your functions: `numericalize(tokenize_fr(s), stoi)` for each `s` (a list comprehension) → `pad_batch(seqs, PAD_IDX)`.
- `stoi` and `PAD_IDX` are globals defined above, so you can use them directly.
''',
    stub=r'''
def collate_fn(batch):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_collate_two_sentences():
    padded, lengths = collate_fn(["Je t'adore.", "Bonjour !"])
    assert padded.shape == (2, 6), padded.shape
    assert lengths.tolist() == [6, 4]
    assert padded[1].tolist()[-2:] == [PAD_IDX, PAD_IDX]
    assert padded[:, 0].tolist() == [BOS_IDX, BOS_IDX]

def test_round_trip_through_itos():
    padded, lengths = collate_fn(["Je suis là."])
    words = [itos[i] for i in padded[0].tolist()]
    assert words == ["<bos>", "Je", "suis", "là", ".", "<eos>"], words

def test_works_inside_dataloader():
    loader = DataLoader(FrenchDataset(corpus), batch_size=4, collate_fn=collate_fn)
    padded, lengths = next(iter(loader))
    assert padded.shape[0] == 4 and padded.shape[1] == int(lengths.max())
''',
    solution=r'''
def collate_fn(batch):
    seqs = [numericalize(tokenize_fr(s), stoi) for s in batch]
    return pad_batch(seqs, PAD_IDX)
''')

nb.md(r'''
---
## 7. How much compute goes to padding?

Every `<pad>` is wasted computation: the GPU multiplies it through every layer and then masks it out. Measure it
with the **padding ratio** = `#pad tokens / #total tokens` over all batches.
''')
nb.figure("l3_sorted_vs_unsorted", "Sorting by length before batching shrinks T and the padding. (Used in Exercises 7 and 8.)")
nb.exercise("ex7", "Padding statistics", 10,
    prompt=r'''
Write `padding_stats(loader, pad_idx=PAD_IDX)` that iterates the loader (each batch is `(padded, lengths)`) and
returns `{"pad": int, "total": int, "ratio": float}`, where `total` is the number of elements in all padded
tensors.
''',
    guide=r'''
- **Comparing a tensor with a number gives a boolean tensor:** `(padded == pad_idx)`.
- **`.sum()`** counts the `True`s. **`int(...)`** turns the 0-d tensor into a Python int.
- **`.numel()`** = total number of elements (rows × columns).
- **Iterating a loader** yields whatever `collate_fn` returned: `for padded, _ in loader:` (the `_` ignores the lengths).
- Return `{"pad": ..., "total": ..., "ratio": pad / total}`.
''',
    guide_code=r'''
m = torch.tensor([[2, 5, 3], [2, 3, 1]])
print(m == 1)
print((m == 1).sum(), int((m == 1).sum()), m.numel())
''',
    stub=r'''
def padding_stats(loader, pad_idx=PAD_IDX):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_on_fake_loader():
    fake = [(torch.tensor([[2, 5, 3], [2, 3, 1]]), None), (torch.tensor([[2, 1]]), None)]
    assert padding_stats(fake) == {"pad": 2, "total": 8, "ratio": 0.25}

def test_unsorted_corpus_wastes_padding():
    s = padding_stats(DataLoader(FrenchDataset(corpus), batch_size=4, collate_fn=collate_fn))
    assert s["total"] == 340 and s["pad"] == 87, s
''',
    solution=r'''
def padding_stats(loader, pad_idx=PAD_IDX):
    pad = total = 0
    for padded, _ in loader:
        pad += int((padded == pad_idx).sum())
        total += padded.numel()
    return {"pad": pad, "total": total, "ratio": pad / total if total else 0.0}
''')

nb.md(r'''
---
## 8. The lab exercise: sort by length, then batch

> *Create a data loader with a collate function that processes batches of French text. Sort the data set on
> sequence length. Then tokenize, numericalize and pad the sequences. Prepare the data in batches of size 4
> and print them.*

If sentences are **sorted by length**, each batch contains similar lengths, so there is little padding. The
trade-off is that sorting removes randomness. Batch order becomes fixed and short and long sentences are
never mixed, which can hurt generalization. Section 9 fixes that.
''')
nb.exercise("ex8", "Sorted DataLoader (batch size 4)", 15,
    prompt=r'''
Write `make_loader(sentences, batch_size=4, sort=True)`:
- If `sort`, order sentences by **number of tokens** (`len(tokenize_fr(s))`), ascending, with a *stable* sort
  (`sorted` is stable, so ties keep their original order).
- Return `DataLoader(FrenchDataset(...), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)`.
''',
    guide=r'''
- **`sorted(sentences, key=lambda s: len(tokenize_fr(s)))`** sorts by number of tokens. Python's sort is
  *stable*, so equal lengths keep their original order (the tests rely on this).
- Build `DataLoader(FrenchDataset(...), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)`.
  **`shuffle=False`** matters, because shuffling would undo your sorting.
- When `sort=False`, use the sentences as given.
''',
    guide_code=r'''
words = ["ccc", "a", "bb", "x", "dd"]
print(sorted(words, key=len))             # stable: 'a' before 'x', 'bb' before 'dd'
''',
    stub=r'''
def make_loader(sentences, batch_size=4, sort=True):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_eight_batches_of_four():
    loader = make_loader(corpus)
    sizes = [b[0].shape[0] for b in loader]
    assert sizes == [4, 4, 4, 4, 4, 4, 4, 2], sizes

def test_lengths_non_decreasing():
    lengths = torch.cat([l for _, l in make_loader(corpus)]).tolist()
    assert lengths == sorted(lengths), lengths

def test_first_batch_is_shortest():
    padded, lengths = next(iter(make_loader(corpus)))
    assert padded.shape == (4, 5), padded.shape
    assert [itos[i] for i in padded[0].tolist()] == ["<bos>", "Bonjour", "!", "<eos>", "<pad>"]
    assert lengths.tolist() == [4, 5, 5, 5], lengths

def test_sorting_cuts_padding():
    sorted_s, unsorted_s = padding_stats(make_loader(corpus)), padding_stats(make_loader(corpus, sort=False))
    assert sorted_s["pad"] < unsorted_s["pad"] / 3, (sorted_s, unsorted_s)
''',
    solution=r'''
def make_loader(sentences, batch_size=4, sort=True):
    if sort:
        sentences = sorted(sentences, key=lambda s: len(tokenize_fr(s)))
    return DataLoader(FrenchDataset(sentences), batch_size=batch_size,
                      shuffle=False, collate_fn=collate_fn)
''')
nb.code(r'''
try:
    for i, (padded, lengths) in enumerate(make_loader(corpus)):
        print(f"batch {i}  shape={tuple(padded.shape)}  lengths={lengths.tolist()}")
        print(padded)
        print("  ↳", " | ".join(" ".join(itos[t] for t in row if t != PAD_IDX) for row in padded.tolist()[:2]), "…\n")
    for name, flag in [("unsorted", False), ("sorted", True)]:
        s = padding_stats(make_loader(corpus, sort=flag))
        print(f"{name:>8}: {s['pad']:3d} pad / {s['total']:3d} total = {s['ratio']:.1%} wasted")
except NotImplementedError:
    print("Finish Exercises 1–8 first.")
''')

nb.md(r'''
---
## 9. Bucketing: low padding *and* randomness

Real training pipelines (fairseq, Hugging Face `group_by_length`) use **bucketing**:
1. Sort indices by length and cut them into batches. Each batch holds similar lengths, so there's little padding.
2. **Shuffle the order of the batches** every epoch, so the model doesn't always see short sentences first.

You give the DataLoader a **batch sampler**: any iterable yielding *lists of indices*, one per batch.
''')
nb.figure("l3_bucketing", "Bucketing: low padding inside each batch, random order across batches.")
nb.exercise("ex9", "A bucketing batch sampler", 20,
    prompt=r'''
Implement `BucketBatchSampler(lengths, batch_size, shuffle=True, seed=0)`:
- `__iter__`: stable-sort the indices by length (`sorted(range(len(lengths)), key=lengths.__getitem__)`), chunk
  them into lists of `batch_size` (the last chunk may be smaller), and if `shuffle`, permute **the order of the
  batches** with `torch.randperm(n_batches, generator=g)`, where `g = torch.Generator().manual_seed(seed + epoch)`.
  Then yield each batch (a `list` of ints). `epoch` starts at 0 and **increments after every full iteration**,
  so each epoch gets a different order.
- `__len__`: number of batches.

Use it as `DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn)`.
''',
    guide=r'''
Follow the bucketing figure above. New Python and PyTorch tools:

- **A batch sampler** is any object whose `__iter__` yields lists of indices and whose `__len__` returns the
  number of batches. Pass it as `DataLoader(..., batch_sampler=sampler)`.
- **Generator methods:** write `__iter__` with `yield` / `yield from batches`. Code placed **after** `yield from`
  runs once the consumer has taken every batch, which is the right moment for `self.epoch += 1`.
- **Chunking:** `[order[i:i + b] for i in range(0, len(order), b)]`.
- **Seeded shuffling:** `g = torch.Generator().manual_seed(self.seed + self.epoch)` then
  `torch.randperm(n, generator=g).tolist()` gives a reproducible permutation of `0..n-1`. Reorder with
  `[batches[i] for i in perm]`.
- **Ceiling division** for `__len__`: `(n + b - 1) // b`.
''',
    guide_code=r'''
def countdown(n):
    yield from range(n, 0, -1)
    print("  (runs after the last item was consumed)")
print(list(countdown(3)))

order = list(range(10))
print([order[i:i + 4] for i in range(0, len(order), 4)])
g = torch.Generator().manual_seed(0)
print(torch.randperm(5, generator=g).tolist(), (10 + 4 - 1) // 4)
''',
    stub=r'''
class BucketBatchSampler:
    # YOUR CODE HERE
    pass
''',
    tests=r'''
def _lens():
    return [len(tokenize_fr(s)) for s in corpus]

def test_len_and_coverage():
    LENS = _lens()
    s = BucketBatchSampler(LENS, 4)
    batches = list(s)
    assert len(s) == 8 and len(batches) == 8
    assert sorted(i for b in batches for i in b) == list(range(30))

def test_batches_are_sorted_chunks():
    LENS = _lens()
    order = sorted(range(30), key=LENS.__getitem__)
    chunks = [order[i:i + 4] for i in range(0, 30, 4)]
    got = list(BucketBatchSampler(LENS, 4))
    assert sorted(got) == sorted(chunks)
    assert list(BucketBatchSampler(LENS, 4, shuffle=False)) == chunks

def test_epochs_differ_but_are_reproducible():
    LENS = _lens()
    a = BucketBatchSampler(LENS, 4, seed=1)
    e0, e1 = list(a), list(a)
    assert e0 != e1, "each epoch should shuffle differently"
    assert list(BucketBatchSampler(LENS, 4, seed=1)) == e0, "same seed → same first epoch"

def test_low_padding_in_dataloader():
    loader = DataLoader(FrenchDataset(corpus), batch_sampler=BucketBatchSampler(_lens(), 4), collate_fn=collate_fn)
    assert padding_stats(loader)["pad"] == padding_stats(make_loader(corpus))["pad"]
''',
    solution=r'''
class BucketBatchSampler:
    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths, self.batch_size = list(lengths), batch_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def _batches(self):
        order = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        return [order[i:i + self.batch_size] for i in range(0, len(order), self.batch_size)]

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        yield from batches
        self.epoch += 1

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
''')

nb.md(r'''
---
## 10. Translation needs pairs: source **and** target

A translation model sees `(French, English)` pairs. Each language has **its own tokenizer and vocabulary**,
and the collate function pads source and target **separately**, because their lengths differ.
''')
nb.code(r'''
nlp_en = spacy.blank("en")
def tokenize_en(sentence):
    return [t.text for t in nlp_en(sentence)]

pairs = [
    ("Je t'adore.", "I love you."),
    ("Bon appétit !", "Enjoy your meal!"),
    ("Je suis heureux.", "I am happy."),
    ("Merci beaucoup !", "Thank you very much!"),
    ("Il fait beau aujourd'hui.", "The weather is nice today."),
    ("Je suis en retard.", "I am late."),
    ("Au revoir !", "Goodbye!"),
    ("Je ne sais pas.", "I don't know."),
]
''')
nb.exercise("ex10", "Collate source/target pairs", 15,
    prompt=r'''
Write `collate_pairs(batch, src_stoi, tgt_stoi)` where `batch` is a list of `(fr, en)` tuples. Tokenize French
with `tokenize_fr` and English with `tokenize_en`, numericalize each side with **its own** vocab (with
`<bos>`/`<eos>`), pad each side separately, and return `(src_padded, src_lengths, tgt_padded)`.

To use it in a DataLoader, bind the vocabs: `collate_fn=lambda b: collate_pairs(b, fr_stoi, en_stoi)`.
''',
    guide=r'''
![pairs](figures/l3_pairs.svg)

- **Unpack pairs in comprehensions:** `[fr for fr, _ in batch]` and `[en for _, en in batch]`.
- Numericalize each side with **its own** vocab, and pad each side with **its own** pad id: `src_stoi["<pad>"]`,
  `tgt_stoi["<pad>"]`.
- Return the 3-tuple `(src_padded, src_lengths, tgt_padded)`.
''',
    guide_code=r'''
batch_demo = [("un", "one"), ("deux", "two")]
print([f for f, _ in batch_demo], [e for _, e in batch_demo])
''',
    stub=r'''
def collate_pairs(batch, src_stoi, tgt_stoi):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def _vocabs():
    fr_stoi, _ = build_vocab([tokenize_fr(f) for f, _ in pairs])
    en_stoi, en_itos = build_vocab([tokenize_en(e) for _, e in pairs])
    return fr_stoi, en_stoi, en_itos

def test_shapes():
    fr_stoi, en_stoi, _ = _vocabs()
    src, src_len, tgt = collate_pairs(pairs[:2], fr_stoi, en_stoi)
    assert src.shape == (2, 6) and src_len.tolist() == [6, 5], (src.shape, src_len)
    assert tgt.shape == (2, 6), tgt.shape

def test_target_uses_english_vocab():
    fr_stoi, en_stoi, en_itos = _vocabs()
    _, _, tgt = collate_pairs([pairs[2]], fr_stoi, en_stoi)
    assert [en_itos[i] for i in tgt[0].tolist()] == ["<bos>", "I", "am", "happy", ".", "<eos>"]

def test_in_dataloader():
    fr_stoi, en_stoi, _ = _vocabs()
    loader = DataLoader(pairs, batch_size=4, collate_fn=lambda b: collate_pairs(b, fr_stoi, en_stoi))
    shapes = [(s.shape[0], t.shape[0]) for s, _, t in loader]
    assert shapes == [(4, 4), (4, 4)], shapes
''',
    solution=r'''
def collate_pairs(batch, src_stoi, tgt_stoi):
    src = [numericalize(tokenize_fr(fr), src_stoi) for fr, _ in batch]
    tgt = [numericalize(tokenize_en(en), tgt_stoi) for _, en in batch]
    src_padded, src_lengths = pad_batch(src, src_stoi["<pad>"])
    tgt_padded, _ = pad_batch(tgt, tgt_stoi["<pad>"])
    return src_padded, src_lengths, tgt_padded
''')
nb.code(r'''
try:
    fr_stoi, fr_itos = build_vocab([tokenize_fr(f) for f, _ in pairs])
    en_stoi, en_itos = build_vocab([tokenize_en(e) for _, e in pairs])
    loader = DataLoader(pairs, batch_size=4, shuffle=True, collate_fn=lambda b: collate_pairs(b, fr_stoi, en_stoi))
    src, src_len, tgt = next(iter(loader))
    print("src", tuple(src.shape), "\n", src, "\ntgt", tuple(tgt.shape), "\n", tgt)
except NotImplementedError:
    print("Finish Exercise 10 first.")
''')
nb.md(r'''
In a real seq2seq model, `tgt[:, :-1]` (starting with `<bos>`) is the **decoder input** and `tgt[:, 1:]`
(ending with `<eos>`) is the **label**. This is *teacher forcing*, and it's why `<bos>`/`<eos>` exist.

---
## Recap

- **`Dataset`** = what the data is (`__len__`, `__getitem__`). **`DataLoader`** = batching, shuffling, collation.
- **`collate_fn`** is where text becomes tensors: tokenize → numericalize → pad.
- **Vocab**: specials at fixed ids, then tokens by frequency, `<unk>` for anything unseen, `min_freq` to prune.
- **Dynamic padding** pads to the batch max. **Sorting/bucketing** by length cuts padding on this corpus from ~26% to ~2%.
- **Bucketing + shuffled batch order** keeps efficiency *and* randomness.
- Translation = **two vocabularies, two paddings**, and `<bos>`/`<eos>` drive teacher forcing.

### Your score
''')
nb.code("lab.scoreboard()")

nb.write(COURSE, "lab_03_nlp_data_loader.ipynb")
