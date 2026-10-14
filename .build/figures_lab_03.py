"""SVG figures for Lab 3 (written to <course>/figures/). Ids are the real ones from the lab's vocab."""
from pathlib import Path
from svgkit import SVG

OUT = Path(__file__).resolve().parent.parent / "generative-ai-llm-architecture-data-preparation" / "figures"
PADK = lambda r, c, v: "pad" if v == "1" else ("orange" if v in ("2", "3") else "blue")


def pipeline():
    s = SVG(960, 250, "Text to tensor pipeline")
    s.title_bar("From a raw sentence to a batch tensor")
    s.box(20, 70, 140, 40, "\"Je t'adore.\"", "white", 12, mono=True)
    s.note(90, 126, "raw string", 10, "middle")
    s.arrow(162, 90, 186, 90, "tokenize", lx=176, ly=62)
    s.chips(190, 77, ["Je", "t'", "adore", "."], "blue")
    s.note(270, 126, "tokens", 10, "middle")
    s.arrow(372, 90, 396, 90, "numericalize", lx=384, ly=62)
    s.chips(400, 77, ["2", "5", "109", "45", "4", "3"], kinds=["orange"] + ["blue"] * 4 + ["orange"], w=36)
    s.note(520, 126, "ids, wrapped in <bos>=2 … <eos>=3", 10, "middle")
    s.arrow(648, 90, 672, 90, "pad", lx=660, ly=62)
    s.chips(676, 77, ["2", "5", "109", "45", "4", "3", "1", "1"], kinds=["orange"] + ["blue"] * 4 + ["orange", "pad", "pad"], w=30, gap=3)
    s.note(800, 126, "<pad>=1 up to the batch max", 10, "middle")
    s.box(20, 160, 920, 74, "", "yellow")
    s.text(34, 180, "Dataset returns raw strings  →  DataLoader groups 4 of them  →  collate_fn does tokenize → numericalize → pad", 12)
    s.text(34, 206, "→ one LongTensor of shape (batch=4, T) where T = longest sequence in THIS batch", 12, mono=True)
    s.save(OUT / "l3_pipeline.svg")


def dataset_loader():
    s = SVG(960, 310, "Dataset and DataLoader")
    s.title_bar("Dataset = indexable samples; DataLoader = sampler + collate_fn")
    s.box(20, 64, 300, 220, "", "blue")
    s.text(170, 82, "class FrenchDataset(Dataset)", 12, "middle", mono=True, weight="bold")
    for i, l in enumerate(["def __init__(self, sentences):", "    store the list on self", "",
                           "def __len__(self):", "    → how many samples", "",
                           "def __getitem__(self, idx):", "    → sample number idx"]):
        s.text(34, 106 + i * 19, l, 11, mono=True, color="#334155" if l.startswith("    ") else "#0f172a")
    s.arrow(322, 150, 362, 150)
    s.text(380, 84, "dataset[i] for the chosen indices", 12, weight="bold")
    for i, (idx, sent) in enumerate([(0, "Ceci est une phrase."), (1, "C'est un autre…"), (2, "Voici une…"), (3, "Il fait beau…")]):
        s.chip(380, 100 + i * 32, f"[{idx}] {sent}", "white", 26, 11, w=220)
    s.note(490, 238, "sampler: 0,1,2,3 (or shuffled)", 10, "middle")
    s.arrow(604, 150, 676, 150, "collate_fn", lx=640, ly=134)
    s.box(680, 100, 260, 100, "", "green")
    s.text(810, 120, "batch", 12, "middle", weight="bold")
    s.text(694, 146, "default: list of 4 strings", 11, color="#334155")
    s.text(694, 170, "yours: (padded LongTensor,", 11, mono=True)
    s.text(694, 188, "        lengths)", 11, mono=True)
    s.note(20, 298, "Subclassing torch.utils.data.Dataset only requires __len__ and __getitem__; DataLoader handles the rest.", 11)
    s.save(OUT / "l3_dataset_loader.svg")


def spacy_fr():
    s = SVG(960, 200, "spaCy French tokenizer")
    s.title_bar("tokenize_fr(): a spaCy Doc is a sequence of Token objects; take each token's .text")
    s.box(20, 70, 260, 40, "nlp_fr(\"C'est un autre exemple.\")", "white", 11, mono=True)
    s.arrow(282, 90, 312, 90)
    s.text(320, 74, "Doc", 11, weight="bold")
    x = s.chips(320, 80, ["Token"] * 6, "gray", w=46, gap=4, size=10)
    s.arrow(x + 4, 93, x + 34, 93, ".text", ly=80)
    s.chips(x + 40, 80, ["C'", "est", "un", "autre", "exemple", "."], "blue")
    s.box(20, 136, 920, 44, "[tok.text for tok in nlp_fr(sentence)]   — a list of plain str, exactly like .split() would return", "yellow", 12, mono=True)
    s.save(OUT / "l3_spacy_fr.svg")


def vocab():
    s = SVG(980, 340, "Building a vocabulary")
    s.title_bar("build_vocab(): count → filter → sort → number")
    s.box(20, 66, 210, 160, "", "gray")
    s.text(125, 84, "Counter over ALL tokens", 11, "middle", weight="bold")
    for i, (t, c) in enumerate([(".", 19), ("Je", 11), ("!", 8), ("suis", 7), ("est", 5), ("de", 4), ("le", 4)]):
        s.text(40, 106 + i * 17, f"{t!r:<8} {c}", 11, mono=True)
    s.arrow(232, 146, 262, 146)
    s.box(264, 100, 170, 92, "keep count >= min_freq\nand not a special\n\nsort key: (-count, t)", "blue", 11)
    s.arrow(436, 146, 466, 146)
    s.text(470, 74, "itos (list): index → string", 12, weight="bold")
    toks = ["<unk>", "<pad>", "<bos>", "<eos>", ".", "Je", "!", "suis", "est", "…"]
    x = 470
    for i, t in enumerate(toks):
        w = s.chip(x, 92, t, "orange" if i < 4 else "blue", 28, 11)
        if t != "…":
            s.note(x + w / 2, 134, str(i), 10, "middle")
        x += w + 4
    s.note(470, 156, "specials first (fixed ids 0–3), then by count ↓; ties A→Z", 11)
    s.text(470, 196, "stoi (dict): string → index  = the inverse", 12, weight="bold")
    s.text(470, 220, "{t: i for i, t in enumerate(itos)}", 12, mono=True)
    s.text(470, 244, "stoi['Je'] == 5     itos[5] == 'Je'", 12, mono=True, color="#1d4ed8")
    s.box(20, 270, 940, 54, "", "yellow")
    s.text(34, 288, "Counting tokens across a list of lists with a nested generator:", 12)
    s.text(34, 308, "Counter(tok for toks in token_lists for tok in toks)", 12, mono=True)
    s.save(OUT / "l3_vocab.svg")


def numericalize():
    s = SVG(960, 230, "Numericalization")
    s.title_bar("numericalize(): look every token up; unknown → <unk>")
    s.chips(20, 74, ["je", "danse"], "blue", w=70)
    s.arrow(170, 87, 220, 87, "stoi.get(t, unk)", ly=72)
    s.chip(230, 74, "4", "blue", w=40)
    s.chip(274, 74, "0", "red", w=40)
    s.note(340, 87, "'danse' not in stoi → stoi['<unk>'] = 0", 11)
    s.text(20, 140, "add_bos_eos=True:", 12, weight="bold")
    s.chips(170, 126, ["2", "4", "0", "3"], kinds=["orange", "blue", "red", "orange"], w=40)
    s.note(350, 140, "[stoi['<bos>']] + ids + [stoi['<eos>']]", 11)
    s.box(20, 172, 920, 40, "torch.tensor(ids, dtype=torch.long)   → 1-D LongTensor (embedding layers need int64 indices)",
          "yellow", 12, mono=True)
    s.save(OUT / "l3_numericalize.svg")


def padding():
    s = SVG(960, 300, "Padding a batch")
    s.title_bar("pad_batch(): make a ragged list rectangular")
    seqs = [["2", "5", "3"], ["2", "5", "109", "45", "3"], ["2", "3"]]
    s.text(20, 72, "seqs (list of 1-D tensors)", 12, weight="bold")
    for r, row in enumerate(seqs):
        s.grid(20, 84 + r * 30, [row], cw=40, ch=28, kind_of=PADK)
    s.arrow(232, 128, 292, 128, "pad_sequence(...,\nbatch_first=True,\npadding_value=1)", ly=168, size=10)
    s.text(300, 72, "padded: shape (3, 5)", 12, weight="bold")
    s.grid(300, 84, [r + ["1"] * (5 - len(r)) for r in seqs], cw=40, ch=30, kind_of=PADK)
    s.text(520, 72, "lengths", 12, weight="bold")
    s.grid(520, 84, [["3"], ["5"], ["2"]], cw=40, ch=30)
    s.note(580, 99, "← len(s) for each original s,", 11)
    s.note(580, 117, "   as a LongTensor", 11)
    s.box(20, 196, 920, 86, "", "yellow")
    s.text(34, 216, "batch_first=True  → (batch, time) = (3, 5)   ← Transformers expect this", 12, mono=True)
    s.text(34, 238, "batch_first=False → (time, batch) = (5, 3)   ← the default, from the RNN era", 12, mono=True)
    s.text(34, 262, "Padding goes on the RIGHT, up to the longest sequence in this batch (dynamic padding).", 12)
    s.save(OUT / "l3_padding.svg")


def collate():
    s = SVG(960, 230, "collate_fn")
    s.title_bar("collate_fn(batch): the glue that DataLoader calls for every batch")
    s.box(20, 70, 170, 70, "batch = [\n \"Je t'adore.\",\n \"Bonjour !\" ]", "white", 11, mono=True)
    s.arrow(192, 105, 222, 105)
    s.box(224, 64, 300, 82, "", "blue")
    s.text(374, 82, "for each sentence s:", 11, "middle", weight="bold")
    s.text(238, 104, "numericalize(tokenize_fr(s), stoi)", 11, mono=True)
    s.text(238, 126, "→ [2,5,109,45,4,3]   [2,31,6,3]", 11, mono=True, color="#334155")
    s.arrow(526, 105, 556, 105)
    s.box(558, 76, 140, 58, "pad_batch(seqs,\n PAD_IDX)", "green", 11, mono=True)
    s.arrow(700, 105, 730, 105)
    s.box(732, 64, 208, 82, "(padded (2, 6),\n lengths [6, 4])", "purple", 11, mono=True)
    s.box(20, 170, 920, 40, "DataLoader(dataset, batch_size=4, collate_fn=collate_fn)  — pass the function itself, don't call it",
          "yellow", 12, mono=True)
    s.save(OUT / "l3_collate.svg")


def sorted_vs_unsorted():
    s = SVG(980, 330, "Sorting reduces padding")
    s.title_bar("Why sort by length? Compare the first batch of 4", "each cell = one token id; gray = <pad> (wasted compute)")
    un = [10 - 3, 10, 7, 7]
    so = [4, 5, 5, 5]
    s.text(20, 82, "unsorted: corpus[0:4]  → T = 10", 12, weight="bold")
    for r, n in enumerate(un):
        s.grid(20, 94 + r * 24, [["■"] * n + ["·"] * (10 - n)], cw=26, ch=24,
               kind_of=lambda rr, c, v: "pad" if v == "·" else "blue", size=9)
    s.text(20, 206, "9 pad of 40 cells", 12, color="#dc2626", weight="bold")
    s.text(520, 82, "sorted by length  → T = 5", 12, weight="bold")
    for r, n in enumerate(so):
        s.grid(520, 94 + r * 24, [["■"] * n + ["·"] * (5 - n)], cw=26, ch=24,
               kind_of=lambda rr, c, v: "pad" if v == "·" else "green", size=9)
    s.text(520, 206, "1 pad of 20 cells", 12, color="#16a34a", weight="bold")
    s.box(20, 232, 940, 82, "", "yellow")
    s.text(34, 252, "Whole corpus, batch_size=4:   unsorted 87 pad / 340 = 25.6% wasted     sorted 5 pad / 258 = 1.9% wasted", 12)
    s.text(34, 276, "pad = int((padded == pad_idx).sum())     total = padded.numel()     summed over every batch", 12, mono=True)
    s.text(34, 298, "sorted(sentences, key=lambda s: len(tokenize_fr(s)))  — stable: ties keep their original order", 12, mono=True)
    s.save(OUT / "l3_sorted_vs_unsorted.svg")


def bucketing():
    s = SVG(980, 360, "Bucketing batch sampler")
    s.title_bar("BucketBatchSampler: sorted chunks, shuffled order (a new order every epoch)")
    s.text(20, 74, "1) indices sorted by length (stable)", 12, weight="bold")
    order = [28, 7, 20, 21, 6, 10, 12, 29, 0, 2, 3, 13]
    s.chips(20, 86, [str(i) for i in order] + ["…"], "blue", w=36, gap=3)
    s.text(20, 140, "2) chunk into batches of batch_size", 12, weight="bold")
    x = 20
    for b in range(4):
        x = s.chips(x, 152, [str(i) for i in order[b * 4:(b + 1) * 4]] if b < 3 else ["…"], "green", w=34, gap=2) + 4
        s.note(x - 74, 194, f"B{b}" if b < 3 else "… B7", 10, "middle")
        x += 16
    s.text(20, 222, "3) shuffle the ORDER of the batches", 12, weight="bold")
    s.chips(20, 234, ["B5", "B0", "B7", "B2", "B6", "B1", "B4", "B3"], "purple", w=40, gap=4)
    s.note(400, 247, "epoch 0: torch.randperm(8, generator=g) with g seeded by seed + 0", 10)
    s.chips(20, 270, ["B3", "B6", "B1", "B4", "B0", "B7", "B2", "B5"], "purple", w=40, gap=4)
    s.note(400, 283, "epoch 1: seed + 1  → a different order", 10)
    s.box(560, 64, 400, 150, "", "yellow")
    for i, l in enumerate(["class BucketBatchSampler:", "  __init__: store lengths, batch_size,", "            shuffle, seed; epoch = 0",
                           "  __iter__: build batches (1, 2)", "            if shuffle: reorder (3)", "            yield from batches",
                           "            self.epoch += 1   ← after yielding", "  __len__:  ceil(n / batch_size)"]):
        s.text(574, 82 + i * 17, l, 11, mono=True)
    s.note(20, 334, "Within a batch: similar lengths → little padding.   Across batches: random order → no length bias.", 11)
    s.save(OUT / "l3_bucketing.svg")


def pairs():
    s = SVG(980, 330, "Source/target pairs")
    s.title_bar("collate_pairs(): two languages, two vocabularies, two paddings")
    s.box(20, 70, 210, 64, "batch = [\n (\"Je t'adore.\", \"I love you.\"),\n …]", "white", 10, mono=True)
    s.arrow(232, 92, 268, 80)
    s.arrow(232, 112, 268, 150)
    s.box(270, 60, 300, 40, "tokenize_fr → numericalize(…, src_stoi)", "blue", 11, mono=True)
    s.box(270, 130, 300, 40, "tokenize_en → numericalize(…, tgt_stoi)", "green", 11, mono=True)
    s.arrow(572, 80, 610, 80)
    s.arrow(572, 150, 610, 150)
    s.box(612, 60, 348, 40, "pad_batch → src_padded (B, Ts), src_lengths", "blue", 11, mono=True)
    s.box(612, 130, 348, 40, "pad_batch → tgt_padded (B, Tt)", "green", 11, mono=True)
    s.note(612, 186, "Ts ≠ Tt in general: each side is padded on its own, with its own <pad> id", 10)
    s.text(20, 222, "Why <bos>/<eos>? Teacher forcing in a seq2seq decoder:", 12, weight="bold")
    s.chips(20, 236, ["<bos>", "I", "love", "you", "."], "green")
    s.note(300, 249, "tgt[:, :-1]  → decoder INPUT", 11)
    s.chips(20 + 54, 272, ["I", "love", "you", ".", "<eos>"], "orange")
    s.note(354, 285, "tgt[:, 1:]   → LABELS (the next token at each position)", 11)
    s.save(OUT / "l3_pairs.svg")


if __name__ == "__main__":
    for f in [pipeline, dataset_loader, spacy_fr, vocab, numericalize, padding, collate, sorted_vs_unsorted,
              bucketing, pairs]:
        f()
    print("lab 3 figures →", OUT)
