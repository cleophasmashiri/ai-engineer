"""SVG figures for Lab 2 (written to <course>/figures/)."""
from pathlib import Path
from svgkit import SVG

OUT = Path(__file__).resolve().parent.parent / "generative-ai-llm-architecture-data-preparation" / "figures"


def granularities():
    s = SVG(940, 330, "Three tokenization granularities")
    s.title_bar("Three ways to cut \"unbelievably fun\" into tokens")
    rows = [("Word-based", "blue", ["unbelievably", "fun"],
             "✔ meaningful units   ✘ huge vocabulary; unseen words → [UNK]"),
            ("Character-based", "orange", list("unbelievably") + ["f", "u", "n"],
             "✔ tiny vocabulary, nothing is ever unknown   ✘ long sequences, little meaning per token"),
            ("Subword-based", "green", ["un", "##believ", "##ably", "fun"],
             "✔ frequent words whole, rare words in reusable pieces → used by every modern LLM")]
    for i, (name, k, toks, note) in enumerate(rows):
        y = 70 + i * 86
        s.box(20, y, 150, 34, name, k, 13, bold=True)
        s.chips(186, y + 4, toks, k, gap=4, h=26, size=12)
        s.note(186, y + 50, note, 11)
    s.save(OUT / "l2_granularities.svg")


def regex():
    s = SVG(940, 300, "Regex tokenization")
    s.title_bar('re.findall(r"\\w+|[^\\w\\s]", text): how the regex scans "I\'m here, ok!"')
    text = "I'm here, ok!"
    kinds = []
    for c in text:
        kinds.append("blue" if c.isalnum() else ("gray" if c.isspace() else "orange"))
    x = 20
    for c, k in zip(text, kinds):
        s.chip(x, 70, "␣" if c == " " else c, k, 30, 13, w=34)
        x += 38
    s.note(20, 118, "blue = \\w (letter, digit, _)    orange = neither \\w nor whitespace → [^\\w\\s]    gray = whitespace → matched by nothing, skipped", 11)
    s.text(20, 156, "matches, left to right:", 12, weight="bold")
    toks = [("I", "blue", "\\w+"), ("'", "orange", "[^\\w\\s]"), ("m", "blue", "\\w+"), ("here", "blue", "\\w+"),
            (",", "orange", "[^\\w\\s]"), ("ok", "blue", "\\w+"), ("!", "orange", "[^\\w\\s]")]
    x = 20
    for t, k, why in toks:
        w = s.chip(x, 172, t, k, 30, 13, w=max(44, len(why) * 8))
        s.note(x + w / 2, 216, why, 10, "middle")
        x += w + 8
    s.box(20, 240, 900, 44, "", "yellow")
    s.text(34, 262, "A | B tries A first, then B, at each position. \\w+ is greedy (takes the whole run); [^\\w\\s] takes exactly one character.", 12)
    s.save(OUT / "l2_regex.svg")


def chars():
    s = SVG(940, 190, "Character tokenization")
    s.title_bar('char_tokenize("hi there\\n!"): every character, whitespace dropped by default')
    text = "hi there\n!"
    x = 20
    for c in text:
        ws = c.isspace()
        lab = {" ": "␣", "\n": "\\n"}.get(c, c)
        s.chip(x, 70, lab, "pad" if ws else "orange", 30, 13, w=40)
        if ws:
            s.line(x + 4, 74, x + 36, 96, "#dc2626", sw=2)
        x += 46
    s.note(20, 124, "c.isspace() is True for ' ', '\\n', '\\t' → skip them unless keep_spaces=True", 11)
    s.note(20, 150, "result: ['h', 'i', 't', 'h', 'e', 'r', 'e', '!']  — a list comprehension with an if-filter", 11)
    s.save(OUT / "l2_chars.svg")


def oov():
    s = SVG(940, 270, "Out-of-vocabulary rate")
    s.title_bar("vocab_and_oov(): which test tokens has the model never seen?")
    s.box(20, 70, 260, 120, "", "blue")
    s.text(150, 88, "train vocabulary = set(train_tokens)", 11, "middle", weight="bold")
    s.chips(36, 112, ["a", "b"], "blue", w=40)
    s.note(36, 160, "train_tokens = ['a', 'b', 'a']\n→ distinct: {'a', 'b'} → vocab_size = 2", 11)
    s.text(320, 88, "test_tokens (repeats count!)", 12, weight="bold")
    toks = [("a", True), ("c", False), ("c", False), ("b", True)]
    x = 320
    for t, ok in toks:
        s.chip(x, 104, t, "green" if ok else "red", 30, 13, w=44)
        s.note(x + 22, 150, "in" if ok else "OOV", 10, "middle")
        x += 52
    s.box(560, 80, 360, 90, "", "yellow")
    s.text(740, 106, "oov_rate = missing / len(test_tokens)", 12, "middle", mono=True)
    s.text(740, 132, "= 2 / 4 = 0.5", 13, "middle", mono=True, weight="bold")
    s.note(740, 156, "empty test list → return 0.0 (avoid dividing by 0)", 10, "middle")
    s.note(20, 228, "Word vocabularies miss new words (high OOV). Character vocabularies almost never miss. Subwords aim for both.", 11)
    s.save(OUT / "l2_oov.svg")


def bpe_pairs():
    s = SVG(960, 360, "BPE pair counting")
    s.title_bar("most_frequent_pair(): count adjacent pairs, weighted by word frequency")
    s.text(20, 72, "word_freqs  (tuple of symbols → count)", 12, weight="bold")
    words = [(("l", "o", "w", "</w>"), 5), (("n", "e", "w", "e", "s", "t", "</w>"), 6), (("w", "i", "d", "e", "s", "t", "</w>"), 3)]
    for i, (w, f) in enumerate(words):
        y = 88 + i * 40
        x = s.chips(20, y, list(w), "blue", gap=3, w=None)
        s.text(x + 10, y + 13, f"× {f}", 12, weight="bold")
    # zip illustration
    s.text(20, 222, "zip(word, word[1:]) walks the adjacent pairs:", 12, weight="bold")
    w = ["n", "e", "w", "e", "s", "t", "</w>"]
    s.chips(20, 238, w, "blue", gap=3)
    s.chips(20, 270, w[1:], "purple", gap=3)
    s.note(20, 318, "read column by column → (n,e) (e,w) (w,e) (e,s) (s,t) (t,</w>)", 11)
    # counts
    s.box(560, 70, 380, 270, "", "gray")
    s.text(750, 88, "pairs Counter: add freq for every occurrence", 11, "middle", weight="bold")
    counts = [("(e, s)", "6 + 3 = 9"), ("(s, t)", "6 + 3 = 9"), ("(t, </w>)", "6 + 3 = 9"), ("(w, e)", "6"),
              ("(l, o)", "5"), ("(o, w)", "5"), ("(w, </w>)", "5"), ("…", "")]
    for i, (p, c) in enumerate(counts):
        y = 108 + i * 24
        s.text(580, y, p, 11, mono=True, color="#9a3412" if i == 0 else "#0f172a", weight="bold" if i == 0 else "normal")
        s.text(700, y, c, 11, mono=True)
    s.note(580, 306, "tie at 9 → pick the lexicographically smallest:", 10)
    s.text(580, 324, "min(pairs, key=lambda p: (-pairs[p], p)) → ('e', 's')", 10, mono=True)
    s.save(OUT / "l2_bpe_pairs.svg")


def bpe_merge():
    s = SVG(960, 300, "BPE merge")
    s.title_bar("merge_pair(('e','s'), word_freqs): scan each word left to right with an index i")
    w = ["n", "e", "w", "e", "s", "t", "</w>"]
    x = 20
    for i, c in enumerate(w):
        hit = i in (3, 4)
        s.chip(x, 76, c, "orange" if hit else "blue", w=44)
        s.note(x + 22, 116, f"i={i}", 10, "middle")
        x += 50
    s.arrow(380, 92, 430, 92)
    x = s.chips(440, 76, ["n", "e", "w", "es", "t", "</w>"], kinds=["blue"] * 3 + ["orange"] + ["blue"] * 2, w=44)
    s.box(20, 140, 440, 110, "", "yellow")
    for i, l in enumerate(["while i < len(word):",
                           "  if word[i] == a and word[i+1] == b (and i+1 exists):",
                           "      new.append(a + b);  i += 2      ← skip both",
                           "  else:",
                           "      new.append(word[i]); i += 1"]):
        s.text(34, 160 + i * 19, l, 11, mono=True)
    s.box(480, 140, 460, 110, "", "red")
    s.text(494, 160, "Why i += 2 matters:", 12, weight="bold")
    s.chips(494, 176, ["a", "a", "a", "a"], "blue", w=34)
    s.arrow(650, 189, 690, 189)
    s.chips(700, 176, ["aa", "aa"], "orange", w=44)
    s.note(494, 226, "merging ('a','a') must give ('aa','aa'), not overlapping pairs.\nBuild a NEW dict, and add counts if two words become identical.", 10)
    s.note(20, 278, "Keys are tuples (hashable): convert the list back with tuple(new) before storing it in the new dict.", 11)
    s.save(OUT / "l2_bpe_merge.svg")


def bpe_train():
    s = SVG(940, 240, "Training BPE")
    s.title_bar("train_bpe(): repeat find → merge, recording every merge")
    s.box(20, 90, 236, 50, "wf = corpus_to_word_freqs(words)", "gray", 10, mono=True)
    s.arrow(258, 115, 282, 115)
    s.box(284, 90, 200, 50, "pair = most_frequent_pair(wf)", "blue", 10, mono=True)
    s.arrow(486, 115, 532, 115, "pair found")
    s.box(534, 90, 180, 50, "merges.append(pair)\nwf = merge_pair(pair, wf)", "green", 10, mono=True)
    s.arrow(624, 142, 384, 142, path="M624,142 C624,190 384,190 384,142")
    s.note(486, 196, "repeat up to num_merges times", 10, "middle")
    s.arrow(384, 88, 820, 96, path="M384,88 C384,60 820,60 820,94", dashed=True)
    s.note(560, 58, "pair is None → break early", 10, "middle")
    s.box(740, 96, 180, 50, "return (merges, wf)", "purple", 11, mono=True)
    s.save(OUT / "l2_bpe_train.svg")


def subwords():
    s = SVG(960, 300, "Subword markers")
    s.title_bar("merge_subwords(): two conventions for word boundaries")
    s.text(20, 70, "WordPiece (BERT): ## = glue onto the previous word", 12, weight="bold")
    toks = ["un", "##bel", "##ie", "##va", "##bly", ",", "the"]
    kinds = ["blue", "purple", "purple", "purple", "purple", "blue", "blue"]
    s.chips(20, 84, toks, kinds=kinds)
    s.arrow(440, 97, 480, 97)
    s.chips(490, 84, ["unbelievably", ",", "the"], "green")
    s.note(20, 128, "starts with ## → words[-1] += t[2:]       otherwise → words.append(t)", 11)
    s.text(20, 170, "SentencePiece (XLNet, T5, Llama): ▁ = a space before → starts a NEW word", 12, weight="bold")
    toks = ["▁I", "'", "m", "▁learning", "▁token", "ization", "!"]
    kinds = ["blue", "purple", "purple", "blue", "blue", "purple", "purple"]
    s.chips(20, 184, toks, kinds=kinds)
    s.arrow(500, 197, 540, 197)
    s.chips(550, 184, ["I'm", "learning", "tokenization!"], "green")
    s.note(20, 228, "starts with ▁ → words.append(t[1:])       otherwise → words[-1] += t   (or append if words is empty)", 11)
    s.note(20, 268, "blue = starts a word, purple = continues one. Any other scheme name → raise ValueError(...)", 11)
    s.save(OUT / "l2_subwords.svg")


def interface():
    s = SVG(940, 270, "One tokenizer interface")
    s.title_bar("make_tokenizers(): four libraries behind one interface  str → list[str]")
    libs = [("nltk", "word_tokenize(s)", "already list[str]", "blue"),
            ("spacy", "nlp_en(s) → Doc of Tokens", "need [t.text for t in doc]", "orange"),
            ("bert", "bert_tok.tokenize(s)", "already list[str]", "purple"),
            ("xlnet", "xlnet_tok.tokenize(s)", "already list[str]", "green")]
    for i, (k, call, note, kind) in enumerate(libs):
        y = 66 + i * 48
        s.chip(20, y, f'"{k}"', kind, 34, 12, w=80)
        s.arrow(102, y + 17, 132, y + 17)
        s.box(134, y, 280, 34, call, kind, 11, mono=True)
        s.note(426, y + 17, note, 11)
    s.box(660, 100, 260, 100, "", "yellow")
    s.text(790, 122, "a dict of functions", 12, "middle", weight="bold")
    s.text(674, 148, "tok = make_tokenizers()", 11, mono=True)
    s.text(674, 168, 'tok["bert"]("Hi!")', 11, mono=True)
    s.text(674, 188, "→ ['hi', '!']", 11, mono=True)
    s.save(OUT / "l2_interface.svg")


def freq():
    s = SVG(960, 260, "Token frequencies")
    s.title_bar("token_frequencies(): filter → count → sort → cut")
    steps = [("tokens", '["The","cat",",","the",…]', "gray"),
             ("lowercase?", "[t.lower() for t in …]", "blue"),
             ("drop_punct?", 'keep t if re.search(r"\\w", t)', "blue"),
             ("Counter(...)", "{'the': 3, 'cat': 2, …}", "green"),
             ("sorted(items,\n key=(-count, token))", "count ↓, then A→Z", "purple"),
             ("[:top_n]", "if top_n is not None", "orange")]
    x = 20
    for i, (t, sub, k) in enumerate(steps):
        w = 140
        s.box(x, 70, w, 54, t, k, 11, mono=True)
        s.note(x + w / 2, 142, sub, 9.5, "middle")
        if i < len(steps) - 1:
            s.arrow(x + w + 1, 97, x + w + 12, 97)
        x += w + 12
    s.box(20, 172, 920, 70, "", "yellow")
    s.text(34, 192, "Sorting by two criteria: use a key that returns a tuple. Negate the count so bigger counts come first.", 12)
    s.text(34, 218, "sorted([('b', 1), ('a', 2), ('c', 2)], key=lambda kv: (-kv[1], kv[0]))  →  [('a', 2), ('c', 2), ('b', 1)]", 11, mono=True)
    s.save(OUT / "l2_freq.svg")


def timing():
    s = SVG(940, 240, "Timing with datetime")
    s.title_bar("time_tokenizer(): time the whole loop once, then divide")
    s.line(40, 124, 900, 124, "#64748b", sw=2)
    s.chip(20, 60, "start = datetime.now()", "blue", 26, 11)
    s.line(60, 86, 60, 124, "#2563eb", dashed=True)
    for i in range(4):
        x = 150 + i * 150
        s.box(x, 108, 132, 32, f"fn(text)  #{i + 1}", "green", 11, mono=True)
    s.chip(730, 60, "end = datetime.now()", "blue", 26, 11)
    s.line(800, 86, 800, 124, "#2563eb", dashed=True)
    s.note(450, 160, "repeats = 4 calls inside ONE timed interval", 11, "middle")
    s.box(20, 180, 900, 44, "avg = (end - start).total_seconds() / repeats      →  return (tokens_from_last_call, avg)",
          "yellow", 12, mono=True)
    s.save(OUT / "l2_timing.svg")


def compare():
    s = SVG(1000, 270, "Comparing tokenizers")
    s.title_bar("compare_tokenizers(): one row (dict) per tokenizer, in sorted-name order")
    s.box(20, 70, 210, 44, "for name in sorted(tokenizers)", "gray", 10, mono=True)
    s.arrow(232, 92, 250, 92)
    s.box(252, 70, 250, 44, "tokens, secs = time_tokenizer(\n   tokenizers[name], text, repeats)", "blue", 10, mono=True)
    s.arrow(504, 92, 524, 92)
    s.box(526, 62, 456, 96, "", "green")
    for i, l in enumerate(['{"name": name,', ' "n_tokens": len(tokens),', ' "n_unique": len(set(tokens)),',
                           ' "top": token_frequencies(tokens, top_n, lowercase=True, drop_punct=True),', ' "seconds": secs}']):
        s.text(536, 76 + i * 17, l, 9.5, mono=True)
    s.text(20, 190, "rows →", 12, weight="bold")
    s.grid(80, 172, [["name", "n_tokens", "n_unique", "top", "seconds"], ["bert", "…", "…", "…", "…"], ["nltk", "…", "…", "…", "…"],
                     ["spacy", "…", "…", "…", "…"], ["xlnet", "…", "…", "…", "…"]],
           cw=80, ch=20, kind_of=lambda r, c, v: "yellow" if r == 0 else ("green" if c == 0 else "white"), size=10)
    s.note(500, 220, "returned as a list of dicts, then printed as a table", 11)
    s.save(OUT / "l2_compare.svg")


if __name__ == "__main__":
    for f in [granularities, regex, chars, oov, bpe_pairs, bpe_merge, bpe_train, subwords, interface, freq,
              timing, compare]:
        f()
    print("lab 2 figures →", OUT)
