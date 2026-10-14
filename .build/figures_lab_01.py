"""SVG figures for Lab 1 (written to <course>/figures/)."""
from pathlib import Path
from svgkit import SVG

OUT = Path(__file__).resolve().parent.parent / "generative-ai-llm-architecture-data-preparation" / "figures"


def families():
    s = SVG(900, 330, "The three Transformer families")
    s.title_bar("The three Transformer families", "Same building blocks (attention + feed-forward), wired differently")
    cols = [(20, "Encoder-only", "BERT, RoBERTa", "blue",
             ["text: the whole sentence", "Encoder\n(every token sees every token)", "one vector per token\n→ classify / embed / search"]),
            (315, "Decoder-only", "GPT, Llama, Qwen", "purple",
             ["prompt", "Decoder\n(each token sees only the past)", "next token → append → repeat"])]
    for x0, name, sub, kind, steps in cols:
        s.box(x0, 64, 270, 46, name, kind, 14, bold=True, sub=sub)
        ys = [130, 186, 252]
        for i, (lab, y) in enumerate(zip(steps, ys)):
            s.box(x0 + 15, y, 240, 42 if i else 32, lab, "white" if i != 1 else kind, 12)
            if i:
                s.arrow(x0 + 135, ys[i - 1] + (32 if i == 1 else 42), x0 + 135, y - 2)
    x0 = 610
    s.box(x0, 64, 270, 46, "Encoder-decoder", "green", 14, bold=True, sub="T5, FLAN-T5, BART, BlenderBot")
    s.box(x0 + 15, 130, 240, 32, "input text (prompt)", "white", 12)
    s.box(x0 + 10, 186, 118, 52, "Encoder\nreads prompt once", "green", 12)
    s.box(x0 + 142, 186, 118, 52, "Decoder\nwrites answer", "green", 12)
    s.arrow(x0 + 69, 162, x0 + 69, 184)
    s.arrow(x0 + 128, 212, x0 + 140, 212)
    s.note(x0 + 135, 248, "cross-attention", 10, "middle")
    s.box(x0 + 15, 262, 240, 32, "output text, token by token", "white", 12)
    s.arrow(x0 + 201, 238, x0 + 201, 260)
    s.chip(x0 + 182, 36, "← this lab", "yellow", 22, 11, mono=False)
    s.save(OUT / "l1_transformer_families.svg")


def hub_loading():
    s = SVG(900, 300, "What from_pretrained() does")
    s.title_bar("What .from_pretrained(\"google/flan-t5-small\") does")
    s.box(20, 62, 250, 220, "", "gray")
    s.text(145, 80, "🤗 Hub: google/flan-t5-small", 12, "middle", weight="bold")
    files = [("config.json", "architecture numbers"), ("model.safetensors", "the weights, ~300 MB"),
             ("spiece.model", "tokenizer vocabulary"), ("tokenizer_config.json", "special tokens")]
    for i, (f, d) in enumerate(files):
        s.chip(34, 100 + i * 44, f, "white", 26, 11)
        s.note(40, 136 + i * 44, d, 10)
    s.box(320, 140, 160, 64, "local cache", "yellow", 13, sub="~/.cache/huggingface")
    s.arrow(272, 172, 318, 172, "download\nonce", lx=295, ly=148)
    s.box(540, 62, 340, 56, "tokenizer = AutoTokenizer.from_pretrained(name)", "blue", 11, mono=True,
          sub="→ T5TokenizerFast: text ⇄ ids")
    s.arrow(482, 160, 538, 96)
    s.box(520, 140, 370, 142, "", "green")
    s.text(705, 156, "model = AutoModelForSeq2SeqLM.from_pretrained(name)", 11, "middle", mono=True)
    s.text(555, 182, "model.config  ← from config.json", 12, weight="bold")
    for i, l in enumerate(["  .is_encoder_decoder   .vocab_size", "  .num_layers  .num_decoder_layers  .d_model"]):
        s.text(555, 202 + i * 18, l, 11, mono=True, color="#334155")
    s.text(555, 248, "model.parameters()  ← weight tensors", 12, weight="bold")
    s.text(555, 266, "  from model.safetensors", 11, mono=True, color="#334155")
    s.arrow(482, 180, 538, 210)
    s.save(OUT / "l1_hub_loading.svg")


def params():
    s = SVG(900, 230, "Counting parameters")
    s.title_bar("Counting parameters: every weight is a tensor, numel() counts its numbers")
    tensors = [("shared.weight", "32128 × 512", "16,449,536"), ("encoder…SelfAttention.q.weight", "384 × 512", "196,608"),
               ("…relative_attention_bias", "32 × 6", "192")]
    x = 20
    for name, shape, n in tensors:
        s.box(x, 70, 210, 86, "", "blue")
        s.text(x + 105, 88, name, 11, "middle", mono=True)
        s.text(x + 105, 112, f"shape ({shape})", 12, "middle")
        s.text(x + 105, 136, f"p.numel() = {n}", 11, "middle", mono=True, color="#1d4ed8")
        x += 222
    s.text(x + 6, 113, "+ … (190 tensors)", 12, color="#334155")
    s.arrow(450, 160, 450, 186)
    s.box(250, 188, 400, 32, "n_params = Σ p.numel() for p in model.parameters()", "green", 12, mono=True)
    s.save(OUT / "l1_params.svg")


def tokenize():
    s = SVG(980, 250, "Tokenization round trip")
    s.title_bar("encode() and decode(): text ⇄ token ids")
    s.box(20, 76, 150, 40, '"Hello world"', "white", 13, mono=True)
    s.arrow(172, 96, 226, 96, "tokenize", ly=84)
    s.chips(230, 83, ["▁Hello", "▁world", "</s>"], kinds=["blue", "blue", "orange"])
    s.arrow(470, 96, 524, 96, "→ ids", ly=84)
    s.chips(528, 83, ["8774", "296", "1"], kinds=["blue", "blue", "orange"], w=56)
    s.note(528, 128, 'tokenizer(text)["input_ids"] → plain list [8774, 296, 1]', 11)
    s.note(230, 128, "▁ = a space came before this piece", 11)
    s.chips(230, 168, ["0", "8774", "296", "1"], kinds=["orange", "blue", "blue", "orange"], w=56)
    s.text(486, 181, "tokenizer.decode(ids, skip_special_tokens=True)", 11, mono=True)
    s.arrow(808, 181, 836, 181)
    s.text(842, 181, '"Hello world"', 12, mono=True)
    s.note(230, 214, "special tokens (orange): 0 = <pad>, 1 = </s> (end of sequence)  → dropped by skip_special_tokens", 11)
    s.note(486, 160, "works on a list or a 1-D tensor", 10)
    s.save(OUT / "l1_tokenize.svg")


def greedy_loop():
    s = SVG(920, 400, "Greedy decoding loop")
    s.title_bar("Greedy decoding, one token per iteration",
                'prompt: "Answer the question: What color is the sky?"')
    s.box(20, 70, 200, 44, "input_ids  (1, n)", "gray", 12, mono=True)
    s.arrow(222, 92, 262, 92)
    s.box(264, 70, 150, 44, "Encoder", "green", 13, bold=True)
    s.note(339, 128, "(runs inside every model(...) call)", 10, "middle")
    rows = [("step 1", ["0"], "(1, 1, 32128)", "1692  '▁blue'", False),
            ("step 2", ["0", "1692"], "(1, 2, 32128)", "1  '</s>'  → stop", True)]
    s.text(20, 160, "decoder_input_ids", 12, weight="bold")
    s.text(330, 160, "model(input_ids=…, decoder_input_ids=…).logits", 11, mono=True)
    s.text(690, 160, "argmax of LAST position", 12, weight="bold")
    for i, (st, ids, shape, pick, stop) in enumerate(rows):
        y = 180 + i * 80
        s.note(20, y + 13, st, 11)
        s.chips(70, y, ids, kinds=["orange"] + ["blue"] * (len(ids) - 1), w=56)
        s.arrow(72 + 61 * len(ids), y + 13, 326, y + 13)
        s.box(330, y - 4, 200, 34, f"logits {shape}", "purple", 11, mono=True)
        s.arrow(532, y + 13, 686, y + 13, "logits[0, -1].argmax()", ly=y + 2, size=10)
        s.chip(690, y, pick, "red" if stop else "blue", w=200)
    s.arrow(790, 208, 190, 258, path="M790,208 C790,236 300,226 190,256", dashed=True)
    s.note(250, 230, "append: torch.cat([dec, next_id], dim=1)", 10, "middle")
    s.box(20, 318, 880, 70, "", "yellow")
    s.text(34, 334, "Loop recipe", 12, weight="bold")
    s.text(34, 354, "dec = [[decoder_start_token_id]] (=0)  →  repeat ≤ max_new_tokens:  forward  →  argmax(last)  →  append  →  break if it was eos (=1)",
           11, mono=True)
    s.text(34, 374, "finally: tokenizer.decode(dec[0], skip_special_tokens=True)  →  \"blue\"", 11, mono=True)
    s.save(OUT / "l1_greedy_loop.svg")


def reply_pipeline():
    s = SVG(990, 250, "reply() pipeline")
    s.title_bar("reply(): tokenize → generate → decode")
    steps = [("prompt (str)", "gray", ""), ('tokenizer(prompt,\n return_tensors="pt")', "blue", "{input_ids, attention_mask}"),
             ("model.generate(**inputs,\n max_new_tokens=…, **gen_kwargs)", "green", "tensor([[0, 1692, 1]])"),
             ("tokenizer.decode(out[0],\n skip_special_tokens=True)", "blue", '"blue"'), (".strip()", "gray", "")]
    x, ws = 20, [110, 180, 250, 210, 90]
    for (lab, k, out), w in zip(steps, ws):
        s.box(x, 70, w, 62, lab, k, 11, mono=True)
        if out:
            s.note(x + w / 2, 150, "→ " + out, 11, "middle")
        if lab != ".strip()":
            s.arrow(x + w + 2, 101, x + w + 18, 101)
        x += w + 20
    s.box(20, 178, 950, 58, "", "yellow")
    s.text(34, 196, "** unpacks a dict into keyword arguments:", 12, weight="bold")
    s.text(34, 218, "inputs = {\"input_ids\": …, \"attention_mask\": …}   generate(**inputs, num_beams=4)  ≡  generate(input_ids=…, attention_mask=…, num_beams=4)",
           11, mono=True)
    s.save(OUT / "l1_reply_pipeline.svg")


def decoding():
    s = SVG(940, 330, "Decoding strategies")
    s.title_bar("Choosing the next token: greedy vs beam search vs sampling",
                'model\'s probabilities for the word after "Tell me about …"')
    probs = [("dogs", .42), ("pets", .25), ("animals", .18), ("cats", .10), ("friends", .05)]
    # panel 1 greedy
    s.text(20, 82, "Greedy (default)", 13, weight="bold")
    for i, (w, p) in enumerate(probs):
        y = 96 + i * 30
        s.text(20, y + 10, w, 11, mono=True)
        s.rect(90, y, 180 * p / .42, 20, "blue" if i == 0 else "gray", rx=3, sw=1)
        s.text(96 + 180 * p / .42, y + 10, f"{p:.2f}", 10, color="#334155")
    s.note(20, 260, "always the top one → deterministic,\nsame output every run", 11)
    # panel 2 beam
    x0 = 330
    s.text(x0, 82, "Beam search (num_beams=k)", 13, weight="bold")
    s.chip(x0, 140, "Tell me…", "gray", mono=False)
    for i, (w, sc, keep) in enumerate([("dogs", .42, True), ("pets", .25, True), ("animals", .18, False)]):
        y = 100 + i * 46
        s.line(x0 + 70, 153, x0 + 100, y + 13)
        s.chip(x0 + 100, y, f"{w} {sc}", "green" if keep else "pad", mono=False)
        if keep:
            s.line(x0 + 190, y + 13, x0 + 210, y + 13)
            s.chip(x0 + 210, y, "… keep growing", "green", mono=False)
    s.note(x0, 260, "keeps the k best partial sentences,\nreturns the best complete one → deterministic", 11)
    # panel 3 sampling
    x0 = 650
    s.text(x0, 82, "Sampling (do_sample=True)", 13, weight="bold")
    for i, (w, p) in enumerate(probs):
        y = 96 + i * 30
        s.text(x0, y + 10, w, 11, mono=True)
        s.rect(x0 + 84, y, 140 * p / .42, 20, "orange", rx=3, sw=1)
    s.text(x0 + 236, 140, "🎲", 22)
    s.note(x0, 260, "draws at random ∝ probability\ntemperature<1 sharpens, top_p trims the tail", 11)
    s.box(20, 290, 900, 30, "Reproducible sampling: call torch.manual_seed(seed) IMMEDIATELY before the sampling generate() call",
          "yellow", 12)
    s.save(OUT / "l1_decoding.svg")


def prompt_stuffing():
    s = SVG(920, 330, "Prompt stuffing")
    s.title_bar("format_prompt(): the model has no memory, so we resend the recent history",
                "history = list of (user, bot) tuples, oldest first   —   example with max_turns = 2")
    turns = [("Hi", "Hello!"), ("I'm Sam", "Nice to meet you, Sam."), ("I like jazz", "Jazz is great.")]
    for i, (u, b) in enumerate(turns):
        x = 20 + i * 200
        kept = i >= 1
        s.box(x, 80, 186, 58, f'("{u}",\n "{b}")', "blue" if kept else "pad", 10, mono=True)
        s.note(x + 93, 152, f"history[{i}]  /  history[{i - 3}]", 10, "middle")
    s.line(222, 168, 606, 168, "#2563eb", sw=2)
    s.text(414, 182, "history[-2:]  → the last 2 exchanges", 11, "middle", "#1d4ed8", mono=True)
    s.box(640, 70, 260, 150, "", "green")
    lines = ["User: I'm Sam", "Bot: Nice to meet you, Sam.", "User: I like jazz", "Bot: Jazz is great.",
             "User: What is my name?", "Bot:"]
    for i, l in enumerate(lines):
        s.text(654, 88 + i * 21, l, 11, mono=True, color="#14532d" if i < 4 else "#9a3412")
    s.arrow(608, 110, 638, 110)
    s.note(770, 236, '"\\n".join(lines)  — ends with "Bot:" (no trailing space)', 10, "middle")
    s.box(20, 262, 880, 54, "", "red")
    s.text(34, 280, "⚠ Python gotcha: history[-0:] is history[0:], the WHOLE list, not an empty one.", 12, weight="bold")
    s.text(34, 300, "So handle max_turns = 0 separately:  kept = history[-max_turns:] if max_turns > 0 else []", 11, mono=True)
    s.save(OUT / "l1_prompt_stuffing.svg")


def context_budget():
    s = SVG(920, 290, "Fitting a token budget")
    s.title_bar("fit_to_context(): drop the OLDEST exchange until the prompt fits",
                "len(encode(prompt)) counts tokens; max_tokens is the budget")
    segs = [("turn 1", 14, "blue"), ("turn 2", 18, "blue"), ("turn 3", 15, "blue"), ("new msg + Bot:", 12, "orange")]
    budget = 48
    scale = 10
    for r, drop in enumerate([0, 1]):
        y = 92 + r * 70
        x = 120
        s.note(20, y + 14, f"attempt {r + 1}", 11)
        total = 0
        for i, (lab, n, k) in enumerate(segs):
            if i < drop:
                continue
            s.box(x, y, n * scale - 3, 30, f"{lab} ({n})", k, 11)
            x += n * scale
            total += n
        ok = total <= budget
        s.chip(730, y + 2, f"{total} tok {'✓ fits' if ok else '✗ too long'}", "green" if ok else "red", mono=False)
    bx = 120 + budget * scale
    s.line(bx, 76, bx, 232, "#dc2626", dashed=True, sw=2)
    s.text(bx, 248, f"max_tokens = {budget}", 11, "middle", "#dc2626", mono=True)
    s.note(20, 272, "kept = kept[1:] drops the oldest. If the history is empty and it still doesn't fit, return that bare prompt anyway.", 11)
    s.save(OUT / "l1_context_budget.svg")


def chatbot():
    s = SVG(960, 300, "ChatBot.ask flow")
    s.title_bar("ChatBot: state (history) + the functions you already wrote")
    s.box(20, 64, 200, 150, "", "purple")
    s.text(120, 82, "ChatBot", 14, "middle", weight="bold")
    for i, l in enumerate(["max_turns", "max_tokens", "gen_kwargs  (dict)", "history  (list)", "", "ask(message) → str", "reset()"]):
        s.text(34, 106 + i * 16, l, 11, mono=True)
    steps = [("message", "gray", 80), ("recent = history[-max_turns:]\n(or [] if max_turns == 0)", "blue", 196),
             ("prompt = fit_to_context(\n recent, message, max_tokens)", "blue", 206),
             ("answer = reply(prompt,\n **gen_kwargs)", "green", 166)]
    x = 240
    for i, (lab, k, w) in enumerate(steps):
        s.box(x, 80, w, 56, lab, k, 10, mono=True)
        if i < len(steps) - 1:
            s.arrow(x + w + 2, 108, x + w + 14, 108)
        x += w + 16
    rx = x - 16 - 83          # centre of the reply box
    s.box(560, 180, 250, 40, "history.append((message, answer))", "orange", 11, mono=True)
    s.arrow(rx, 138, 812, 200, path=f"M{rx},138 C{rx},180 840,200 812,200")
    s.box(850, 180, 90, 40, "return\nanswer", "gray", 11, mono=True)
    s.arrow(810, 230, 850, 230, path="M685,222 C700,250 840,250 852,222")
    s.arrow(560, 200, 222, 170, path="M558,200 C420,244 300,230 222,176", dashed=True)
    s.note(390, 250, "the next ask() sees this turn", 10, "middle")
    s.note(20, 280, "Store **gen_kwargs from __init__ in an attribute so ask() can forward them, e.g. ChatBot(num_beams=4).", 11)
    s.save(OUT / "l1_chatbot.svg")


if __name__ == "__main__":
    for f in [families, hub_loading, params, tokenize, greedy_loop, reply_pipeline, decoding,
              prompt_stuffing, context_budget, chatbot]:
        f()
    print("lab 1 figures →", OUT)
