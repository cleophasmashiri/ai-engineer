"""SPOILERS — builds lab_01_exploring_genai_libraries.ipynb (contains solutions)."""
from pathlib import Path
from nbbuild import Notebook

COURSE = Path(__file__).resolve().parent.parent / "generative-ai-llm-architecture-data-preparation"
nb = Notebook("lab_01_exploring_genai_libraries", "Lab 1 — Exploring Generative AI Libraries")

nb.md(r'''
# Lab 1 — Exploring Generative AI Libraries: build a chatbot with 🤗 Transformers

*Adapted from the AI Engineer course "Generative AI and LLMs: Architecture and Data Preparation".*

The original lab gives you a working chatbot in about 15 lines and asks you to run it. Here you'll **build the
same chatbot yourself, one piece at a time**, so you understand what each piece does:

| # | You will learn | You will build |
|---|---|---|
| 1 | What the Hugging Face ecosystem is and the three Transformer families | `describe_model()` |
| 2 | How text becomes numbers (tokenization) | `encode()` / `decode()` |
| 3 | How an encoder-decoder model generates text **one token at a time** | `greedy_generate()` (by hand!) |
| 4 | The high-level `generate()` API | `reply()` |
| 5 | Decoding strategies: greedy, beam search, sampling | `compare_decoding()` |
| 6 | Why chatbots "remember": prompt-stuffing the history | `format_prompt()` |
| 7 | Context windows and token budgets | `fit_to_context()` |
| 8 | Putting it together | a `ChatBot` class |

## How grading works

Each exercise has a **stub cell** for your code and a **tests cell** under it. Run the tests cell:
- ✅ all tests pass → you **earn** the exercise's points (once).
- 🔒 the **Show solution** button reveals the reference answer. If you haven't passed yet, it **costs the
  same number of points** (and passing afterwards earns nothing). After you've passed, revealing is free,
  so compare your answer with the reference.
- Your score is saved in `.progress/` and survives kernel restarts. `lab.scoreboard()` shows it at any time.

> Run the cells **in order**. Later exercises reuse your earlier functions.
''')
nb.setup()

nb.md(r'''
---
## 1. The Hugging Face ecosystem

[Hugging Face](https://huggingface.co/models) is to ML models what GitHub is to code. Three pieces matter here:

- **The Hub**: hosts more than a million pretrained models. A model is named `owner/name`, e.g. `google/flan-t5-small`.
- **`transformers`**: the Python library that downloads a model and gives you a PyTorch (or TF/JAX) object.
- **`tokenizers`**: fast text ⇄ token-id conversion. Every model ships with its **own** tokenizer.

The `Auto*` classes read the model's `config.json` and choose the right architecture for you:

| Class | Use it for | Example models |
|---|---|---|
| `AutoModel` | raw hidden states (embeddings) | BERT |
| `AutoModelForCausalLM` | **decoder-only** text generation | GPT-2, Llama, Qwen |
| `AutoModelForSeq2SeqLM` | **encoder-decoder** text-to-text | T5, FLAN-T5, BART, BlenderBot |

### The three Transformer families
''')
nb.figure("l1_transformer_families", "FLAN-T5 is an encoder-decoder: the encoder reads the whole prompt once; the decoder writes the answer token by token, looking back at the encoder output through cross-attention.")
nb.md(r'''
**FLAN-T5** is Google's T5 after *instruction tuning* ("FLAN" = Finetuned LAnguage Net). It was trained on more than
1,000 tasks phrased as instructions ("Translate…", "Answer the question…"), so it follows instructions
without any further training. The `small` variant is tiny (~77M parameters), so it runs on a laptop CPU.
Expect it to be confidently wrong sometimes, and we'll see that happen.

Load it. The first run downloads about 300 MB to `~/.cache/huggingface`:
''')
nb.code(r'''
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()  # inference mode: disables dropout

print(type(tokenizer).__name__, "|", type(model).__name__)
print(model.config)  # every architectural hyper-parameter lives here
''')
nb.figure("l1_hub_loading", "from_pretrained downloads the files once, then builds a tokenizer object and a model object whose .config and .parameters() you can inspect.")
nb.md(r'''
Things to look for in that config: `d_model` (hidden size), `num_layers` / `num_decoder_layers`, `num_heads`,
`vocab_size`, and `is_encoder_decoder`. Every parameter of a PyTorch model is a tensor you can iterate over with
`model.parameters()`, and `tensor.numel()` gives the number of elements in a tensor.
''')
nb.exercise("ex1", "Describe a model", 10,
    prompt=r'''
Write `describe_model(model)` returning a dict with:

| key | value |
|---|---|
| `"n_params"` | total number of parameters (int) |
| `"params_millions"` | the same, in millions, rounded to 1 decimal |
| `"encoder_decoder"` | `True`/`False`, read from the config |
| `"vocab_size"` | from the config |
| `"layers"` | tuple `(encoder_layers, decoder_layers)` from the config |
''',
    guide=r'''
- **`model.parameters()`** yields every weight tensor of the network (190 of them for this model).
- **`p.numel()`** is the number of numbers inside tensor `p`, the product of its shape: a `(2, 3)` tensor has 6.
- **`sum(<generator>)`** adds things up without building a list: `sum(x * x for x in [1, 2, 3])` → `14`.
- **`model.config`** holds the architecture as plain Python attributes. The ones you need are
  `is_encoder_decoder`, `vocab_size`, `num_layers` (encoder blocks) and `num_decoder_layers`.
- **`round(value, 1)`** rounds to one decimal: `round(12_345_678 / 1e6, 1)` → `12.3`.

![Counting parameters](figures/l1_params.svg)

**🪜 Steps:** ① add up `numel()` over all parameters → ② compute millions → ③ read the four config attributes →
④ return a dict with **exactly** the five keys, with `layers` as a **tuple** `(encoder, decoder)`.
''',
    guide_code=r'''
toy = torch.nn.Linear(3, 2)                    # a tiny layer: weight (2, 3) + bias (2,)
for name, p in toy.named_parameters():
    print(f"{name:<7} shape={tuple(p.shape)}  numel={p.numel()}")
print("toy total:", sum(p.numel() for p in toy.parameters()))
print("rounded  :", round(12_345_678 / 1e6, 1))
print("a config attribute:", model.config.d_model, type(model.config.d_model).__name__)
''',
    stub=r'''
def describe_model(model):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_returns_dict_with_keys():
    d = describe_model(model)
    assert set(d) == {"n_params", "params_millions", "encoder_decoder", "vocab_size", "layers"}, f"keys: {set(d)}"

def test_parameter_count():
    d = describe_model(model)
    assert d["n_params"] == 76_961_152, f"got {d['n_params']:,}"
    assert d["params_millions"] == 77.0, f"got {d['params_millions']}"

def test_config_facts():
    d = describe_model(model)
    assert d["encoder_decoder"] is True
    assert d["vocab_size"] == 32128
    assert tuple(d["layers"]) == (8, 8), f"got {d['layers']}"
''',
    solution=r'''
def describe_model(model):
    n = sum(p.numel() for p in model.parameters())
    cfg = model.config
    return {
        "n_params": n,
        "params_millions": round(n / 1e6, 1),
        "encoder_decoder": cfg.is_encoder_decoder,
        "vocab_size": cfg.vocab_size,
        "layers": (cfg.num_layers, cfg.num_decoder_layers),
    }
''')

nb.md(r'''
---
## 2. Tokenization: text → numbers → text

Neural networks only do arithmetic, so text must become integers. T5 uses **SentencePiece**, a *subword*
tokenizer. Common words get one token, and rare words are split into pieces. `▁` marks "a space came before
this piece". (Lab 2 covers tokenization in depth.)
''')
nb.code(r'''
text = "Tokenization is unbelievably enlightening!"
enc = tokenizer(text)                     # a dict-like BatchEncoding
print("input_ids     :", enc["input_ids"])
print("tokens        :", tokenizer.convert_ids_to_tokens(enc["input_ids"]))
print("attention_mask:", enc["attention_mask"])
print("special tokens:", {"eos": (tokenizer.eos_token, tokenizer.eos_token_id),
                          "pad": (tokenizer.pad_token, tokenizer.pad_token_id),
                          "unk": (tokenizer.unk_token, tokenizer.unk_token_id)})
''')
nb.figure("l1_tokenize", "Text → subword tokens → integer ids, and back. Special tokens (orange) mark structure, not words.")
nb.md(r'''
Notice:
- The tokenizer **appends `</s>` (id 1)**, the *end-of-sequence* token, so the model knows where the input stops.
- `attention_mask` is all 1s here. It becomes useful when batching sequences of different lengths, where padding
  positions get 0 (Lab 3).
- `return_tensors="pt"` gives PyTorch tensors of shape `(batch, seq_len)` instead of lists.
- `tokenizer.decode(ids, skip_special_tokens=True)` goes the other way and drops `</s>`/`<pad>`.
''')
nb.code(r'''
batch = tokenizer(["Hi!", "A slightly longer sentence."], padding=True, return_tensors="pt")
print(batch["input_ids"])        # shorter row padded with 0 = <pad>
print(batch["attention_mask"])   # 0 where padded
print(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))
''')
nb.exercise("ex2", "Encode and decode", 10,
    prompt=r'''
Using the global `tokenizer`:
- `encode(text)` → a **plain Python list** of token ids (including the trailing EOS id).
- `decode(ids)` → the text, **without** special tokens. It must accept a list *or* a 1-D tensor.
''',
    guide=r'''
- **`tokenizer(text)`** returns a dict-like `BatchEncoding`. Without `return_tensors`, `["input_ids"]` is a plain
  **list** of ints that already ends with the EOS id `1` (see the figure above).
- **`tokenizer.decode(ids, skip_special_tokens=True)`** turns ids back into text and drops `<pad>` (0) and `</s>` (1).
  It accepts a Python list **or** a 1-D tensor, so you don't need any conversion.
- Each function is a single line.

**🪜 Steps:** `encode` → call the tokenizer and pick out `"input_ids"`. `decode` → call `tokenizer.decode` with the flag.
''',
    guide_code=r'''
enc = tokenizer("Good morning")
print(type(enc).__name__, dict(enc))
print(tokenizer.decode([0, 1563, 1]), "|", tokenizer.decode([0, 1563, 1], skip_special_tokens=True))
''',
    stub=r'''
def encode(text):
    # YOUR CODE HERE
    raise NotImplementedError

def decode(ids):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_encode_is_list_ending_with_eos():
    ids = encode("Hello world")
    assert isinstance(ids, list), f"expected list, got {type(ids).__name__}"
    assert ids == [8774, 296, 1], f"got {ids}"

def test_decode_strips_special_tokens():
    assert decode([8774, 296, 1]) == "Hello world"
    assert decode(torch.tensor([0, 8774, 296, 1])) == "Hello world"

def test_round_trip():
    s = "Generative AI is fun."
    assert decode(encode(s)) == s
''',
    solution=r'''
def encode(text):
    return tokenizer(text)["input_ids"]

def decode(ids):
    return tokenizer.decode(ids, skip_special_tokens=True)
''')

nb.md(r'''
---
## 3. How generation really works: one token at a time

`model.generate()` hides a simple loop. For an encoder-decoder model:
''')
nb.figure("l1_greedy_loop", "Autoregressive decoding. Each new token depends on the prompt (via the encoder) and on every token generated so far.")
nb.md(r'''
One forward pass gives you **logits** of shape `(batch, decoder_len, vocab_size)`. Only the **last position**
`logits[0, -1]` matters, because it scores every possible *next* token. Let's take a single step by hand:
''')
nb.code(r'''
prompt_ids = tokenizer("Answer the question: What color is the sky?", return_tensors="pt").input_ids
start = torch.tensor([[model.config.decoder_start_token_id]])        # [[0]]
with torch.no_grad():                                                 # no gradients needed for inference
    out = model(input_ids=prompt_ids, decoder_input_ids=start)
print("logits shape:", tuple(out.logits.shape))
top = torch.topk(out.logits[0, -1].softmax(-1), 5)
for p, i in zip(top.values, top.indices):
    print(f"  {tokenizer.convert_ids_to_tokens(int(i)):>10}  p={p:.3f}")
''')
nb.exercise("ex3", "Greedy decoding by hand", 20,
    prompt=r'''
Implement `greedy_generate(prompt, max_new_tokens=20)` **without calling `model.generate`**:

1. Tokenize the prompt to a `(1, n)` tensor.
2. Start `decoder_input_ids` as `[[model.config.decoder_start_token_id]]`.
3. Loop up to `max_new_tokens` times: forward pass, `argmax` of the last position's logits, append it.
   **Stop** once the appended token is `tokenizer.eos_token_id`.
4. Return the decoded string (skip special tokens).

💡 `torch.cat([a, b], dim=1)` appends along the sequence axis; wrap the loop in `torch.no_grad()`.
(Real implementations also cache the encoder output and past keys/values. Ignore that for now.)
''',
    guide=r'''
Look at the **greedy decoding figure** above: you're implementing exactly that loop. The pieces:

| piece | what it does |
|---|---|
| `tokenizer(prompt, return_tensors="pt").input_ids` | prompt ids as a `(1, n)` tensor |
| `torch.tensor([[model.config.decoder_start_token_id]])` | the decoder's starting sequence `[[0]]`, shape `(1, 1)` |
| `with torch.no_grad():` | turn off gradient tracking (faster, less memory) for inference |
| `model(input_ids=..., decoder_input_ids=...).logits` | scores of shape `(1, decoder_len, 32128)` |
| `logits[0, -1]` | the scores for the **next** token: batch item 0, **last** position |
| `.argmax()` | index of the highest score, as a 0-d tensor |
| `.view(1, 1)` | reshape that 0-d tensor to `(1, 1)` so it can be concatenated |
| `torch.cat([dec, next_id], dim=1)` | append along the sequence axis: `(1, t)` → `(1, t+1)` |
| `.item()` | 0-d tensor → Python int, for comparing with `tokenizer.eos_token_id` |

**🪜 Steps:**
1. Tokenize the prompt and create `dec = [[0]]`.
2. `for _ in range(max_new_tokens):` forward pass → pick the argmax of the last position → append it to `dec` →
   `break` if it equals `tokenizer.eos_token_id`.
3. After the loop, `tokenizer.decode(dec[0], skip_special_tokens=True)`.
''',
    guide_code=r'''
a = torch.tensor([[0, 5]])                     # shape (1, 2)
b = torch.tensor(7).view(1, 1)                 # shape (1, 1)
print(torch.cat([a, b], dim=1))                # tensor([[0, 5, 7]])

scores = torch.tensor([[[0.1, 0.9, 0.0],       # shape (1, 2, 3): batch=1, positions=2, vocab=3
                        [0.2, 0.1, 0.7]]])
print(scores[0, -1], "→ argmax", scores[0, -1].argmax(), "→ item", scores[0, -1].argmax().item())
''',
    stub=r'''
def greedy_generate(prompt, max_new_tokens=20):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def _reference(prompt, n):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n, do_sample=False, num_beams=1)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def test_matches_generate_short_answer():
    p = "Answer the question: What color is the sky?"
    assert greedy_generate(p) == _reference(p, 20), f"got {greedy_generate(p)!r}"

def test_matches_generate_translation():
    p = "Translate English to German: The house is wonderful."
    assert greedy_generate(p, 30) == _reference(p, 30), f"got {greedy_generate(p, 30)!r}"

def test_respects_max_new_tokens():
    p = "Write a long story about a dragon."
    got = greedy_generate(p, max_new_tokens=3)
    assert len(tokenizer(got, add_special_tokens=False).input_ids) <= 3, f"too long: {got!r}"
''',
    solution=r'''
def greedy_generate(prompt, max_new_tokens=20):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    dec = torch.tensor([[model.config.decoder_start_token_id]])
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids=input_ids, decoder_input_ids=dec).logits
            next_id = logits[0, -1].argmax().view(1, 1)
            dec = torch.cat([dec, next_id], dim=1)
            if next_id.item() == tokenizer.eos_token_id:
                break
    return tokenizer.decode(dec[0], skip_special_tokens=True)
''')

nb.md(r'''
---
## 4. The high-level API: `model.generate()`

Now that you know what's inside, use the real thing. Useful arguments:

| argument | meaning |
|---|---|
| `max_new_tokens` | hard cap on output length |
| `num_beams` | beam search width (1 = greedy) |
| `do_sample`, `temperature`, `top_k`, `top_p` | random sampling controls |
| `no_repeat_ngram_size`, `repetition_penalty` | fight loops like "slender slender slender" |

There is also `pipeline("text2text-generation", model=...)`, which wraps tokenize → generate → decode in one call.
''')
nb.figure("l1_reply_pipeline", "The three calls behind every Hugging Face text-generation helper.")
nb.code(r'''
from transformers import pipeline
try:
    t2t = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device="cpu")
    print(t2t("Translate English to French: I love learning.", max_new_tokens=30))
except Exception as e:   # task names differ between transformers versions
    print("pipeline unavailable in this version:", type(e).__name__, e)
''')
nb.exercise("ex4", "A reply function", 10,
    prompt=r'''
Write `reply(prompt, max_new_tokens=40, **gen_kwargs)`: tokenize → `model.generate(...)` (forwarding any extra
keyword arguments, such as `num_beams=4`) → decode **with special tokens skipped** → return the `.strip()`ped string.
''',
    guide=r'''
- **`return_tensors="pt"`** makes the tokenizer return PyTorch tensors of shape `(1, n)`, which is what `generate` expects.
- **`**inputs`** unpacks the tokenizer's dict into keyword arguments `input_ids=…, attention_mask=…`.
- **`**gen_kwargs` in a signature** collects any *extra* keyword arguments into a dict. Writing `**gen_kwargs`
  again in a *call* passes them on unchanged. That's how `reply(p, num_beams=4)` reaches `generate(num_beams=4)`.
- **`model.generate(...)`** returns a tensor of shape `(1, length)`. `out[0]` is the single generated sequence.
- **`str.strip()`** removes leading and trailing whitespace.

**🪜 Steps:** tokenize (tensors) → `generate` with `max_new_tokens` and the forwarded `**gen_kwargs` → decode
`out[0]` skipping special tokens → `.strip()`.
''',
    guide_code=r'''
def outer(a, **extra):                          # extra collects any other keyword args into a dict
    print("outer got a =", a, "and extra =", extra)
    inner(**extra)                              # ...and ** passes them on
def inner(x=0, y=0):
    print("inner got x =", x, "y =", y)
outer(1, x=10, y=20)

inputs = tokenizer("Hi there", return_tensors="pt")
print(list(inputs.keys()), inputs["input_ids"].shape)
''',
    stub=r'''
def reply(prompt, max_new_tokens=40, **gen_kwargs):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_answers_simple_question():
    ans = reply("Answer the question: What color is the sky?")
    assert isinstance(ans, str) and "blue" in ans.lower(), f"got {ans!r}"

def test_no_special_tokens_in_output():
    ans = reply("Translate English to German: How old are you?")
    assert "<pad>" not in ans and "</s>" not in ans, f"got {ans!r}"

def test_forwards_generation_kwargs():
    p = "Tell me about dogs."
    beam = reply(p, num_beams=4)
    ref = tokenizer.decode(model.generate(**tokenizer(p, return_tensors="pt"), max_new_tokens=40,
                                          num_beams=4)[0], skip_special_tokens=True).strip()
    assert beam == ref, f"num_beams not forwarded? got {beam!r}, expected {ref!r}"
''',
    solution=r'''
def reply(prompt, max_new_tokens=40, **gen_kwargs):
    inputs = tokenizer(prompt, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()
''')

nb.md(r'''
### A reality check on small models
''')
nb.code(r'''
try:
    for q in ["What is the capital of France?", "What is 2 plus 3?", "Answer the question: What color is the sky?"]:
        print(f"{q:<45} → {reply(q)!r}")
except NotImplementedError:
    print("Finish Exercise 4 (reply) first, then re-run this cell.")
''')
nb.md(r'''
A 77M-parameter model knows "the sky is blue" but may make up a capital or fall into repetition. It **always
answers fluently, even when it's wrong.** That is *hallucination*, and model size, training data and
instruction tuning all affect how often it happens. Prompt phrasing matters too: FLAN-T5 does better with the
instruction formats it saw in training ("Answer the question: …", "Translate English to German: …").
''')

nb.md(r'''
---
## 5. Decoding strategies

| strategy | how it picks the next token | character |
|---|---|---|
| **Greedy** | always the single most likely | deterministic, can loop |
| **Beam search** (`num_beams=k`) | keeps the k best *partial sequences*, returns the best full one | more fluent, still deterministic, bland |
| **Sampling** (`do_sample=True`) | draws randomly from the distribution | diverse/creative, needs a **seed** to reproduce |
| `temperature` | divides logits before softmax: <1 sharper, >1 flatter | |
| `top_p` (nucleus) | sample only from the smallest set whose prob ≥ p | trims the long tail |
''')
nb.figure("l1_decoding", "Same probabilities, three ways to choose.")
nb.code(r'''
p = "Tell me about dogs."
try:
    for seed in [0, 0, 1]:
        torch.manual_seed(seed)
        print(f"seed={seed}:", reply(p, do_sample=True, temperature=0.9, top_p=0.9))
except NotImplementedError:
    print("Finish Exercise 4 (reply) first, then re-run this cell.")
''')
nb.md("Same seed → same output; different seed → different output. This is how you make sampling experiments reproducible.")
nb.exercise("ex5", "Compare decoding strategies", 15,
    prompt=r'''
Write `compare_decoding(prompt, seed=0)` that returns a dict with three replies (use your `reply`):
- `"greedy"`: default settings
- `"beam"`: `num_beams=4`
- `"sampled"`: `do_sample=True, temperature=0.9, top_p=0.9`, **after** `torch.manual_seed(seed)`

Calling it twice with the same seed must give identical results.
''',
    guide=r'''
- Reuse your **`reply`**: greedy is the default, beam search is `num_beams=4`, sampling is
  `do_sample=True, temperature=0.9, top_p=0.9`.
- **Random number generators have state.** `torch.manual_seed(s)` resets it, and every random draw advances it.
  To make the *sampled* reply reproducible, seed **immediately before** the sampling call, so nothing else
  consumes random numbers in between.
- Return a dict literal: `{"greedy": ..., "beam": ..., "sampled": ...}`.

**🪜 Steps:** greedy reply → beam reply → `torch.manual_seed(seed)` → sampled reply → return the dict.
''',
    guide_code=r'''
torch.manual_seed(42); print(torch.rand(2))
torch.manual_seed(42); print(torch.rand(2))   # identical: same seed, same state
print(torch.rand(2))                          # different: the generator has moved on
''',
    stub=r'''
def compare_decoding(prompt, seed=0):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
def test_has_three_strategies():
    d = compare_decoding("Tell me about cats.")
    assert set(d) == {"greedy", "beam", "sampled"}, f"keys: {set(d)}"
    assert all(isinstance(v, str) and v for v in d.values()), d

def test_greedy_and_beam_are_the_deterministic_ones():
    p = "Tell me about cats."
    d = compare_decoding(p)
    assert d["greedy"] == reply(p)
    assert d["beam"] == reply(p, num_beams=4)

def test_sampling_reproducible_with_seed():
    p = "Write a sentence about the ocean."
    assert compare_decoding(p, seed=7)["sampled"] == compare_decoding(p, seed=7)["sampled"]
    outs = {compare_decoding(p, seed=s)["sampled"] for s in range(5)}
    assert len(outs) > 1, "sampling never varied across seeds; is do_sample=True?"
''',
    solution=r'''
def compare_decoding(prompt, seed=0):
    greedy = reply(prompt)
    beam = reply(prompt, num_beams=4)
    torch.manual_seed(seed)                      # seed right before the random step
    sampled = reply(prompt, do_sample=True, temperature=0.9, top_p=0.9)
    return {"greedy": greedy, "beam": beam, "sampled": sampled}
''')

nb.md(r'''
---
## 6. How does a chatbot "remember"?

**It doesn't.** The model is a pure function: same input → (for greedy) same output. Nothing is stored between
calls. A chat app *simulates* memory by **pasting the conversation history into every new prompt**:
''')
nb.figure("l1_prompt_stuffing", "Prompt-stuffing: the recent transcript is pasted into every new prompt.")
nb.code(r'''
try:
    print(reply("What is my name?"))                                                   # no context → guesses
    print(reply("User: Hi, my name is Sam.\nBot: Hello Sam!\nUser: What is my name?\nBot:"))  # with context
except NotImplementedError:
    print("Finish Exercise 4 (reply) first, then re-run this cell.")
''')
nb.exercise("ex6", "Format the chat prompt", 15,
    prompt=r'''
Write `format_prompt(history, user_msg, max_turns=3)`:
- `history` is a list of `(user_text, bot_text)` tuples, oldest first.
- Keep only the **last `max_turns`** exchanges (`max_turns=0` → no history).
- Produce lines `User: …` / `Bot: …` for each kept exchange, then `User: {user_msg}` and finally `Bot:`
  (no trailing space), all joined with `"\n"`.
''',
    guide=r'''
- **Negative slicing** takes items from the end: `xs[-2:]` is the last two. **Gotcha:** `xs[-0:]` equals `xs[0:]`,
  the *whole* list, so `max_turns == 0` needs its own branch (see the red box in the figure above).
- **Tuple unpacking in a loop:** `for user, bot in history:` gives you both halves of each exchange.
- **f-strings** build each line: `f"User: {user}"`.
- **Collect lines in a list, then `"\n".join(lines)`.** That puts newlines *between* lines, never at the end.
- The prompt must end with `"Bot:"` exactly, with no trailing space.

**🪜 Steps:** choose `kept` → for each `(u, b)` add `User: u` and `Bot: b` → add `User: {user_msg}` → add `Bot:` → join.
''',
    guide_code=r'''
xs = ["a", "b", "c"]
print(xs[-2:], xs[-1:], xs[-0:], "← careful: -0 is 0!")
for u, b in [("hi", "hello"), ("bye", "ciao")]:
    print(f"U={u} | B={b}")
print(repr("\n".join(["line 1", "line 2"])))
''',
    stub=r'''
def format_prompt(history, user_msg, max_turns=3):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
H = [("Hi", "Hello!"), ("I'm Sam", "Nice to meet you, Sam."), ("I like jazz", "Jazz is great.")]

def test_empty_history():
    assert format_prompt([], "Hello") == "User: Hello\nBot:"

def test_full_history():
    exp = ("User: Hi\nBot: Hello!\nUser: I'm Sam\nBot: Nice to meet you, Sam.\n"
           "User: I like jazz\nBot: Jazz is great.\nUser: What is my name?\nBot:")
    assert format_prompt(H, "What is my name?") == exp, format_prompt(H, "What is my name?")

def test_keeps_only_last_turns():
    exp = "User: I like jazz\nBot: Jazz is great.\nUser: And you?\nBot:"
    assert format_prompt(H, "And you?", max_turns=1) == exp
    assert format_prompt(H, "And you?", max_turns=0) == "User: And you?\nBot:"
''',
    solution=r'''
def format_prompt(history, user_msg, max_turns=3):
    kept = history[-max_turns:] if max_turns > 0 else []
    lines = []
    for u, b in kept:
        lines += [f"User: {u}", f"Bot: {b}"]
    lines += [f"User: {user_msg}", "Bot:"]
    return "\n".join(lines)
''')

nb.md(r'''
---
## 7. Context windows: the memory has a size limit

Every model has a maximum input length, its **context window**, measured in **tokens**, not characters.
FLAN-T5 was trained on 512 tokens (`tokenizer.model_max_length`). Modern LLMs have 8k to 1M+ tokens, but the
problem is the same: **history grows every turn, so eventually you must drop or summarize old turns.**
The simplest policy is *drop the oldest exchange until the prompt fits*.
''')
nb.figure("l1_context_budget", "Trimming history to a token budget, oldest exchange first.")
nb.code(r'''
print("model_max_length:", tokenizer.model_max_length)
s = "The quick brown fox jumps over the lazy dog. " * 3
print(len(s), "characters →", len(tokenizer(s).input_ids), "tokens")
''')
nb.exercise("ex7", "Fit the history into a token budget", 15,
    prompt=r'''
Write `fit_to_context(history, user_msg, max_tokens)` that returns the prompt string from
`format_prompt(history_kept, user_msg, max_turns=len(history_kept))`, where `history_kept` is `history`
with the **oldest** exchanges dropped, one at a time, until `len(encode(prompt)) <= max_tokens`.
If even an empty history doesn't fit, return the prompt with no history anyway (the best you can do).
''',
    guide=r'''
- **Count tokens** with your `encode` from Exercise 2: `len(encode(prompt))`.
- **Drop the oldest exchange** with `kept = kept[1:]`. Index 0 is the oldest.
- **Copy first** (`kept = list(history)`) so you never modify the caller's list.
- **`while True:` + `return`** is a clean way to write "try, check, shrink, repeat".
- When `kept` is empty, `format_prompt(kept, msg, max_turns=0)` gives the bare prompt, because your Exercise 6
  handles `max_turns=0`. That's also your **stop condition**, since you can't drop anything more.

**🪜 Steps:** copy → loop { build prompt with `max_turns=len(kept)` → if it fits **or** `kept` is empty, return it →
else drop the first item }.
''',
    guide_code=r'''
kept = ["turn1", "turn2", "turn3"]
while True:
    text = " ".join(kept)
    print(f"{len(text):2d} chars: {kept}")
    if len(text) <= 11 or not kept:
        break
    kept = kept[1:]
''',
    stub=r'''
def fit_to_context(history, user_msg, max_tokens):
    # YOUR CODE HERE
    raise NotImplementedError
''',
    tests=r'''
H = [("Hi", "Hello!"), ("I'm Sam", "Nice to meet you, Sam."), ("I like jazz", "Jazz is great.")]

def test_everything_fits():
    p = fit_to_context(H, "What is my name?", max_tokens=512)
    assert p == format_prompt(H, "What is my name?", max_turns=3)

def test_drops_oldest_first():
    full = len(encode(format_prompt(H, "Hey", max_turns=3)))
    p = fit_to_context(H, "Hey", max_tokens=full - 1)
    assert "User: Hi\n" not in p, "the oldest turn should have been dropped"
    assert "I'm Sam" in p and "I like jazz" in p, "only the oldest turn should go"
    assert len(encode(p)) <= full - 1

def test_impossible_budget_returns_bare_prompt():
    assert fit_to_context(H, "Hey", max_tokens=1) == "User: Hey\nBot:"
''',
    solution=r'''
def fit_to_context(history, user_msg, max_tokens):
    kept = list(history)
    while True:
        prompt = format_prompt(kept, user_msg, max_turns=len(kept))
        if len(encode(prompt)) <= max_tokens or not kept:
            return prompt
        kept = kept[1:]
''')

nb.md(r'''
---
## 8. Put it all together: a `ChatBot` class

This is the original lab's chatbot, but now you know what each line does. We don't use `input()` because notebooks
should run top-to-bottom unattended. The bot is driven by method calls instead, and an interactive loop is
included at the end for you to try.
''')
nb.exercise("ex8", "The ChatBot class", 20,
    prompt=r'''
Implement `ChatBot(max_turns=3, max_tokens=512, **gen_kwargs)`:
- `history`: a list of `(user, bot)` tuples, starting empty.
- `ask(message)`: build the prompt with `fit_to_context(history[-max_turns:], message, max_tokens)` (if
  `max_turns` is 0, use no history), get a `reply(prompt, **gen_kwargs)`, **append** `(message, answer)` to
  `history`, and return the answer.
- `reset()`: clears the history.
''',
    guide=r'''
![ChatBot flow](figures/l1_chatbot.svg)

- **`__init__(self, max_turns=3, max_tokens=512, **gen_kwargs)`** stores everything on `self`, including the
  `gen_kwargs` dict, and starts `self.history = []`.
- **`ask`** chains functions you already wrote: pick the recent history (careful with `max_turns == 0`, as in
  Exercise 6) → `fit_to_context` → `reply(prompt, **self.gen_kwargs)` → append the `(message, answer)` tuple →
  return the answer.
- **`reset`** sets `self.history` back to an empty list.
''',
    guide_code=r'''
class Tally:                                   # same shape as ChatBot: config + state + methods
    def __init__(self, start=0, **options):
        self.total, self.options, self.log = start, options, []
    def add(self, n):
        self.total += n
        self.log.append(n)
        return self.total
    def reset(self):
        self.total, self.log = 0, []

t = Tally(verbose=True)
t.add(3); t.add(4)
print(t.total, t.log, t.options)
t.reset(); print(t.total, t.log)
''',
    stub=r'''
class ChatBot:
    # YOUR CODE HERE
    pass
''',
    tests=r'''
def test_ask_returns_string_and_records_history():
    bot = ChatBot()
    a = bot.ask("Hi, my name is Sam.")
    assert isinstance(a, str)
    assert bot.history == [("Hi, my name is Sam.", a)], bot.history

def test_history_is_used_in_prompt():
    bot = ChatBot()
    bot.history = [("Hi, my name is Sam.", "Hello Sam!")]
    a = bot.ask("What is my name?")
    assert a == reply(format_prompt([("Hi, my name is Sam.", "Hello Sam!")], "What is my name?")), a

def test_max_turns_limits_context():
    bot = ChatBot(max_turns=0)
    bot.history = [("Hi, my name is Sam.", "Hello Sam!")]
    assert bot.ask("What is my name?") == reply("User: What is my name?\nBot:")

def test_reset_and_gen_kwargs():
    bot = ChatBot(num_beams=4)
    bot.ask("Tell me about dogs.")
    assert bot.history[-1][1] == reply("User: Tell me about dogs.\nBot:", num_beams=4)
    bot.reset()
    assert bot.history == []
''',
    solution=r'''
class ChatBot:
    def __init__(self, max_turns=3, max_tokens=512, **gen_kwargs):
        self.max_turns, self.max_tokens, self.gen_kwargs = max_turns, max_tokens, gen_kwargs
        self.history = []

    def ask(self, message):
        recent = self.history[-self.max_turns:] if self.max_turns > 0 else []
        prompt = fit_to_context(recent, message, self.max_tokens)
        answer = reply(prompt, **self.gen_kwargs)
        self.history.append((message, answer))
        return answer

    def reset(self):
        self.history = []
''')

nb.md(r'''
### Try your bot

The cell below only does something once your `ChatBot` works. Change the messages and see what the model does
with context.
''')
nb.code(r'''
try:
    bot = ChatBot(num_beams=4)
    for msg in ["Hi, my name is Sam and I love jazz.", "What is my name?", "What music do I like?"]:
        print(f"🧑 {msg}\n🤖 {bot.ask(msg)}\n")
except Exception as e:
    print("Finish Exercise 8 first:", type(e).__name__, e)
''')
nb.md(r'''
Want a real interactive loop like the original lab? Uncomment and run (type `quit` to stop):

```python
# bot = ChatBot(num_beams=4)
# while (msg := input("> ")).lower() not in {"quit", "exit", "bye"}:
#     print("🤖", bot.ask(msg))
```
''')

nb.md(r'''
---
## 9. Compare models (optional, larger downloads)

The original lab also asks you to compare models. Swapping models is a one-line change, which is what the `Auto*`
classes give you. Set `RUN_COMPARISON = True` to try them (each is a several-hundred-MB download):

| model | size | type |
|---|---|---|
| `google/flan-t5-small` | 77M | encoder-decoder, instruction-tuned |
| `google/flan-t5-base` | 248M | same family, ~3× bigger: noticeably better answers |
| `facebook/blenderbot-400M-distill` | 365M | encoder-decoder, trained on **conversations** (chit-chat, not facts) |
''')
nb.code(r'''
RUN_COMPARISON = False
QUESTIONS = ["What is the capital of France?", "How are you today?", "What is 2 plus 3?"]

if RUN_COMPARISON:
    for name in ["google/flan-t5-base", "facebook/blenderbot-400M-distill"]:
        tok_ = AutoTokenizer.from_pretrained(name)
        mod_ = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
        print(f"\n=== {name} ===")
        for q in QUESTIONS:
            out = mod_.generate(**tok_(q, return_tensors="pt"), max_new_tokens=40)
            print(f"{q:<35} → {tok_.decode(out[0], skip_special_tokens=True)!r}")
else:
    print("Skipped. Set RUN_COMPARISON = True to compare models.")
''')
nb.md(r'''
---
## Recap

- **Hugging Face Hub + `transformers`**: `AutoTokenizer` + `AutoModelFor…` load any model by name.
- **Tokenization** turns text into ids. Every model has its own tokenizer, with special tokens like `</s>` and `<pad>`.
- **Generation is a loop**: forward pass → choose next token → append → repeat until EOS. `generate()` wraps it.
- **Decoding strategy** (greedy / beam / sampling + temperature / top-p) changes outputs a lot, and sampling needs a seed.
- **Chat memory is prompt-stuffing**, limited by the **context window**, so old turns must be dropped.
- **Small models hallucinate fluently.** Model choice and fine-tuning shape the output.

### Your score
''')
nb.code("lab.scoreboard()")

nb.write(COURSE, "lab_01_exploring_genai_libraries.ipynb")
