*This readme file documents my understanding to building miniGPT small-size LLM model inorder to better feel & understand it's workings.*

# miniGPT — Building a GPT from Scratch

Family of Large Language Models is nothing but sequence models, where the goal is to understand textual data and derive reasoning. Just like our brain when we read a book — it understands the plot, the nuances of characters, appreciates the writing — our brain concludes *"what a story"* or *"what a bullshit!"* or *"what happens now?"* at the end. Similarly, language models can reason, summarize, debate, and conclude just like our brains. The only open question is whether they can feel the emotions in the stories.

For example:
- Wavelength of red light → encodable
- Person says "that looks red" → encodable
- The actual felt experience of redness (philosophers call this **qualia**) → may be beyond symbolic representation

This chain of thought is out of scope for this guide. Here we focus on how to build a miniGPT.

---

## Architecture overview

Language models are based on the Transformer architecture, where every token looks at every other token simultaneously, in parallel, through **Attention**. The high-level structure:

```
Input tokens
    ↓
Embedding + positional encoding
    ↓
┌─────────────────────────────────┐
│  × N layers                     │
│  Multi-head self-attention      │
│  Add & norm (residual)          │
│  Feed-forward network (FFN)     │
│  Add & norm (residual)          │
└─────────────────────────────────┘
    ↓
Linear + softmax (output head)
    ↓
Next token prediction
```

---

## Stage 0 — Tokenization

> See `tokenizer.py`

Before anything, raw text must be broken into **tokens** — subword units represented as integer IDs. For instance, `"unbelievable"` might become `["un", "believ", "able"]`.

Design decisions around tokenizers depend on:
- What level of granularity you want the model to learn
- How you want to handle **OOV (Out of Vocabulary)** cases

We implement two tokenizers:

### Character-level tokenizer
Every unique character gets its own ID. Simple but produces long sequences — `"the cat"` becomes 7 tokens.

### BPE (Byte Pair Encoding)
Starts from all 256 possible bytes as a base vocabulary, then repeatedly merges the most frequent adjacent pair until a target vocabulary size is reached. BPE never produces OOV tokens because any text can be represented as bytes.

```
Round 1: ('t','h') → 'th'     most frequent pair
Round 2: ('th','e') → 'the'   next most frequent
Round 3: ('i','n') → 'in'
...repeat until vocab_size reached
```

**Key contract every tokenizer must satisfy:**
```python
assert tok.decode(tok.encode("Hello World")) == "Hello World"
```

---

## Stage 1 — Embeddings

> See `GPTEmbedding`, `TokenEmbedding`, `PositionalEmbedding` in `model.py`

In order to attend to every past token for every current token, two signals need to be encoded for each token: **what it is** and **where it is**.

### Token embeddings

Language models understand tokens as vectors in `N_EMBD`-dimensional space. `nn.Embedding(vocab_size, N_EMBD)` creates a lookup table — a matrix of shape `[vocab_size, N_EMBD]` where each row is a learnable vector for one token. At initialization these are random; after training they encode semantic meaning.

### Positional embeddings

At this stage the token vectors carry no positional meaning. A transformer processes all tokens in parallel with no inherent sense of order — `"cat sat"` and `"sat cat"` would be identical without position signals. Positional encoding adds a position-specific signal using the same dimension `N_EMBD`.

### Why the same dimension for both?

Both signals are combined using **matrix addition**, which requires identical shapes. The alternative is concatenation, but addition is preferred because it keeps the embedding size fixed at `N_EMBD` throughout all downstream layers — concatenation would double every subsequent weight matrix.

```python
x = token_emb + pos_emb   # [B, T, N_EMBD] + [T, N_EMBD] → [B, T, N_EMBD]
#                            broadcasting adds pos_emb to every sequence in batch
```

At this stage you have a tensor of shape `[B, T, N_EMBD]` where:

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size — number of sequences processed in parallel |
| `T` | Sequence length — context window |
| `N_EMBD` | Embedding vector size per token |

---

## Stage 2 — Single-head attention

> See `Head` class in `attention.py` and `test_attention.py`

This is where every token learns relationships to every other token in the sequence. Take the sentence: *"The river bank was flooded"*. Attention learns a high weight between `"bank"` and `"river"` to disambiguate `"bank"` as geographical rather than financial.

### Q, K, V — three projections, three jobs

Each token simultaneously plays three roles via learned linear projections:

| | Q | K | V |
|---|---|---|---|
| **Role** | Searcher | Advertiser | Content carrier |
| **Active/Passive** | Active — goes looking | Passive — waits to be found | Passive — shares when found |
| **Used for** | Computing scores | Computing scores | Computing output |
| **Can be zero-magnitude?** | Yes — nouns in coreference head barely search | No — always advertising | No — always carrying content |
| **Symmetric with counterpart?** | Q·K asymmetric by design | K·Q asymmetric by design | Independent of Q, K |

`"river"`'s K matches `"bank"`'s Q strongly — high dot product. So `"bank"` attends heavily to `"river"`, pulling `"river"`'s V (which carries geographical meaning) into its own representation. Now `"bank"` knows it's geographical.

The crucial point: `"river"`'s K and V don't have to encode the same thing. Its K might encode *"I am a water-related noun"* (useful for routing). Its V might encode *"rivers flow, have banks, are geographical features"* (useful for content transfer).

### Why not just Q and K, without V?

```python
# Without V — use raw x as values
weights = softmax(q @ k.transpose(-2, -1) * scale)
out = weights @ x   # raw x instead of projected v
```

This mechanically works but forces every token to share its entire `N_EMBD`-dimensional representation when attended to. V gives the model a separate learnable projection to control *what information gets shared* — different heads can share different aspects of the same token.

### Why not Q and V, without K (i.e. K = Q)?

```python
q = x @ Wq
k = q              # same as query
scores = q @ k.transpose(-2, -1)   # = q @ qᵀ
```

`q @ qᵀ` is a symmetric matrix — `score[i][j]` always equals `score[j][i]`. This forces attention to be symmetric: if token i attends strongly to j, then j attends equally strongly to i. But attention should be asymmetric. In *"she gave him the award she won"*, `"won"` should attend strongly to `"she"` (to find its subject), but `"she"` doesn't need to attend to `"won"` with the same weight. Separate K and Q matrices allow this asymmetry.

### Why not more than three — like Q, K, V, U?

Three projections cleanly cover the three distinct operations:
- `Q × K` → routing (who attends to whom)
- `V` → content (what gets transferred)

A fourth matrix would duplicate something already covered. In practice, expressiveness comes from having **multiple heads** each with their own Q, K, V — not from adding more projections per head.

### The five steps inside one attention head

Input tensor of shape `[B, T, N_EMBD]` transforms to `[B, T, HEAD_SIZE]`:

```
input x [B, T, N_EMBD]
    ↓
1. Project to Q, K, V          — three separate linear layers
    ↓
2. Compute attention scores     — Q · Kᵀ / √dₖ  (scale prevents softmax saturation)
    ↓
3. Apply causal mask            — future tokens → -inf
    ↓
4. Softmax                      — -inf becomes exactly 0.0, rows sum to 1.0
    ↓
5. Weighted sum of V            — weights · V
    ↓
output [B, T, HEAD_SIZE]
```

The causal mask ensures token at position `i` cannot attend to positions `j > i` — preventing the model from cheating by peeking at future tokens during training.

---

## Stage 3 — Multi-head attention

> See `MultiHeadAttention` class in `attention.py` and `test_multihead.py`

You've built one head. Now run several in parallel, each learning different relationship types, then combine them.

```
input x [B, T, N_EMBD]
    ↓
┌─────────────────────────────────────┐
│  Head 0  →  [B, T, HEAD_SIZE]       │
│  Head 1  →  [B, T, HEAD_SIZE]       │  all heads run in parallel
│  Head 2  →  [B, T, HEAD_SIZE]       │
│  Head 3  →  [B, T, HEAD_SIZE]       │
└─────────────────────────────────────┘
    ↓ concat along last dim
[B, T, N_HEADS × HEAD_SIZE]  =  [B, T, N_EMBD]
    ↓ linear projection Wo
output [B, T, N_EMBD]
```

> **Constraint:** `N_EMBD` must be exactly divisible by `N_HEADS`. This ensures each head gets equal capacity: `HEAD_SIZE = N_EMBD // N_HEADS`.

### Why project with Wo after concat?

After concatenating heads you have `N_EMBD` numbers — but they are independently computed slices with no relationship to each other. Head 0's output in dimension 3 has no relationship to Head 2's output in dimension 35. The next layer (FFN or next transformer block) needs to reason across all dimensions together, but the information is sitting in separate compartments.

`Wo` is a `[N_EMBD, N_EMBD]` learned matrix that allows every output dimension to be a combination of ALL head outputs:

```
output_dim_7 = 0.8×(head0 syntactic signal) + 0.6×(head1 coreference signal) + ...
```

This synthesis is information that neither head alone could provide. `Wo` transforms isolated head outputs into a single coherent representation.

---

## Stage 4 — Transformer block

> See `Block` and `FFN` classes in `model.py` and `test_block.py`

The transformer block wraps multi-head attention and FFN together with LayerNorm and residual connections.

```
input x [B, T, N_EMBD]
    ↓
LayerNorm
    ↓
MultiHeadAttention      ← Stage 3
    ↓
residual connection     (x = x + attn_out)
    ↓
LayerNorm
    ↓
FFN
    ↓
residual connection     (x = x + ffn_out)
    ↓
output [B, T, N_EMBD]
```

### LayerNorm

For each token independently, normalizes its `N_EMBD` values to mean=0, std=1, then applies learned scale and shift:

```
output = gamma × ((x - mean) / std) + beta
           ↑              ↑               ↑
        learned       fixed math       learned
        scale        normalization      shift
```

`gamma` (scale) and `beta` (shift) are vectors of size `N_EMBD` — one value per embedding dimension, both learned during training. At initialization `gamma=1` and `beta=0`, so LayerNorm starts as pure normalization and gradually learns useful rescaling.

**Why LayerNorm?** As information flows through many layers, activations drift wildly:

```
without LayerNorm: activations after 12 layers → [0.0001, 847.3, 0.003, ...]
with LayerNorm:    activations after 12 layers → [0.3, -0.8, 1.2, ...]  ← controlled
```

**Why gamma and beta?** Without them, LayerNorm would permanently force mean=0, std=1 — the model could never learn that some dimensions should be larger or more biased than others. Scale and shift give back the expressiveness that normalization removes, but in a stable, controlled way.

### Residual connections

After each sub-layer, the original input is added back:

```python
x = x + attention(layernorm(x))   # not just attention(x)
x = x + ffn(layernorm(x))         # not just ffn(x)
```

Each layer only learns a small correction `Δ` on top of what's already there:

```
output = input + Δ
```

**Two benefits:**

1. **Gradient highway.** The `+x` addition creates a direct path with gradient = 1 from output all the way back to input — gradients don't have to travel through all the attention and FFN computation and risk vanishing or exploding. Without residuals, gradients shrink or explode through 12 layers of multiplications and early layers receive essentially zero training signal.

2. **Selective layers.** If a layer learns `Δ=0` (nothing useful), the input passes through unchanged — `x = x + 0 = x`. The layer is silently skipped without corrupting the computation. Without residuals, one bad layer corrupts every layer that follows.

### Feed-forward network (FFN)

The FFN processes each token independently after attention has mixed context across tokens. It expands then contracts:

```python
nn.Sequential(
    nn.Linear(N_EMBD, 4 * N_EMBD),   # expand — 4× breathing room for computation
    nn.GELU(),                         # non-linearity — makes two linear layers non-trivial
    nn.Linear(4 * N_EMBD, N_EMBD),   # contract — back to N_EMBD
    nn.Dropout(dropout),
)
```

**Why 4× expansion?** The expanded space lets the model compute many intermediate features simultaneously (syntactic role, semantic category, positional relationships, etc.) before distilling them back down. Smaller ratios underperform; larger give diminishing returns.

**Why GELU and not just two linear layers?** Two linear layers without a non-linearity are mathematically equivalent to one linear layer — they collapse. GELU introduces the non-linearity that makes the FFN expressive.

**Attention vs FFN — complementary roles:**
- **Attention** mixes information *across* tokens — `"bank"` gathers context from `"river"`
- **FFN** transforms information *within* each token — given `"bank"` now has `"river"` context, reason about what that means

---

## Stage 5 — Full GPT model

> See `GPT` class in `model.py` and `test_gpt.py`

Assembles everything into one complete architecture:

```
input idx [B, T]                    ← token IDs
    ↓
GPTEmbedding                        ← Stage 1
    ↓
Block × N_LAYERS                    ← Stage 4 repeated N times
    ↓
LayerNorm (final)
    ↓
lm_head: Linear(N_EMBD, vocab_size) ← project to vocabulary
    ↓
logits [B, T, vocab_size]           ← raw scores, one per token per position
```

**Weight tying:** The token embedding matrix `[vocab_size, N_EMBD]` and `lm_head` weight matrix `[vocab_size, N_EMBD]` are the same object in memory. Both are answering the same question — "what does each token look like in embedding space?" — so sharing them saves parameters and improves performance.

**Why N transformer layers?** One layer is one step of reasoning. Complex language understanding requires building understanding progressively — simple patterns first, abstract reasoning later. Each layer refines the representations produced by the previous one:

```
Layer 1:   punctuation, word boundaries, basic syntax
Layer 4:   parts of speech, grammatical roles
Layer 8:   semantic relationships, entity tracking
Layer 12:  high-level reasoning, task understanding
```

---

## Stage 6 — Training

> See `train.py`

### Data preparation

The dataset is encoded into a flat tensor of token IDs. Batches are random windows into this tensor:

```python
x = data[i   : i + block_size]      # input:  tokens 0..T-1
y = data[i+1 : i + block_size + 1]  # target: tokens 1..T (shifted by 1)
```

`x` and `y` are the same sequence offset by one position. The model learns to predict `y` from `x` — every position in the sequence provides one training signal simultaneously.

### Loss function

Cross-entropy loss measures how wrong the model is at predicting the next token:

```
loss = -log(probability assigned to correct next token)
```

At random initialization with vocab size 65:
```
expected loss = -log(1/65) = log(65) ≈ 4.17
```

This is the sanity check — initial loss should be near `log(vocab_size)`. If it's much lower, initialization is wrong. If much higher, something is broken.

### Training loop

```python
logits, loss = model(x, y)   # forward pass
optimizer.zero_grad()         # clear previous gradients
loss.backward()               # compute gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent explosion
optimizer.step()              # update weights
```

**Gradient clipping** caps the gradient norm at 1.0 — preventing any single bad batch from sending weights to infinity.

**AdamW optimizer** maintains momentum (remembers recent gradient directions) and per-weight adaptive learning rates, making training significantly more stable than plain gradient descent.

### Batching

Training on one example at a time produces noisy gradients that oscillate wildly. A batch of 64 sequences averages those gradients into a stable signal:

```
batch_size = 1:   gradient estimate very noisy → training oscillates
batch_size = 64:  gradient estimate stable → training converges
```

Batch size is limited by GPU memory — the larger the batch, the more activations must be held in memory simultaneously. For a T4 GPU (16GB), `batch_size=64` with `block_size=256` and `N_EMBD=384` fits comfortably.

---

## Hyperparameters

| Parameter | Tiny (debugging) | Small (training) |
|-----------|-----------------|-----------------|
| `block_size` | 32 | 256 |
| `N_EMBD` | 64 | 384 |
| `N_HEADS` | 4 | 6 |
| `N_LAYERS` | 3 | 6 |
| `batch_size` | 16 | 64 |
| `dropout` | 0.0 | 0.1 |
| `lr` | 3e-4 | 3e-4 |

Start with the tiny config to verify everything works. Scale to small for a real training run on Shakespeare.

---

## Project structure

```
gpt-from-scratch/
├── tokenizer.py        # CharTokenizer and BPETokenizer
├── model.py            # GPTEmbedding, FFN, Block, GPT
├── attention.py        # Head, MultiHeadAttention
├── train.py            # training loop and text generation
├── test_tokenizer.py
├── test_embeddings.py
├── test_attention.py
├── test_multihead.py
├── test_block.py
├── test_gpt.py
└── data/
    └── shakespeare.txt
```

---

## Getting started

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/gpt-from-scratch.git
cd gpt-from-scratch

# Install dependencies (Python 3.11 required)
pip3.11 install torch numpy matplotlib

# Run tests locally
python3.11 test_tokenizer.py
python3.11 test_embeddings.py
python3.11 test_attention.py
python3.11 test_multihead.py
python3.11 test_block.py
python3.11 test_gpt.py

# Train on Shakespeare (run in Colab for GPU)
python train.py
```