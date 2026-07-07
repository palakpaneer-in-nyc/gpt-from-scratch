import torch
import torch.nn as nn
from attention import MultiHeadAttention

BATCH_SIZE = 4
BLOCK_SIZE = 32
N_EMBED = 64
N_HEADS = 4
HEAD_SIZE = 16 # 64/4 -> 16 derived. 

def test_output_shape():
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
    mheads = MultiHeadAttention(N_EMBED, N_HEADS, BLOCK_SIZE)
    out = mheads(x)

    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, N_EMBED), \
    f"Expected {(BATCH_SIZE, BLOCK_SIZE, N_EMBED)}, got {out.shape}"

    print(f"PASS: output shape {out.shape}")

    assert x.shape == out.shape, \
        f"Expected same input {(x.shape)} and output {out.shape} shape"
    
    print(f"PASS: Input shape {(x.shape)} is same as output shape {(out.shape)}")


def test_n_heads_and_head_size():
    mheads = MultiHeadAttention(N_EMBED, N_HEADS, BLOCK_SIZE)
    head_size = N_EMBED // N_HEADS

    assert len(mheads.heads) == N_HEADS, \
        f"Expected {N_HEADS} heads, but found {len(mheads.heads)} heads"
    print(f"PASS: Expected {N_HEADS} heads created")
    
    assert mheads.head_size == head_size, \
        f"Expected {head_size} head size per head, but found {mheads.head_size}"
    print(f"PASS: Expected {head_size} head size per head")


def test_independent_heads():
    mheads = MultiHeadAttention(N_EMBED, N_HEADS, BLOCK_SIZE)

    for i in range(N_HEADS):
        for j in range(i+1, N_HEADS):
            q_weight_i = mheads.heads[i].query.weight
            q_weight_j = mheads.heads[j].query.weight

            assert not torch.allclose(q_weight_i, q_weight_j), \
                f"Heads {i} and {j} share identical query weights - not INDEPENDENT"
    print("PASS: All heads have indepenent Wq, Wk and Wv weights")


def test_single_head_equals_multihead_n1():
    mheads = MultiHeadAttention(
        n_embed=N_EMBED,
        n_heads=1,
        block_size=BLOCK_SIZE)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
    out = mheads(x)

    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, N_EMBED), \
        f"Expected {(BATCH_SIZE, BLOCK_SIZE, N_EMBED)} shape, but got {out.shape}"
    print(f"PASS: n_heads = 1 produces correct output shape ({out.shape})")


def test_invalid_n_heads_raises():
    try:
        mheads = MultiHeadAttention(
            n_embed=N_EMBED,
            n_heads=3, 
            block_size=BLOCK_SIZE
        )
        assert False, "should have raised AssertionError"
    except AssertionError:
        print("PASS: Raised invalid n_head error")


def test_gradient_flow():
    mheads = MultiHeadAttention(N_EMBED, N_HEADS, BLOCK_SIZE)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
    out = mheads(x)
    loss = out.sum()
    loss.backward()

    for i, head in enumerate(mheads.heads):
        assert head.query.weight.grad is not None, \
            f"No gradient in head {i} query"
        assert head.key.weight.grad is not None, \
            f"No gradient in head {i} key"
        assert head.value.weight.grad is not None, \
            f"No gradient in head {i} value"

    mheads.proj.weight.grad is not None, \
        f"No gradient in output projection Wo"
    print(f"PASS: gradient flow through all {N_HEADS} heads and Wo")


if __name__ == '__main__':
    torch.manual_seed(42)
    test_output_shape()
    test_n_heads_and_head_size()
    test_independent_heads()
    test_single_head_equals_multihead_n1()
    test_invalid_n_heads_raises()
    test_gradient_flow()