import torch
import torch.nn as nn
from attention import Head

BATCH_SIZE = 4
BLOCK_SIZE = 32
N_EMBED = 64
# n_embed / n_heads. say n_head = 4, 64/4 -> 16 
HEAD_SIZE = 16 # remember this is not a hyper-parameter that needs tuning.

def test_output_shape():
    head = Head(
        head_size=HEAD_SIZE,
        n_embed=N_EMBED,
        block_size=BLOCK_SIZE)
    
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
    out = head(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE), \
    f"Expected {(BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE)}"
    print(f"PASS: input shape {(BATCH_SIZE, BLOCK_SIZE, N_EMBED)} -> "
          f"output shape {(BATCH_SIZE, BLOCK_SIZE, HEAD_SIZE)}")
    

def test_casual_mask():
    """
    Critical test - future tokens must have zero attention weight.
    Token at position i must NOT attend to positions j > i.
    """

    head = Head(HEAD_SIZE, N_EMBED, BLOCK_SIZE)
    head.eval() # method derived from nn.Module torch module
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)

    with torch.no_grad():
        _, weights = head(x, return_weights=True)
        upper = weights[0].triu(diagonal=1)
        assert upper.sum().item() == 0.0, \
        f"Future tokens shouldn't have non-zero weights: {upper.sum()}"
    print(f"PASS: Future tokens should have zero weights.")


def test_token_row_sum_1():
    """
    Each row corresponding to a token in weight matrix must sum to 1.
    """
    head = Head(HEAD_SIZE, N_EMBED, BLOCK_SIZE)
    head.eval()

    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)

    with torch.no_grad():
        _, weights = head(x, return_weights=True) # weights = [B, T, T]

        row_sums = weights[0].sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(BLOCK_SIZE), atol=1e-5), \
            f"Rows don't sum to 1: {row_sums}"
        print(f"PASS: attention weights sum to 0 per row")
        
        assert abs(weights[0][0][0]) == 1.0, \
        f"first token should 100% attend to itself."
        print("PASS: first tokens attends to only itself")


def test_variable_sequence_length():
    head = Head(HEAD_SIZE, N_EMBED, BLOCK_SIZE)
    head.eval()

    T = 7
    x = torch.randn(BATCH_SIZE, T, N_EMBED)
    out, weights = head(x, return_weights=True)
    assert weights.shape == (BATCH_SIZE, T, T), \
        f"Weights should be {(BATCH_SIZE, T, T)}"
    
    assert out.shape == (BATCH_SIZE, T, HEAD_SIZE), \
        f"Output shape should be {(BATCH_SIZE, T, HEAD_SIZE)}"
    
    print("PASS: variable sequence length")

def test_gradients_flow():
    """Gradients must reach Q, K, V projection weights."""
    head = Head(HEAD_SIZE, N_EMBED, BLOCK_SIZE)
    head.eval()
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
    out = head(x)
    loss = out.sum()
    loss.backward()

    assert head.query.weight.grad is not None, "No gradient in Q weights"
    assert head.key.weight.grad is not None, "No gradient in K weights"
    assert head.value.weight.grad is not None, "No gradient in V weights"

    print("PASS: gradients flow through Q, K, V weights")


if __name__ == '__main__':
    torch.manual_seed(42)
    test_output_shape()
    test_casual_mask()
    test_token_row_sum_1()
    test_variable_sequence_length()
    test_gradients_flow()