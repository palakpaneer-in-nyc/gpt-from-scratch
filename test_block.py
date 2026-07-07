import torch
import torch.nn as nn
from model import FFN, Block

BATCH_SIZE = 4
BLOCK_SIZE = 32
N_EMBED = 64
N_HEADS = 4

# -- FFN Tests ---------------------------------

def test_ffn_output_shape():
    ffn = FFN(N_EMBED)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)

    out = ffn(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, N_EMBED), \
        f"Expected {(BATCH_SIZE, BLOCK_SIZE, N_EMBED)} shape, got {out.shape}"
    print(f"PASS: output shape {out.shape}")


def test_ffn_expands_then_contracts():
    ffn = FFN(N_EMBED)
    # First linear: n_embed -> 4 * n_embed
    assert ffn.net[0].out_features == 4 * N_EMBED, \
        f"First layer should be expanded to {4 * N_EMBED}"
    
    # Second linear: 4 * n_embed -> n_embed
    assert ffn.net[2].out_features == N_EMBED, \
        f"Second layer should contract to {N_EMBED}"

    print(f"PASS: FNN expands to {4 * N_EMBED} then contracts to {N_EMBED}")


def test_ffn_per_token():
    ffn = FFN(N_EMBED)
    ffn.eval()
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)

    out1 = ffn(x)

    x_modified = x.clone()
    x_modified[0, 5, :] = torch.randn(N_EMBED)
    out_modified = ffn(x_modified)

    assert torch.allclose(out1[0, 0, :], out_modified[0, 0, :]), \
        f"FNN should process each token independently -> changing token at 5 affects output at 0"
    print("PASS: FFN is process each token independently -> tokens don't affect each other")
    

if __name__ == '__main__':
    torch.manual_seed(42)
    test_ffn_output_shape()
    test_ffn_expands_then_contracts()
    test_ffn_per_token()
