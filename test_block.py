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
    
# -- Block Tests ------------------------------------

def test_block_output_shape():
    block = Block(N_EMBED, N_HEADS, BLOCK_SIZE)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)

    out = block(x)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, N_EMBED), \
        f"Expected {x.shape}, got {out.shape}"
    print(f"PASS: Block output shape {out.shape}")


def test_residual_connection():
    """
    Residual connection means output != just attention(x)
    Output must be x + attention(x).
    """
    block = Block(N_EMBED, N_HEADS, BLOCK_SIZE)
    block.eval()

    # Zero initialize all weights so Attention + FFN output = 0
    # Then residual output should be 0 as x + 0 -> x
    with torch.no_grad():
        for p in block.parameters():
            p.zero_()

    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
    with torch.no_grad():
        out = block(x)
    
    # With zeroed weights attn(ln(x)) = 0, ffn(ln(x)) = 0
    # so: x + 0 = x; x + 0 = x
    # output should be equal to input
    assert torch.allclose(out, x, 1e-5), \
        f"Residual broken - zeroed weights should give poutput ~= input." 
    print(f"PASS: residual connection works -> zeroed weights given output ~= input")


def test_layernorm_applied():
    block = Block(N_EMBED, N_HEADS, BLOCK_SIZE)
    block.eval()

    ln1_output = []
    def hook(module, input, output):
        ln1_output.append(output)
    block.ln1.register_forward_hook(hook)

    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED) * 100 # large scale input
    with torch.no_grad():
        block(x)
    
    assert len(ln1_output) > 0, \
        f"Hook should fire ln1 and list should not be empty"
    
    ln_out = ln1_output[0]
    mean = ln_out.mean(dim=-1)
    std = ln_out.std(dim=-1)

    assert mean.abs().max().item() < 1e-4, \
        f"LayerNorm mean should be = 0, got {mean.abs().max().item()}"
    assert (std-1).abs().max().item() < 0.1, \
        f"LayerNorm std should be = 1, got max std={std.max().item()}"
    print("PASS: LayerNorm normalizes activations to mean=0, std=1")


def test_block_gradient_flow():
    block = Block(N_EMBED, N_HEADS, BLOCK_SIZE)
    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
    out = block(x)
    loss = out.sum()
    loss.backward()

    assert block.ln1.weight.grad is not None, "No grad in ln1 weights"
    assert block.ln2.weight.grad is not None, "No grad in ln2 weights"

    for i, head in enumerate(block.attn.heads):
        assert head.query.weight.grad is not None, "No grad in attn Q weights"
        assert head.key.weight.grad is not None, "No grad in attn K weights"
        assert head.value.weight.grad is not None, "No grad in attn V weights"
    
    assert block.ffn.net[0].weight.grad is not None, "No grad in FFN layer 1"
    assert block.ffn.net[2].weight.grad is not None, "No grad in FFN layer 2"

    print("PASS: gradients flow through LayerNorm, Attention & FFN")


def test_stacking_blocks():
    """
    Multiple blocks must be stackable - output of one feeds into next.
    """
    blocks = nn.Sequential(*[
        Block(N_EMBED, N_HEADS, BLOCK_SIZE) for _ in range(3)
    ])

    x = torch.randn(BATCH_SIZE, BLOCK_SIZE, N_EMBED)
    out = blocks(x)
    assert out.shape == x.shape, \
        f"Stacked blocks broke shape: {out.shape}"
    print(f"PASS: 3 stacked blocks work - input shape {x.shape} preserved in output {out.shape}")


def test_variable_sequence_length():
    block = Block(N_EMBED, N_HEADS, BLOCK_SIZE)
    for T in [1, 8, 16, 32]:
        x = torch.randn(BATCH_SIZE, T, N_EMBED)
        out = block(x)
        assert x.shape == out.shape, \
            f"Failed for {T}: Expected {x.shape}, got {out.shape}"
    print("PASS: variable sequence length works")



if __name__ == '__main__':
    torch.manual_seed(42)
    test_ffn_output_shape()
    test_ffn_expands_then_contracts()
    test_ffn_per_token()

    test_block_output_shape()
    test_residual_connection()
    test_layernorm_applied()
    test_block_gradient_flow()
    test_stacking_blocks()
    test_variable_sequence_length()