import torch
from model import GPTEmbedding

VOCAB_SIZE = 65 # Shakespeare character vocab
BLOCK_SIZE = 32 # context length
N_EMBED = 64    # embedding dimension
BATCH_SIZE = 4

def test_output_shape():
    embed = GPTEmbedding(
        vocab_size=VOCAB_SIZE, 
        block_size=BLOCK_SIZE, 
        n_embed=N_EMBED)
    idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    out = embed(idx)
    assert out.shape == (BATCH_SIZE, BLOCK_SIZE, N_EMBED), \
    f"Expected {(BATCH_SIZE, BLOCK_SIZE, N_EMBED)}, got {out.shape}"


if __name__ == '__main__':
    test_output_shape()