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
    print(f"PASS: output shape {out.shape}")


def test_token_pos_different():
    """
    Token & positional embedding must be independent.
    """
    embed = GPTEmbedding(
        vocab_size=VOCAB_SIZE, # total distinct tokens -> vocab size
        block_size=BLOCK_SIZE, # sequence length -> context window
        n_embed=N_EMBED # dimension to represent a token & its position
    )

    # Test1: Different token should always return different embeddings
    token_1 = embed.tok_embed(torch.tensor([[1]])) # (batch size) B = 1, (block size) T = 1
    token_2 = embed.tok_embed(torch.tensor([[2]])) # (batch size) B = 1, (block size) T = 1

    assert not torch.allclose(token_1, token_2), \
    "Different tokens must have different embeddings"
    print("PASS: Different tokens must have different embeddings")

    # Test2: Same token should always return same embeddings
    token_a_1 = embed.tok_embed(torch.tensor([[5]])) # assume id/value for token 'a' is 5
    token_a_2 = embed.tok_embed(torch.tensor([[5]])) # same as above
    assert torch.allclose(token_a_1, token_a_2), \
    "Same token must always return same embedding"
    print(f"PASS: Same token -> same embedding always")

    pos_0 = embed.pos_embed(1) # sequence length a.k.a. T = 1 -> position 0
    pos_3 = embed.pos_embed(3) # sequence length a.k.a. T = 3 -> position 0,1,2

    assert torch.allclose(pos_0, pos_3[:1]), \
    "Position embedding at a specific position must be equal regardless of sequence length"
    print("PASS: same position 0 encoding consistent across sequence length")

    pos_embed_all = embed.pos_embed(4) # get encodings for 0, 1, 2, 3
    assert not torch.allclose(pos_embed_all[0], pos_embed_all[1]), \
    "Different positions must have different encodings"
    print("PASS: different positions -> different encodings")

    


if __name__ == '__main__':
    test_output_shape()