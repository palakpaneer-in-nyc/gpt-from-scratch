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
    token_1 = embed.token_embed(torch.tensor([[1]])) # (batch size) B = 1, (block size) T = 1
    token_2 = embed.token_embed(torch.tensor([[2]])) # (batch size) B = 1, (block size) T = 1

    assert not torch.allclose(token_1, token_2), \
    "Different tokens must have different embeddings"
    print("PASS: Different tokens must have different embeddings")

    # Test2: Same token should always return same embeddings
    token_a_1 = embed.token_embed(torch.tensor([[5]])) # assume id/value for token 'a' is 5
    token_a_2 = embed.token_embed(torch.tensor([[5]])) # same as above
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


def test_same_token_diff_positions():
    embed = GPTEmbedding(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embed=N_EMBED
        # dropout=0.0
    )
    embed.eval() # defaults to dropout = 0.0
    idx = torch.tensor([[5, 10, 21, 32, 5, 53, 34, 17]]) # B = 1, T = 8
    out = embed(idx) # shape [1, 8, 64]

    vec_at_pos_0 = out[0][0]
    vec_at_pos_4 = out[0][4]

    assert not torch.allclose(vec_at_pos_0, vec_at_pos_4), \
    "Same token at different position must produce diff output vector"
    print(f"PASS: Same token at different position will produce different output vector")

    tok_id = torch.tensor([[5]])
    tok_emb_vec = embed.token_embed(tok_id)

    pos_enc_all = embed.pos_embed(8)
    pos_0_enc = pos_enc_all[0]
    pos_4_enc = pos_enc_all[4]

    # Reconstruct: final vectors of 5 based on positions.
    reconstructed_pos_0 = tok_emb_vec.squeeze() + pos_0_enc
    reconstructed_pos_4 = tok_emb_vec.squeeze() + pos_4_enc

    assert torch.allclose(reconstructed_pos_0, vec_at_pos_0, atol=1e-6), \
    "Reconstruction at pos0 must match forward pass output"
    assert torch.allclose(reconstructed_pos_4, vec_at_pos_4, atol=1e-6), \
    "Reconstruction at pos4 must match forward pass output"
    print("PASS: final vector = token embed + position embed confirmed")


def test_variable_sequence_length():
    embed = GPTEmbedding(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embed=N_EMBED
    )
    for T in [1, 8, 16, 32]:
        idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, T))
        out = embed(idx)
        assert out.shape == (BATCH_SIZE, T, N_EMBED)
    print("PASS: Variable sequence length works")


def test_gradients_flow():
    embed = GPTEmbedding(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_embed=N_EMBED
    )

    idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    out = embed(idx)
    loss = out.sum()
    loss.backward()
    assert embed.token_embed.embedding.weight.grad is not None
    assert embed.pos_embed.embedding.weight.grad is not None
    print(f"PASS: gradients flow through the embeddings")


if __name__ == '__main__':
    test_output_shape()
    test_token_pos_different()
    test_same_token_diff_positions()
    test_variable_sequence_length()
    test_gradients_flow()