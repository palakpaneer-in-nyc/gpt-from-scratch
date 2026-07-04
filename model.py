import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Learnable lookup table: token_id -> vector of size n_embed
    Internally just a matrix of shape [vocab_size, n_embed].
    Each row is the embedding for one token.
    """
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)

    def forward(self, idx):
        # idx shape: [B, T] (batch size, sequence length)
        # output : [B, T, n_embed]
        return self.embedding(idx)
    

class PositionalEmbedding(nn.Module):
    """
    Learnable lookup table: position_index -> vector of size n_embed
    Same structure as token embedding, but indexes by position, not token_id.
    """
    def __init__(self, block_size, n_embed):
        super().__init__()
        self.embedding = nn.Embedding(block_size, n_embed)

    def forward(self, T):
        # T: current sequence length
        # positions: [0, 1, 2, 3, ...., T-1]
        positions = torch.arrange(T)  # shape: [T]
        return self.embedding(positions)  # shape: [T, n_embed]
    

class GPTEmbedding(nn.Module):
    """
    Full embedding layer: combines token + positional embeddings.
    This is the first thing every input passes through.
    """
    def __init__(self, vocab_size, block_size, n_embed, dropout = 0.1):
        super().__init__()
        self.token_emb = TokenEmbedding(vocab_size, n_embed)
        self.pos_emb = PositionalEmbedding(block_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        # idx shape: [B, T]
        B, T = idx.shape

        tok = self.token_emb(idx)   # [B, T, n_embed]
        pos = self.pos_emb(T)       # [T, n_embed]

        # pos broadcast across batch dimension automatically
        # [B, T, n_embed] + [T, n_embed] = [B, T, n_embed]
        x = tok + pos

        return self.dropout(x)