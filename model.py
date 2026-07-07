import torch
import torch.nn as nn
from attention import MultiHeadAttention

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
        positions = torch.arange(T)  # shape: [T]
        return self.embedding(positions)  # shape: [T, n_embed]
    

class GPTEmbedding(nn.Module):
    """
    Full embedding layer: combines token + positional embeddings.
    This is the first thing every input passes through.
    """
    def __init__(self, vocab_size, block_size, n_embed, dropout = 0.1):
        super().__init__()
        self.token_embed = TokenEmbedding(vocab_size, n_embed)
        self.pos_embed = PositionalEmbedding(block_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx):
        # idx shape: [B, T]
        B, T = idx.shape

        tok = self.token_embed(idx)   # [B, T, n_embed]
        pos = self.pos_embed(T)       # [T, n_embed]

        # pos broadcast across batch dimension automatically
        # [B, T, n_embed] + [T, n_embed] = [B, T, n_embed]
        x = tok + pos

        return self.dropout(x)
    

class FFN(nn.Module):
    """
    Feed Forward Network -> applied to each token independently (test - independently).
    2 linear layers with GeLU activation between them (wtf is that - LOL).
    Inner dimension is 4x n_embed -> standard transformer convention (wtf is that again - LOLLLLZZ)

    Takes:      [B, T, n_embed]
    Returns:    [B, T, n_embed] <- cool atleast stick to n_embed, but god knows what transformation now? 
    """

    def __init__(self, n_embed, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # expand to 4x width -> why?
            nn.GELU(), # activation (smooth, non-zero for negatives -> let the negatives pass through)
            nn.Linear(4 * n_embed, n_embed), # project back down -> what's happening??
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) # [B, T, n_embed] -> [B, T, n_embed]
    

class Block(nn.Module):
    """
    Full transformer block.
    Pre-LayerNorm architecture: LayerNorm applied before attention and FFN.
    Residual connections after both sub-layers.

    Takes:      x of shape [B, T, n_embed]
    Returns:    x of shape [B, T, n_embed]
    """

    def __init__(self, n_embed, n_heads, block_size, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FFN(n_embed, dropout)

    def forward(self, x):
        # Pre-LN + residual connection around attention
        x = x + self.attn(self.ln1(x))

        # Pre-LN + residual connection around FFN
        x = x + self.ffn(self.ln2(x))
        return x