import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """
    Single self-attention head. 

    Takes: x shape of [B, T, n_embed]
    Returns: output of shape [B, T, head_size]
    """

    def __init__(self, head_size, n_embed, block_size, dropout=0.0):
        super().__init__()

        # Create Q, K, V weight matrix of shape [head_size, n_embed]
        self.query  =   nn.Linear(n_embed, head_size, bias=False)
        self.key    =   nn.Linear(n_embed, head_size, bias=False)
        self.value  =   nn.Linear(n_embed, head_size, bias=False)

        # Casual mask - lower triangular matrix of 1s
        # registered as buffer so it moves to GPU with the model
        # but is NOT a learnable paramater.
        self.register_buffer(
            'tril',
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout    =   nn.Dropout(dropout)
        self.head_size  =   head_size


    def forward(self, x, return_weights=False):
        B, T, C = x.shape # C = n_embed

        # Step1: Project input to Q, K, V - nn.Linear(x) does matrix multiplictaion.
        # [B, T, n_embed] @ [n_embed, head_size] (w*T transposed weight matrix) -> [B, T, head_size]  
        q = self.query(x) # x @ WqT (i.e. Wq transposed) [B, T, head_size]
        k = self.key(x)   # x @ WkT (i.e. Wk transposed) [B, T, head_size]
        v = self.value(x) # x @ WvT (i.e. Wv transposed) [B, T, head_size]

        # Step2: compute attention score
        # q @ k.transpose(-2, -1) does batched matrix multiply
        # [B, T, head_size] @ [B, head size, T] -> [B, T, T]
        scale = self.head_size ** -0.5
        scores = q @ k.transpose(-2, -1) * scale # [B, T, T]

        # Step3: Masking
        # positions where tril == 0 as future toekns -> set to -inf
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        # Step4: softmax - turns scores into weights that sum to 1 for each token (on x axis)
        weights = F.softmax(scores, dim=-1) # [B, T, T]
        weights = self.dropout(weights)

        # Step5: weighted sum of values
        # [B, T, T] @ [B, T, head_size] -> [B, T, head_size]
        out = weights @ v

        if return_weights:
            return out, weights
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel.
    Takes: x of shape [B, T, n_embed]
    Returns: output of shape [B, T, n_embed]
    """ 
    def __init__(self, n_heads, n_embed, block_size, dropout=0.0):
        super().__init__()
        assert n_embed & n_heads == 0, \
        f"n_embed ({n_embed}) must be divisible by n_heads ({n_heads})"

        self.head_size = n_embed // n_heads
        self.n_heads = n_heads

        self.heads = nn.ModuleList([
            Head(self.head_size, n_embed, block_size, dropout)
            for _ in range(n_heads)
        ])

        # Ouput projection Wo -> project concatenated heads back to embedding space, n_embed
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        # Run all heads in parallel.
        if return_weights:
            head_outputs = []
            head_weights = []
            for h in self.heads:
                out, w = h(x, return_weights=True)
                head_outputs.append(out)
                head_weights.append(w)
        else:
            head_outputs = [h(x) for h in self.heads]

        # Concatenate along the last dimension which is n_embed
        # each head: [B, T, head_size]
        # after cat: [B, T, n_heads X head_size] = [B, T, n_embed]
        out = torch.cat(head_outputs, dim=-1)

        # Project back to n_embed via Wo
        out = self.dropout(self.proj(out))
    
        if return_weights:
            return out, head_weights
        return out
