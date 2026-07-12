import os
import torch
import torch.nn as nn
from model import GPT
from tokenizer import CharTokenizer

# TINY config - fast to test
VOCAB_SIZE = 65
BLOCK_SIZE = 32
N_EMBED = 64
N_HEADS = 4
N_LAYERS = 3
BATCH_SIZE = 4

# Always find shakespeare.txt relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'shakespeare.txt')

def get_model():
    return GPT(VOCAB_SIZE, BLOCK_SIZE, N_EMBED, N_HEADS, N_LAYERS)

def test_output_shape():
    "Logits must be [B, T, VOCAB_SIZE]"
    gpt = get_model()
    # Generate random integers [0, VOCAB_SIZE) in the shape of [B, T]
    idx = torch.randint(0,VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    logits, loss = gpt(idx)
    assert logits.shape == (BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE), \
        f"Expected logits.shape {(BATCH_SIZE, BLOCK_SIZE, VOCAB_SIZE)}, but got {logits.shape}"
    
    assert loss is None, "Loss should be NOne when no targets provided"
    print(f"PASS: Logits shape is correct {logits.shape}")

def test_loss_computed_with_targets():
    gpt = get_model()
    idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    logits, loss = gpt(idx, targets)
    assert loss is not None, "Loss should not be none with targets"
    assert loss.shape == torch.Size([]), "loss must be scaler"
    print(f"PASS: loss computed correct : {loss.item():.4f}")

def test_initial_loss_near_log_vocab_size():
    gpt = get_model()
    idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    _, loss = gpt(idx, targets)
    expected = torch.log(torch.tensor(VOCAB_SIZE, dtype=torch.float))
    assert abs(loss.item() - expected.item()) < 1.0, \
        f"Initial loss {loss.item():.2f} far from expected {expected.item():.2f}"
    print(f"PASS: Inital loss {loss.item():.2f} ~= log({VOCAB_SIZE}) = {expected.item():.2f}")

def test_overfit_on_single_batch():
    """
    Most critical test - model must be able to overfit, if given
    same input & output for 100s steps. If loss doesn't reduce near zero,
    something is broken. This proves model can learn - gradient flow works
    end to end.
    """
    torch.manual_seed(42)
    gpt = get_model()
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)
    idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))

    initial_loss = None
    for i in range(200):
        logits, loss = gpt(idx, targets)
        # clean-up all grads from before, to avoid contamination.
        # zeros all weight.grad() in gpt parameters.
        optimizer.zero_grad()
        loss.backward()     # compute gradients
        optimizer.step()    # Update weights
        if initial_loss is None:
            initial_loss = loss.item()
    final_loss = loss.item()
    print(f"Inital loss: {initial_loss:.2f}")
    print(f"Final loss : {final_loss:.2f}")

    assert final_loss < initial_loss * 0.1, \
        f"Model didn't overfit, loss whent from {initial_loss} -> {final_loss} only"
    print(f"PASS: Model overfits single batch - end to end learning works")
        

def test_generate_shape_and_tokens_in_vocab(verbose=False):
    gpt = get_model()
    gpt.eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = gpt.generate(idx, max_new_tokens=50)
    generated = out[0, 1:] # skip the seed token
    if verbose:
        print(f"\tgenerate shape: {generated.shape}, tokens: {generated.tolist()}")

    assert generated.min().item() >= 0, \
        "generated token is less than 0"
    assert generated.max().item() < VOCAB_SIZE, \
        "generated token is larger than vocab size"
    if verbose:
        text = open(DATA_PATH, 'r').read()
        char_tokenizer = CharTokenizer(text)
        generated_tokens = [idx for idx in generated.tolist()]
        generated_text = char_tokenizer.decode(generated_tokens)
        print(f"\tgenerated text: {generated_text}")
    print("PASS: genearted tokens are within bounds of Vocab size")


def test_weight_tying():
    """
    lm_head and token embedding must share the same weight matrix.
    """
    gpt = get_model()
    emb_weight = gpt.transformer['embedding'].token_embed.embedding.weight
    lm_head = gpt.lm_head.weight

    assert emb_weight.data_ptr() == lm_head.data_ptr(), \
        f"weight tying broken - token embedding and lm_head are not sharing same weight matrix"
    print("PASS: shared weight matrix between token embed & lm head")

def test_gradients_flow_end_to_end():
    """Gradients must reacj the embedding table - full pipeline works."""
    gpt = get_model()
    idx = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))
    targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, BLOCK_SIZE))

    _, loss = gpt(idx, targets)
    before_token_embed_grad = gpt.transformer['embedding'].token_embed.embedding.weight.grad
    assert before_token_embed_grad is None, \
        f"Token embedding grad should be empty ideally, before back proagation."
    loss.backward()
    after_token_embed_grad = gpt.transformer['embedding'].token_embed.embedding.weight.grad
    assert after_token_embed_grad is not None, \
        f"Token embedding grad is empty after backpropagation - pipeline broken"
    print(f"PASS: gradient flow end to end - embedding table received the grad.")

if __name__ == '__main__':
    torch.manual_seed(42)
    test_output_shape()
    test_loss_computed_with_targets()
    test_initial_loss_near_log_vocab_size()
    test_overfit_on_single_batch()
    test_generate_shape_and_tokens_in_vocab()
    test_weight_tying()
    test_gradients_flow_end_to_end()