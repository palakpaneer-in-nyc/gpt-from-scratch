import os
import torch
import torch.nn as nn
from model import GPT
from tokenizer import CharTokenizer

# Always find shakespeare.txt relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------ Config --------------------------------

config = {
    # Run identity - keeps separate checkpoints from different tuning runs
    'run_name'  :   'default',

    # Data
    'data_path' :   'data/shakespeare.txt',
    'ckpt_dir'  :   'checkpoints',

    # Model
    'vocab_size'    :   65,
    'block_size'    :   256,
    'n_embed'       :   384,
    'n_heads'       :   6,
    'n_layers'      :   6,
    'dropout'       :   0.1,

    # Training
    'batch_size'    :   64,
    'max_steps'     :   5000,
    'eval_every'    :   250, # eval on val set every N steps
    'lr'            :   3e-4,
    'grad_clip'     :   1.0,

    # Split
    'train_split'   :   0.9, # 90% train, 10% val
}

# ------ Device -----------------------------------

device = (
    'cuda' if torch.cuda.is_available() else
    'mps'  if torch.backends.mps.is_available() else
    'cpu'
)
print(f"using device: {device}")

# ------ Data --------------------------------------

def load_data(config):
    data_path = os.path.join(BASE_DIR, 'data', 'shakespeare.txt')
    text = open(data_path, 'r').read()
    tok = CharTokenizer(text)
    config['vocab_size'] = tok.vocab_size

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(len(data) * config['train_split'])
    return data[:n], data[n:], tok

def get_batch(data, config):
    "Sample a random batch of (x, y) pairs."
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([data[i     : i+config['block_size']] for i in ix])
    y = torch.stack([data[i+1   : i+config['block_size']+1] for i in ix])
    return x.to(device), y.to(device)

# ------ Evaluation -----------------------------------

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, eval_batches=20):
    """
    Estimate loss on train and val sets.
    Averahe over eval batches from stability.
    """
    model.eval()
    losses = {}
    for split, data in [('train', train_data), ('val', val_data)]:
        split_losses = []
        for _ in range(eval_batches):
            x, y = get_batch(data, config)
            _, loss = model(x, y)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


# ------ Training loop -----------------------------------

def train(config):
    # Load data
    train_data, test_data, tok = load_data(config)
    print(f"Train tokens: {len(train_data):,} Val tokens: {len(test_data):,}")

    # Build model
    model = GPT(
        vocab_size  = config['vocab_size'],
        block_size  = config['block_size'],
        n_embed     = config['n_embed'],
        n_heads     = config['n_heads'],
        n_layers    = config['n_layers'],
        dropout     = config['dropout'],
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = config['lr'],
    )

    # Checkpoint directory
    ckpt_dir = os.path.join(BASE_DIR, config['ckpt_dir'])
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')

    for step in range(config['max_steps']):
        
        # Evaluate periodically
        if step % config['eval_every'] == 0:
            losses = estimate_loss(model, train_data, test_data, config)
            print(f"step {step:5d} | "
                  f"train loss: {losses['train']:.4f} | "
                  f"val loss: {losses['val']:.4f}")
            
        # Save best checkpoint
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ckpt_path = os.path.join(ckpt_dir, f"{config['run_name']}_best_model.pt")
            torch.save({
                'step'      : step,
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'config'    : config,
                'val_loss'  : best_val_loss,    
            }, ckpt_path)
            print(f" -> saved checkpoint (val loss {best_val_loss:.4f})")

        # Forward pass
        x, y = get_batch(train_data, config)
        logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping - prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        # weight update
        optimizer.step()
    
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    return model, tok
          
# ----- Generate sample text -----------------------------------

def generate_sample(model, tok, prompt="To be", max_tokens=200, temperature=0.8):
    model.eval()
    context = torch.tensor(
        tok.encode(prompt),
        dtype=torch.long
    ).unsqueeze(0).to(device) # [1, T]

    with torch.no_grad():
        output = model.generate(
            context,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
        )

    generated = tok.decode(output[0].tolist())
    return generated

# ----- Entry point -------------------------------------------------
if __name__ == '__main__':
    model, tok = train(config)

    print("\n ---- Generated text after training ----")
    print(generate_sample(model, tok))