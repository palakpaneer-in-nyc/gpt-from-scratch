class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, tokens):
        return ''.join(self.itos[i] for i in tokens)