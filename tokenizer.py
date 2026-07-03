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


class BPETokenizer:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)} # i token index - bytes
        self.merges = {} # (pair) -> new_token_id - merge rule during training

    def _get_pair_counts(self, ids):
        """Count the frequency of every adjacent pair in the token list."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge_pair(self, ids, pair, new_id):
        """Replace every given pair occurance with new_id."""
        out = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out
    
    def train(self, text, vocab_size, verbose=True):
        """
        Learn merge rules from text until vocab reaches vocab_size.
        vocab_size must be > 256 (we start with 256 base byte tokens).
        """

        assert vocab_size > 256, "vocab_size must be > 256"
        num_merges = vocab_size - 256

        # Step1: convert / encode text to bytes char tokens?
        ids = list(text.encode('utf-8'))
        if verbose:
            print(f"Training BPE on {len(ids)} bytes")
            print(f"Targeting vocab size: {vocab_size} ({num_merges} merges to learn)")

        i = 0 
        for i in range(num_merges):
            # Step2: Pair up most frequent adjacent pairs.
            pair_counts = self._get_pair_counts(ids)
            if not pair_counts:
                break

            # Step3: Find the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i

            # Step4: Merge the most frequent pair everyhwere
            ids = self._merge_pair(ids, best_pair, new_id)

            # Register the new merge
            self.merges[best_pair] = new_id
            self.vocab[new_id] = (self.vocab[best_pair[0]] + self.vocab[best_pair[1]])

            if verbose:
                merged_str = self.vocab[new_id].decode('utf-8', errors='replace')
                print(f"Merged {i+1:3d}/{num_merges}: "
                      f"{best_pair} -> {new_id} "
                      f"| '{merged_str} "
                      f"| freq={pair_counts[best_pair]}")
                
    def encode(self, text):
        # Start with raw bytes
        ids = list(text.encode('utf-8'))

        while len(ids) >= 2:
            pair_counts = self._get_pair_counts(ids)

            best_pair = min(
                (p for p in pair_counts if p in self.merges),
                key=lambda p: self.merges[p],
                default=None
            )

            if best_pair is None:
                break # no more merges to apply
            
            ids = self._merge_pair(ids, best_pair, self.merges[best_pair])
        return ids

    def decode(self, ids):
        """Decode ids back to text."""
        raw_bytes = b''.join(self.vocab[id] for id in ids)
        return raw_bytes.decode('utf-8', errors='replace')

