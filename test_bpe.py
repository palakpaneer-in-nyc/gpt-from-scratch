import os
from tokenizer import BPETokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'shakespeare.txt')

def test_roundtrip():
    tok = BPETokenizer()
    tok.train("hello world hello world hello", vocab_size=260, verbose=False)
    sample_text = "hello world"
    assert tok.decode(tok.encode(sample_text)) == sample_text
    print("PASS: roundtrip")


def test_compression():
    text = open(DATA_PATH).read()[:10000] # read first 10K chars
    tok = BPETokenizer()
    tok.train(text, vocab_size=275, verbose=False)

    char_tokens = len(list(text.encode('utf-8')))
    bpe_tokens = len(tok.encode(text))
    ratio = char_tokens / bpe_tokens

    print(f"PASS: compression -> {char_tokens} bytes -> {bpe_tokens} tokens "
          f"(ratio {ratio:.2f}x)")
    assert ratio > 1.0, "BPE should compress"


def test_no_unknown_tokens():
    """BPE should handle any unicode text - no unknown ever."""
    tok = BPETokenizer()
    tok.train("hello world", vocab_size=260, verbose=False)
    # Emoji, Chinese, Arabic — all encodable as bytes
    weird = "Hello 🌍 世界 مرحبا"
    assert tok.decode(tok.encode(weird)) == weird
    print("PASS: No unknown tokens -> handles any unicode")


if __name__ == "__main__":
    test_roundtrip()
    test_compression()
    test_no_unknown_tokens()