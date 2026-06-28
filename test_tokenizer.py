from tokenizer import CharTokenizer

def test_roundtrip():
    text = "hello world"
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text
    print("PASS: roundtrip")


def test_vocab_coverage():
    text = open('data/shakespeare.txt', 'r').read()
    tok = CharTokenizer(text)
    for c in set(text):
        assert c in tok.stoi, f"'{c}' not in vocab"
    print(f"PASS: vocab coverage - {tok.vocab_size} chars")


def test_empty():
    tok = CharTokenizer("abc")
    assert tok.encode("") == []
    print("PASS: empty string")


def test_consistent():
    tok = CharTokenizer("hello world")
    assert tok.encode("hello") == tok.encode("hello")
    print("PASS: consistent encoding")


if __name__ == '__main__':
    test_roundtrip()
    test_vocab_coverage()
    test_empty()
    test_consistent()
    print("\nAll tokenizer tests passed!")