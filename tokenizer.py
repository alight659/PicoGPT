import json
import base64

# Byte-Pair Encoding Tokenizer
class BPETokenizer:
    def __init__(self, vocab_size=276):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def tokenize_text(self, text):
        tokens = text.encode('utf-8')
        tokens = list(map(int, tokens))
        return tokens
    
    def count_occurences(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def fit(self, tokens):
        ids = list(tokens)
        for i in range(self.num_merges):
            stats = self.count_occurences(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"Merging {pair} into {idx}")
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

    def decode(self, ids):
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text

    def encode(self, text):
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.count_occurences(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break                                           # nothing can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
            print(len(tokens))
        return tokens
    
    def save(self, filepath):
        merges_json = {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
        vocab_json = {k: base64.b64encode(v).decode('utf-8') for k, v in self.vocab.items()}
        data = {
            'vocab_size': self.vocab_size,
            'merges': merges_json,
            'vocab': vocab_json
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)


    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        bpe = cls(vocab_size=data['vocab_size'])
        bpe.merges = {tuple(map(int, k.split())): v for k, v in data['merges'].items()}
        bpe.vocab = {int(k): base64.b64decode(v) for k, v in data['vocab'].items()}
        return bpe