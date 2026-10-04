import os, urllib.request
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def shakespeare(path="data/shakespeare.txt"):
    """Tiny Shakespeare (1.1 MB), downloaded once. Returns (text, chars, ids) with the 90/10 train/val split done by the caller."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    text = open(path).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    return text, chars, torch.tensor([stoi[c] for c in text])
