"""Spawn-safe CPU tokenization workers; deliberately no FAISS import here.

PyTorch initializes DataLoader worker processes before unpickling the collator.
Keeping the collator here avoids loading FAISS after torch in those processes.
"""
import random
import numpy as np
import torch


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


class Tokenize:
    def __init__(self, tokenizer, max_length):
        self.tokenizer, self.max_length = tokenizer, max_length

    def __call__(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")
