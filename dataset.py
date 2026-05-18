"""
dataset.py — Multi30k Dataset Loading and Preprocessing
DA6401 Assignment 3: "Attention Is All You Need"
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
from datasets import load_dataset


class Vocabulary:
    def __init__(self, specials=["<unk>", "<pad>", "<sos>", "<eos>"]):
        self.specials = specials
        self.itos = {}
        self.stoi = {}
        for i, tok in enumerate(specials):
            self.itos[i] = tok
            self.stoi[tok] = i

    def build_from_counter(self, counter, min_freq=1):
        idx = len(self.itos)
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def __len__(self):
        return len(self.itos)

    def lookup_token(self, idx):
        return self.itos.get(idx, "<unk>")

    def lookup_indices(self, tokens):
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]


class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        dataset = load_dataset("bentrevett/multi30k")
        self.data = dataset[split]

        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        if self.src_vocab is None or self.tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()

        self.processed = self.process_data()

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        src_counter = Counter()
        tgt_counter = Counter()

        full_dataset = load_dataset("bentrevett/multi30k")

        for split_name in ["train", "validation", "test"]:
            for item in full_dataset[split_name]:
                src_counter.update(self.tokenize_de(item["de"]))
                tgt_counter.update(self.tokenize_en(item["en"]))

        src_vocab = Vocabulary()
        src_vocab.build_from_counter(src_counter, min_freq=1)

        tgt_vocab = Vocabulary()
        tgt_vocab.build_from_counter(tgt_counter, min_freq=1)

        return src_vocab, tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        # TODO: Tokenize and convert words to indices
        processed = []
        pad_idx = self.src_vocab.stoi["<pad>"]
        sos_idx_src = self.src_vocab.stoi["<sos>"]
        eos_idx_src = self.src_vocab.stoi["<eos>"]
        sos_idx_tgt = self.tgt_vocab.stoi["<sos>"]
        eos_idx_tgt = self.tgt_vocab.stoi["<eos>"]

        for item in self.data:
            src_tokens = self.tokenize_de(item["de"])
            tgt_tokens = self.tokenize_en(item["en"])

            src_indices = [sos_idx_src] + self.src_vocab.lookup_indices(src_tokens) + [eos_idx_src]
            tgt_indices = [sos_idx_tgt] + self.tgt_vocab.lookup_indices(tgt_tokens) + [eos_idx_tgt]

            processed.append((src_indices, tgt_indices))

        return processed

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        src, tgt = self.processed[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


def collate_fn(batch, src_pad_idx=1, tgt_pad_idx=1):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded


def get_dataloaders(batch_size=128):
    train_dataset = Multi30kDataset(split="train")
    src_vocab = train_dataset.src_vocab
    tgt_vocab = train_dataset.tgt_vocab

    val_dataset = Multi30kDataset(split="validation", src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_dataset = Multi30kDataset(split="test", src_vocab=src_vocab, tgt_vocab=tgt_vocab)

    pad_idx = src_vocab.stoi["<pad>"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_idx, pad_idx),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx, pad_idx),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_idx, pad_idx),
    )

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
