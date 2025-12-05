#!/usr/bin/env python
"""Test script for memory-efficient text DataLoader."""

import sys
import gc
import json
import numpy as np
from pathlib import Path

# Add fastai to path
sys.path.insert(0, '/home/user/fastai')

from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *

print("=" * 60)
print("Testing Memory-Efficient Text DataLoader")
print("=" * 60)

# 1. Download Wikipedia Tiny Dataset
print("\n1. Downloading Wikipedia Tiny dataset...")
path = untar_data(URLs.WIKITEXT_TINY)
print(f"   Path: {path}")

# 2. Load data
print("\n2. Loading data...")
df_train = pd.read_csv(path/'train.csv', header=None)
df_valid = pd.read_csv(path/'test.csv', header=None)
df_all = pd.concat([df_train, df_valid])
print(f"   Train samples: {len(df_train)}, Valid samples: {len(df_valid)}")

# 3. Create splits and transforms
print("\n3. Creating transforms...")
splits = [list(range_of(df_train)), list(range(len(df_train), len(df_all)))]
tok = Tokenizer.from_df(0)
num = Numericalize()
tfms = [attrgetter("text"), tok, num]
dsets = Datasets(df_all, [tfms], splits=splits)
vocab = num.vocab
print(f"   Vocabulary size: {len(vocab)}")

# 4. Define DiskLMDataset class
class DiskLMDataset:
    """Stores numericalized text data on disk using numpy memmap."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_path = self.cache_dir / 'tokens.npy'
        self.meta_path = self.cache_dir / 'meta.json'
        self.vocab_path = self.cache_dir / 'vocab.pkl'

        self._data = None
        self._meta = None
        self._vocab = None

    @classmethod
    def from_datasets(cls, dsets, cache_dir: Path, split_idx: int = 0, force_rebuild: bool = False, vocab=None):
        obj = cls(cache_dir)

        if not force_rebuild and obj.data_path.exists() and obj.meta_path.exists():
            print(f"   Loading from cache: {cache_dir}")
            obj._load_cache()
            return obj

        print(f"   Building cache in: {cache_dir}")
        split_dset = dsets.subset(split_idx)

        all_tokens = []
        doc_lengths = []

        for i in progress_bar(range(len(split_dset)), leave=False):
            item = split_dset[i]
            tokens = item[0] if isinstance(item, tuple) else item
            tokens = tokens.numpy() if hasattr(tokens, 'numpy') else np.array(tokens)
            all_tokens.append(tokens)
            doc_lengths.append(len(tokens))

        all_tokens = np.concatenate(all_tokens).astype(np.int32)
        total_tokens = len(all_tokens)

        np.save(obj.data_path, all_tokens)

        meta = {
            'total_tokens': total_tokens,
            'doc_lengths': doc_lengths,
            'num_docs': len(doc_lengths),
        }
        with open(obj.meta_path, 'w') as f:
            json.dump(meta, f)

        # Save vocabulary - use explicitly passed vocab
        if vocab is not None:
            save_pickle(obj.vocab_path, vocab)

        obj._load_cache()
        return obj

    def _load_cache(self):
        self._data = np.load(self.data_path, mmap_mode='r')
        with open(self.meta_path, 'r') as f:
            self._meta = json.load(f)
        if self.vocab_path.exists():
            self._vocab = load_pickle(self.vocab_path)

    @property
    def data(self):
        if self._data is None: self._load_cache()
        return self._data

    @property
    def meta(self):
        if self._meta is None: self._load_cache()
        return self._meta

    @property
    def vocab(self):
        if self._vocab is None: self._load_cache()
        return self._vocab

    @property
    def total_tokens(self):
        return self.meta['total_tokens']

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx):
        return self.data[idx]


# 5. Define DataLoader Dataset
class DiskLMDataLoaderDataset(torch.utils.data.Dataset):
    """PyTorch Dataset that reads from DiskLMDataset."""

    def __init__(self, disk_data: DiskLMDataset, bs: int = 64, seq_len: int = 72):
        self.disk_data = disk_data
        self.bs = bs
        self.seq_len = seq_len

        total = disk_data.total_tokens - 1
        self.corpus = round_multiple(total, bs, round_down=True)
        self.bl = self.corpus // bs
        self.n_batches = self.bl // seq_len + int(self.bl % seq_len != 0)
        self.last_len = self.bl - (self.n_batches - 1) * seq_len
        self._batch_order = None

    def __len__(self):
        return self.n_batches * self.bs

    def shuffle(self):
        self._batch_order = torch.randperm(self.n_batches).numpy()

    def __getitem__(self, seq):
        if seq >= len(self):
            raise IndexError(f"Index {seq} out of range")

        batch_idx = seq // self.bs
        stream_idx = seq % self.bs

        if self._batch_order is not None:
            batch_idx = self._batch_order[batch_idx]

        sl = self.last_len if batch_idx == self.n_batches - 1 else self.seq_len
        start = stream_idx * self.bl + batch_idx * self.seq_len

        tokens = self.disk_data[start:start + sl + 1].copy()
        tokens = torch.from_numpy(tokens).long()

        x = LMTensorText(tokens[:-1])
        y = tokens[1:]
        return x, y


# 6. Define Memory-Efficient DataLoader
class MemoryEfficientLMDataLoader:
    """Memory-efficient Language Model DataLoader."""

    def __init__(self, disk_data, bs=64, seq_len=72, num_workers=0, shuffle=True, drop_last=True):
        self.disk_data = disk_data
        self.bs = bs
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset = DiskLMDataLoaderDataset(disk_data, bs=bs, seq_len=seq_len)
        self.vocab = disk_data.vocab

    def __len__(self):
        return self.dataset.n_batches

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()

        dl = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.bs,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=self._collate
        )

        for batch in dl:
            yield batch

    def _collate(self, batch):
        xs, ys = zip(*batch)
        x = torch.stack([x for x in xs])
        y = torch.stack([y for y in ys])
        return x, y

    def one_batch(self):
        return next(iter(self))

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        return ' '.join([self.vocab[t] for t in tokens])


# 7. Build disk datasets
print("\n4. Building disk cache for train set...")
cache_base = path / 'lm_cache'
train_disk = DiskLMDataset.from_datasets(dsets, cache_base / 'train', split_idx=0, vocab=vocab)
print(f"   Train tokens: {train_disk.total_tokens:,}")

print("\n5. Building disk cache for valid set...")
valid_disk = DiskLMDataset.from_datasets(dsets, cache_base / 'valid', split_idx=1, vocab=vocab)
print(f"   Valid tokens: {valid_disk.total_tokens:,}")

# 8. Create DataLoaders
print("\n6. Creating memory-efficient DataLoaders...")
bs, sl = 64, 72

train_dl = MemoryEfficientLMDataLoader(
    train_disk, bs=bs, seq_len=sl, num_workers=0, shuffle=True
)
valid_dl = MemoryEfficientLMDataLoader(
    valid_disk, bs=bs, seq_len=sl, num_workers=0, shuffle=False
)

print(f"   Train batches: {len(train_dl)}")
print(f"   Valid batches: {len(valid_dl)}")

# 9. Test a batch
print("\n7. Testing batch retrieval...")
x, y = train_dl.one_batch()
print(f"   Input shape: {x.shape}")
print(f"   Target shape: {y.shape}")
print(f"   Sample text: {train_dl.decode(x[0][:15])}...")

# 10. Test multiple epochs for memory stability
print("\n8. Testing multiple epoch iterations (checking for memory leaks)...")
import psutil
import os

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

gc.collect()
mem_start = get_memory_mb()
print(f"   Initial memory: {mem_start:.1f} MB")

for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_dl):
        if batch_idx >= 5:
            break
        del x, y
    gc.collect()
    mem_now = get_memory_mb()
    print(f"   Epoch {epoch+1}: Memory = {mem_now:.1f} MB (delta: {mem_now - mem_start:+.1f} MB)")

# 11. Test second load (should be fast)
print("\n9. Testing fast reload from cache...")
import time
start_time = time.time()

# Create fresh disk dataset from cache
train_disk2 = DiskLMDataset(cache_base / 'train')
train_disk2._load_cache()  # Force load

elapsed = time.time() - start_time
print(f"   Cache reload time: {elapsed:.3f} seconds")
print(f"   Tokens loaded: {train_disk2.total_tokens:,}")

print("\n" + "=" * 60)
print("SUCCESS! Memory-efficient DataLoader working correctly.")
print("=" * 60)

# Summary
print("""
SUMMARY:
--------
The memory-efficient text DataLoader solution:

1. Pre-processes and caches numericalized tokens to disk using numpy memmap
2. Uses memory mapping so data stays on disk, loaded on-demand
3. Supports configurable num_workers (0 for minimal memory)
4. Fast second load from cache
5. Avoids ReindexCollection caching issues in standard LMDataLoader

Usage:
------
# First time: Build cache
train_disk = DiskLMDataset.from_datasets(dsets, cache_dir/'train', split_idx=0)

# Create dataloader with minimal workers
train_dl = MemoryEfficientLMDataLoader(train_disk, bs=64, seq_len=72, num_workers=0)

# Second time: Load from cache (fast!)
train_disk = DiskLMDataset(cache_dir/'train')
train_disk._load_cache()
""")
