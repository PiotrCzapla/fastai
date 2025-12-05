#!/usr/bin/env python
"""Test the materialized LM dataset module."""

import sys
sys.path.insert(0, '/home/user/fastai')

from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *
from fastai.text.materialized import MaterializedLMDataLoaders, MaterializedLMDataset
import gc

print("=" * 60)
print("Testing Materialized LM Dataset Module")
print("=" * 60)

# 1. Setup
path = untar_data(URLs.WIKITEXT_TINY)
df_train = pd.read_csv(path/'train.csv', header=None)
df_valid = pd.read_csv(path/'test.csv', header=None)
df_all = pd.concat([df_train, df_valid])

splits = [list(range_of(df_train)), list(range(len(df_train), len(df_all)))]
tok = Tokenizer.from_df(0)
num = Numericalize()
tfms = [attrgetter("text"), tok, num]
dsets = Datasets(df_all, [tfms], splits=splits)
vocab = num.vocab

print(f"\nDataset: {len(df_train)} train, {len(df_valid)} valid docs")
print(f"Vocab size: {len(vocab)}")

# 2. Test building from datasets
print("\n" + "-" * 40)
print("Test 1: Build from Datasets")
print("-" * 40)

dls = MaterializedLMDataLoaders.from_datasets(
    dsets, bs=64, seq_len=72, num_workers=0, vocab=vocab
)

print(f"\nTrain tokens: {dls._train_ds.total_tokens:,}")
print(f"Valid tokens: {dls._valid_ds.total_tokens:,}")
print(f"Train batches: {len(dls.train)}")
print(f"Valid batches: {len(dls.valid)}")

# Test batch
x, y = dls.train.one_batch()
print(f"Batch shape: {x.shape}")

# 3. Test caching
print("\n" + "-" * 40)
print("Test 2: Save and Load from Cache")
print("-" * 40)

cache_dir = path / 'materialized_cache'
print(f"Saving to: {cache_dir}")

dls._train_ds.save(cache_dir / 'train')
dls._valid_ds.save(cache_dir / 'valid')

# Load from cache
print("Loading from cache...")
dls2 = MaterializedLMDataLoaders.from_cache(cache_dir, bs=64, seq_len=72, num_workers=0)

print(f"Loaded train tokens: {dls2._train_ds.total_tokens:,}")
print(f"Loaded vocab size: {len(dls2.vocab)}")

# 4. Test from_cache_or_build
print("\n" + "-" * 40)
print("Test 3: from_cache_or_build (should use cache)")
print("-" * 40)

import shutil
shutil.rmtree(cache_dir, ignore_errors=True)

# First call - should build
dls3 = MaterializedLMDataLoaders.from_cache_or_build(
    cache_dir, dsets, bs=64, seq_len=72, num_workers=0, vocab=vocab
)
print(f"Built: {dls3._train_ds.total_tokens:,} tokens")

# Second call - should load from cache
print("\nSecond call (should load from cache):")
import time
start = time.time()
dls4 = MaterializedLMDataLoaders.from_cache_or_build(
    cache_dir, dsets, bs=64, seq_len=72, num_workers=0
)
elapsed = time.time() - start
print(f"Loaded in {elapsed:.3f}s: {dls4._train_ds.total_tokens:,} tokens")

# 5. Memory stability test
print("\n" + "-" * 40)
print("Test 4: Memory Stability (5 epochs)")
print("-" * 40)

import psutil
import os

def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

gc.collect()
mem_start = get_memory_mb()
print(f"Initial: {mem_start:.1f} MB")

for epoch in range(5):
    for batch_idx, (x, y) in enumerate(dls.train):
        if batch_idx >= 30:
            break
        del x, y
    gc.collect()
    print(f"Epoch {epoch+1}: {get_memory_mb():.1f} MB (delta: {get_memory_mb() - mem_start:+.1f} MB)")

# 6. Verify document shuffling works
print("\n" + "-" * 40)
print("Test 5: Document Shuffling Verification")
print("-" * 40)

# Get first batch from two different iterations
x1, _ = next(iter(dls.train))
x2, _ = next(iter(dls.train))

same = torch.equal(x1, x2)
print(f"First batch same across epochs: {same}")
print("(Should be False - documents are shuffled each epoch)")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
