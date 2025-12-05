#!/usr/bin/env python
"""
Simple solution: Materialize all transforms to lists upfront.
This eliminates on-the-fly transformation overhead and caching issues.
"""

import sys
import gc
sys.path.insert(0, '/home/user/fastai')

from fastai.basics import *
from fastai.callback.all import *
from fastai.text.all import *

print("=" * 60)
print("Testing Materialized Dataset Approach")
print("=" * 60)

# 1. Download Wikipedia Tiny Dataset
print("\n1. Downloading Wikipedia Tiny dataset...")
path = untar_data(URLs.WIKITEXT_TINY)

# 2. Load data
print("\n2. Loading data...")
df_train = pd.read_csv(path/'train.csv', header=None)
df_valid = pd.read_csv(path/'test.csv', header=None)
df_all = pd.concat([df_train, df_valid])
print(f"   Train samples: {len(df_train)}, Valid samples: {len(df_valid)}")

# 3. Create standard transforms and dataset
print("\n3. Creating transforms and dataset...")
splits = [list(range_of(df_train)), list(range(len(df_train), len(df_all)))]
tok = Tokenizer.from_df(0)
num = Numericalize()
tfms = [attrgetter("text"), tok, num]
dsets = Datasets(df_all, [tfms], splits=splits)
vocab = num.vocab
print(f"   Vocabulary size: {len(vocab)}")


# 4. Materialize the dataset - convert all items to plain lists
print("\n4. Materializing train dataset (applying all transforms once)...")

def materialize_dataset(dsets, split_idx):
    """
    Materialize a dataset split by applying all transforms and storing results.
    Returns a simple list of tensors.
    """
    split_dset = dsets.subset(split_idx)
    materialized = []

    for i in progress_bar(range(len(split_dset))):
        item = split_dset[i]
        # Get the tensor - handle tuple or single item
        tokens = item[0] if isinstance(item, tuple) else item
        # Convert to plain tensor (detached, on CPU)
        if hasattr(tokens, 'clone'):
            tokens = tokens.clone().detach()
        materialized.append(tokens)

    return materialized

train_items = materialize_dataset(dsets, 0)
print(f"   Materialized {len(train_items)} train documents")

print("\n5. Materializing valid dataset...")
valid_items = materialize_dataset(dsets, 1)
print(f"   Materialized {len(valid_items)} valid documents")

# Check total tokens
train_tokens = sum(len(t) for t in train_items)
valid_tokens = sum(len(t) for t in valid_items)
print(f"\n   Train tokens: {train_tokens:,}")
print(f"   Valid tokens: {valid_tokens:,}")


# 5. Create a simple wrapper dataset that just returns pre-computed items
class MaterializedLMDataset:
    """
    A simple dataset wrapper for materialized (pre-computed) items.
    Works with LMDataLoader - just returns the pre-computed tensors.
    """
    def __init__(self, items, vocab=None):
        self.items = items
        self.vocab = vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return (self.items[idx],)  # Return as tuple like Datasets does


# 6. Create the materialized datasets
print("\n6. Creating materialized dataset wrappers...")
train_mat = MaterializedLMDataset(train_items, vocab)
valid_mat = MaterializedLMDataset(valid_items, vocab)


# 7. Create LMDataLoader with the materialized data
print("\n7. Creating LMDataLoaders with materialized data...")
bs, sl = 64, 72

# Pre-compute lengths to avoid recomputation
train_lens = [len(t) for t in train_items]
valid_lens = [len(t) for t in valid_items]

train_dl = LMDataLoader(train_mat, lens=train_lens, bs=bs, seq_len=sl, num_workers=0, shuffle=True)
valid_dl = LMDataLoader(valid_mat, lens=valid_lens, bs=bs, seq_len=sl, num_workers=0, shuffle=False)

print(f"   Train batches: {len(train_dl)}")
print(f"   Valid batches: {len(valid_dl)}")


# 8. Test batch retrieval
print("\n8. Testing batch retrieval...")
x, y = train_dl.one_batch()
print(f"   Input shape: {x.shape}")
print(f"   Target shape: {y.shape}")

# Decode sample
sample_text = ' '.join([vocab[t] for t in x[0][:15].tolist()])
print(f"   Sample text: {sample_text}...")


# 9. Test memory stability across epochs
print("\n9. Testing memory stability across epochs...")
import psutil
import os

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

gc.collect()
mem_start = get_memory_mb()
print(f"   Initial memory: {mem_start:.1f} MB")

for epoch in range(5):
    batch_count = 0
    for x, y in train_dl:
        batch_count += 1
        if batch_count >= 20:  # Test more batches
            break
        del x, y

    gc.collect()
    mem_now = get_memory_mb()
    print(f"   Epoch {epoch+1}: Memory = {mem_now:.1f} MB (delta: {mem_now - mem_start:+.1f} MB)")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)

print("""
SUMMARY:
--------
This approach materializes all transforms upfront:

1. Apply all transforms (tokenization, numericalization) ONCE
2. Store results as plain tensors in a list
3. Use standard LMDataLoader with the pre-computed data
4. Pass pre-computed lengths to avoid recomputation

This eliminates:
- On-the-fly transform overhead
- Transform caching issues
- ReindexCollection complexity (items are just plain tensors)

The LMDataLoader still handles:
- Document-level shuffling (correct behavior)
- Chunking into bs streams
- Proper sequence extraction
""")
