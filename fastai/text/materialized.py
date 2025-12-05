"""
Memory-efficient text data loading via materialization.

The key insight: apply all transforms ONCE upfront and store as plain tensors.
This eliminates on-the-fly transform overhead and caching issues while
preserving the standard LMDataLoader document shuffling behavior.
"""

from __future__ import annotations
from ..torch_basics import *
from ..data.all import *
from .core import *
from .data import LMDataLoader, Numericalize
import numpy as np
import json

__all__ = ['materialize_lm_dataset', 'MaterializedLMDataset', 'MaterializedLMDataLoaders']


def materialize_lm_dataset(dsets, split_idx: int, show_progress: bool = True):
    """
    Materialize a dataset split by applying all transforms once.

    Args:
        dsets: fastai Datasets object with text transforms
        split_idx: Which split to materialize (0=train, 1=valid)
        show_progress: Show progress bar

    Returns:
        Tuple of (list of tensors, list of lengths)
    """
    split_dset = dsets.subset(split_idx)
    items = []
    lens = []

    iterator = range(len(split_dset))
    if show_progress:
        iterator = progress_bar(iterator)

    for i in iterator:
        item = split_dset[i]
        # Handle tuple (x,) or single item
        tokens = item[0] if isinstance(item, tuple) else item
        # Ensure it's a detached tensor
        if hasattr(tokens, 'clone'):
            tokens = tokens.clone().detach()
        items.append(tokens)
        lens.append(len(tokens))

    return items, lens


class MaterializedLMDataset:
    """
    A dataset wrapper for materialized (pre-computed) text items.

    Works directly with LMDataLoader - just returns pre-computed tensors.
    Optionally saves/loads from disk cache.
    """

    def __init__(self, items: list, lens: list = None, vocab=None):
        """
        Args:
            items: List of numericalized tensors
            lens: List of lengths (computed if not provided)
            vocab: Vocabulary list for decoding
        """
        self.items = items
        self.lens = lens if lens is not None else [len(t) for t in items]
        self.vocab = vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return (self.items[idx],)

    @property
    def total_tokens(self):
        return sum(self.lens)

    @classmethod
    def from_datasets(cls, dsets, split_idx: int, vocab=None, show_progress: bool = True):
        """
        Create from a fastai Datasets by materializing transforms.

        Args:
            dsets: fastai Datasets with text transforms
            split_idx: Which split (0=train, 1=valid)
            vocab: Vocabulary (auto-detected from Numericalize if None)
            show_progress: Show progress bar
        """
        items, lens = materialize_lm_dataset(dsets, split_idx, show_progress)

        # Auto-detect vocab from Numericalize transform if not provided
        if vocab is None:
            tfms = dsets.tfms[0] if hasattr(dsets.tfms[0], 'fs') else [dsets.tfms[0]]
            if hasattr(tfms, 'fs'):
                tfms = tfms.fs
            for tfm in tfms:
                if hasattr(tfm, 'vocab'):
                    vocab = tfm.vocab
                    break

        return cls(items, lens, vocab)

    def save(self, path: Path):
        """Save materialized dataset to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save items as numpy arrays (more compact)
        items_np = [t.numpy() if hasattr(t, 'numpy') else np.array(t) for t in self.items]
        np.savez_compressed(path / 'items.npz', *items_np)

        # Save metadata
        meta = {'lens': self.lens, 'num_items': len(self.items)}
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f)

        # Save vocab
        if self.vocab is not None:
            save_pickle(path / 'vocab.pkl', self.vocab)

    @classmethod
    def load(cls, path: Path):
        """Load materialized dataset from disk."""
        path = Path(path)

        # Load items
        with np.load(path / 'items.npz') as data:
            items = [tensor(data[k]).long() for k in data.files]

        # Load metadata
        with open(path / 'meta.json', 'r') as f:
            meta = json.load(f)

        # Load vocab
        vocab = None
        if (path / 'vocab.pkl').exists():
            vocab = load_pickle(path / 'vocab.pkl')

        return cls(items, meta['lens'], vocab)

    @classmethod
    def from_cache_or_build(cls, cache_path: Path, dsets, split_idx: int,
                            vocab=None, force_rebuild: bool = False):
        """
        Load from cache if available, otherwise build and cache.

        Args:
            cache_path: Path to cache directory
            dsets: fastai Datasets (used if cache doesn't exist)
            split_idx: Which split (0=train, 1=valid)
            vocab: Vocabulary
            force_rebuild: Force rebuild even if cache exists
        """
        cache_path = Path(cache_path)

        if not force_rebuild and (cache_path / 'items.npz').exists():
            print(f"Loading from cache: {cache_path}")
            return cls.load(cache_path)

        print(f"Building and caching to: {cache_path}")
        obj = cls.from_datasets(dsets, split_idx, vocab)
        obj.save(cache_path)
        return obj


class MaterializedLMDataLoaders:
    """
    DataLoaders for materialized language model datasets.

    Uses standard LMDataLoader internally, preserving proper document shuffling.
    """

    def __init__(self, train_ds: MaterializedLMDataset, valid_ds: MaterializedLMDataset,
                 bs: int = 64, seq_len: int = 72, num_workers: int = 0, **kwargs):
        """
        Args:
            train_ds: Materialized training dataset
            valid_ds: Materialized validation dataset
            bs: Batch size
            seq_len: Sequence length
            num_workers: Number of dataloader workers (0 recommended)
            **kwargs: Additional args passed to LMDataLoader
        """
        self._train_ds = train_ds
        self._valid_ds = valid_ds
        self._vocab = train_ds.vocab

        # Create LMDataLoaders with pre-computed lengths
        self._train = LMDataLoader(
            train_ds, lens=train_ds.lens, bs=bs, seq_len=seq_len,
            num_workers=num_workers, shuffle=True, **kwargs
        )
        self._valid = LMDataLoader(
            valid_ds, lens=valid_ds.lens, bs=bs, seq_len=seq_len,
            num_workers=num_workers, shuffle=False, **kwargs
        )

    @property
    def train(self):
        return self._train

    @property
    def valid(self):
        return self._valid

    @property
    def vocab(self):
        return self._vocab

    def __iter__(self):
        return iter([self._train, self._valid])

    def __len__(self):
        return 2

    def __getitem__(self, i):
        return [self._train, self._valid][i]

    @classmethod
    def from_datasets(cls, dsets, bs: int = 64, seq_len: int = 72,
                      num_workers: int = 0, vocab=None, **kwargs):
        """
        Create from fastai Datasets by materializing transforms.

        Args:
            dsets: fastai Datasets with text transforms
            bs: Batch size
            seq_len: Sequence length
            num_workers: Number of workers (0 recommended)
            vocab: Vocabulary (auto-detected if None)
        """
        print("Materializing train dataset...")
        train_ds = MaterializedLMDataset.from_datasets(dsets, 0, vocab)

        print("Materializing valid dataset...")
        valid_ds = MaterializedLMDataset.from_datasets(dsets, 1, train_ds.vocab)

        return cls(train_ds, valid_ds, bs=bs, seq_len=seq_len,
                   num_workers=num_workers, **kwargs)

    @classmethod
    def from_cache(cls, cache_dir: Path, bs: int = 64, seq_len: int = 72,
                   num_workers: int = 0, **kwargs):
        """
        Load from cached materialized datasets.

        Args:
            cache_dir: Directory containing 'train' and 'valid' subdirs
            bs: Batch size
            seq_len: Sequence length
            num_workers: Number of workers (0 recommended)
        """
        cache_dir = Path(cache_dir)
        train_ds = MaterializedLMDataset.load(cache_dir / 'train')
        valid_ds = MaterializedLMDataset.load(cache_dir / 'valid')

        return cls(train_ds, valid_ds, bs=bs, seq_len=seq_len,
                   num_workers=num_workers, **kwargs)

    @classmethod
    def from_cache_or_build(cls, cache_dir: Path, dsets, bs: int = 64,
                            seq_len: int = 72, num_workers: int = 0,
                            vocab=None, force_rebuild: bool = False, **kwargs):
        """
        Load from cache if available, otherwise build and cache.
        """
        cache_dir = Path(cache_dir)

        train_ds = MaterializedLMDataset.from_cache_or_build(
            cache_dir / 'train', dsets, 0, vocab, force_rebuild
        )
        valid_ds = MaterializedLMDataset.from_cache_or_build(
            cache_dir / 'valid', dsets, 1, train_ds.vocab, force_rebuild
        )

        return cls(train_ds, valid_ds, bs=bs, seq_len=seq_len,
                   num_workers=num_workers, **kwargs)
