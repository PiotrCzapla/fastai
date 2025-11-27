# ULMFit Training Documentation

## Overview

This document explains how ULMFit is trained in the current fastai codebase for replicating the paper results on IMDB.

## Key Resources

### Notebooks
1. **`nbs/examples/ulmfit.ipynb`** - Complete working example (⭐ PRIMARY REFERENCE)
2. **`nbs/38_tutorial.text.ipynb`** - Detailed tutorial with explanations
3. **`dev_nbs/course/lesson3-imdb.ipynb`** - Comprehensive lesson

### Source Code
- **`fastai/text/learner.py`** - Main learner functions
- **`fastai/text/models/awdlstm.py`** - AWD-LSTM architecture
- **`fastai/text/data.py`** - Data loading
- **`fastai/text/core.py`** - Tokenization

## Available Pretrained Models

⚠️ **Important:** There is NO pretrained IMDB classifier available for download.

Available pretrained models:
- **WT103_FWD** - WikiText-103 forward language model
- **WT103_BWD** - WikiText-103 backward language model

These are automatically downloaded when you create a learner with `pretrained=True`.

## Training Process (Two Stages)

### Stage 1: Language Model Fine-tuning

Adapts the Wikipedia-pretrained LM to movie review language using **all 100k IMDB texts**.

```python
from fastai.text.all import *

# Get all texts (including 50k unlabeled)
path = untar_data(URLs.IMDB)
texts = get_files(path, extensions=['.txt'],
                 folders=['unsup', 'train', 'test'])
# Returns: 100,000 texts

# Create LM dataloader
splits = RandomSplitter(valid_pct=0.1)(texts)
tfms = [Tokenizer.from_folder(path), Numericalize()]
dsets = Datasets(texts, [tfms], splits=splits, dl_type=LMDataLoader)

bs, sl = 256, 80  # batch size, sequence length
dbunch_lm = dsets.dataloaders(bs=bs, seq_len=sl, val_bs=bs)

# Create learner with pretrained weights
opt_func = partial(Adam, wd=0.1)
learn = language_model_learner(
    dbunch_lm,
    AWD_LSTM,
    opt_func=opt_func,
    metrics=[accuracy, Perplexity()],
    path=path
)
learn = learn.to_fp16()  # Mixed precision (skip on M1 Mac)

# Train frozen (head only)
learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7,0.8))
learn.save('stage1')

# Unfreeze and train full model
learn.unfreeze()
learn.fit_one_cycle(10, 2e-3, moms=(0.8,0.7,0.8))

# Save encoder for classifier
learn.save_encoder('finetuned1')
```

### Stage 2: Classifier Training

Trains on **50k labeled reviews** using gradual unfreezing.

```python
# Get labeled texts only
texts = get_files(path, extensions=['.txt'],
                 folders=['train', 'test'])
splits = GrandparentSplitter(valid_name='test')(texts)

# Create classifier data - MUST use same vocab!
x_tfms = [Tokenizer.from_folder(path),
          Numericalize(vocab=dbunch_lm.vocab)]
dsets = Datasets(texts,
                [x_tfms, [parent_label, Categorize()]],
                splits=splits,
                dl_type=SortedDL)

bs = 64
dls = dsets.dataloaders(before_batch=pad_input_chunk, bs=bs)

# Create classifier and load encoder
opt_func = partial(Adam, wd=0.1)
learn = text_classifier_learner(
    dls,
    AWD_LSTM,
    metrics=[accuracy],
    path=path,
    drop_mult=0.5,
    opt_func=opt_func
)
learn = learn.load_encoder('finetuned1')
learn = learn.to_fp16(clip=0.1)

# Gradual unfreezing with discriminative learning rates
lr = 1e-1 * bs/128

# Train head only
learn.fit_one_cycle(1, lr, moms=(0.8,0.7,0.8), wd=0.1)

# Unfreeze last 2 param groups
learn.freeze_to(-2)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4), lr),
                   moms=(0.8,0.7,0.8), wd=0.1)

# Unfreeze last 3 param groups
learn.freeze_to(-3)
lr /= 2
learn.fit_one_cycle(1, slice(lr/(2.6**4), lr),
                   moms=(0.8,0.7,0.8), wd=0.1)

# Unfreeze all
learn.unfreeze()
lr /= 5
learn.fit_one_cycle(2, slice(lr/(2.6**4), lr),
                   moms=(0.8,0.7,0.8), wd=0.1)
```

## Critical Details

### Must-Have for Paper Replication

✅ **Same vocabulary** - Classifier must use LM's vocab: `Numericalize(vocab=dbunch_lm.vocab)`

✅ **All texts for LM** - Use all 100k texts (train/test/unsup) for language model fine-tuning

✅ **Gradual unfreezing** - Don't just use `fine_tune()`, use the manual unfreezing process

✅ **Discriminative learning rates** - Use `slice(lr/(2.6**4), lr)` where 2.6^4 ≈ 46x slower for early layers

✅ **Momentum schedule** - Use `moms=(0.8,0.7,0.8)` with one-cycle training

✅ **Weight decay** - 0.1 throughout

✅ **Batch-size scaled LR** - For classifier: `lr = 1e-1 * bs/128`

### Architecture: AWD-LSTM

**Configuration (from `awd_lstm_lm_config`):**
- Embedding size: 400
- Hidden size: 1152
- Layers: 3 LSTM layers
- Dropout types:
  - Weight dropout: 0.2
  - Hidden dropout: 0.15
  - Input dropout: 0.25
  - Embedding dropout: 0.02
  - Output dropout: 0.1

**For classification** (`awd_lstm_clas_config`):
- Higher dropouts with `drop_mult=0.5`

## M1 Mac Compatibility Notes

⚠️ **Mixed Precision (FP16):**
- The notebooks use `.to_fp16()` for faster training
- On M1 Macs, you can:
  1. Skip `.to_fp16()` entirely (use FP32)
  2. Training will be slower but work fine
  3. MPS backend should handle this

⚠️ **Training Time:**
- Without GPU acceleration, full training will be slow
- Consider using Google Colab or cloud GPU for full replication
- Or train on smaller sample (URLs.IMDB_SAMPLE) for testing

## Expected Results

**Paper results:**
- IMDB test accuracy: ~95%

**Without LM fine-tuning:**
- Using Wikipedia pretrained weights only: ~92-93%

**Quick training (just `fine_tune()`):**
- Using fine-tuned LM but simple training: ~93-94%

## Testing Inference

Run the test script to see how the model works:

```bash
python test_ulmfit_inference.py
```

This script:
- Downloads pretrained Wikipedia LM weights
- Tests text generation
- Shows how to use the classifier
- Uses IMDB_SAMPLE for quick testing

## Data URLs

```python
URLs.IMDB        # Full dataset (100k reviews)
URLs.IMDB_SAMPLE # Small sample for testing
URLs.WT103_FWD   # Pretrained Wikipedia LM (forward)
URLs.WT103_BWD   # Pretrained Wikipedia LM (backward)
```

## Special Tokens

The tokenizer adds special tokens:
- `xxbos` - Beginning of sentence
- `xxeos` - End of sentence
- `xxmaj` - Next char is uppercase
- `xxup` - Whole word is uppercase
- `xxrep` - Character repetition (e.g., "aaa" → "xxrep 3 a")
- `xxwrep` - Word repetition
- `xxunk` - Unknown token
- `xxpad` - Padding token

## Training Script

For distributed training, see: `nbs/examples/train_imdbclassifier.py`

Supports:
- Distributed data parallel
- Mixed precision (FP16)
- Command-line arguments for hyperparameters

## Files Reference

| File | Purpose |
|------|---------|
| `nbs/examples/ulmfit.ipynb` | ⭐ Main implementation |
| `nbs/38_tutorial.text.ipynb` | Detailed tutorial |
| `nbs/examples/train_imdbclassifier.py` | Distributed training script |
| `fastai/text/learner.py:234` | `language_model_learner()` |
| `fastai/text/learner.py:256` | `text_classifier_learner()` |
| `fastai/text/models/awdlstm.py` | AWD-LSTM implementation |
| `fastai/data/external.py` | Dataset URLs |

## Next Steps

1. **Quick test:** Run `python test_ulmfit_inference.py`
2. **Full training:** Open `nbs/examples/ulmfit.ipynb` in Jupyter
3. **Paper replication:** Follow the two-stage process above with full IMDB dataset
4. **Experiments:** Try different hyperparameters, architectures, etc.

---

**Created:** 2025-11-27
**Purpose:** Documentation for ULMFit training exploration
