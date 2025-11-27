# Quick Start: IMDB Classifier with Pretrained Weights

## TL;DR - Train Now

### On CUDA (GPU):
```bash
# Quick test with sample data (~5 min)
python train_imdb_classifier_simple.py --sample --fp16

# Full training (~30-60 min)
python train_imdb_classifier_simple.py --fp16
```

### On MPS (M1 Mac):
```bash
# Quick test with sample data
python train_imdb_classifier_simple.py --sample

# Full training (will be slower)
python train_imdb_classifier_simple.py
```

### On CPU:
```bash
# Only use sample data (full dataset will be very slow)
python train_imdb_classifier_simple.py --sample
```

## What This Does

✅ **Automatically downloads** Wikipedia pretrained weights (WT103)
✅ **Downloads IMDB dataset** automatically
✅ **Trains classifier** using pretrained encoder
✅ **Saves model** for later use
✅ **Tests predictions** on sample reviews
✅ **Works on CUDA, MPS, and CPU**

## Command Line Options

```bash
python train_imdb_classifier_simple.py [OPTIONS]

Options:
  --bs BS          Batch size (default: 64)
  --epochs N       Number of epochs (default: 4)
  --lr LR          Learning rate (default: 1e-2)
  --fp16           Use mixed precision (CUDA only)
  --sample         Use small IMDB_SAMPLE for testing
  --test           Load saved model and test interactively
```

## Examples

### 1. Quick Test (Small Dataset)
```bash
# Train on sample data to verify everything works
python train_imdb_classifier_simple.py --sample
```

### 2. Full Training with Custom Settings
```bash
# Train on full IMDB with larger batch size
python train_imdb_classifier_simple.py --bs 128 --epochs 6 --lr 2e-2
```

### 3. Interactive Testing
```bash
# After training, test the saved model interactively
python train_imdb_classifier_simple.py --test

# Then type movie reviews and get predictions
Enter movie review: This movie was amazing!
→ POS (98.5% confidence)
```

## Expected Results

### With Wikipedia Pretrained Only (this script):
- **IMDB Test Accuracy: ~92-93%**
- Training time: 30-60 min on GPU
- Training time: 2-4 hours on M1 Mac
- Training time: Much longer on CPU (use --sample)

### With Full LM Fine-tuning (see ULMFIT_TRAINING_NOTES.md):
- **IMDB Test Accuracy: ~95%**
- But requires additional LM fine-tuning step

## What Gets Downloaded

When you run the script, it automatically downloads:

1. **Pretrained model** (~400 MB): Wikipedia-trained language model
   - Location: `~/.fastai/models/`
   - Only downloaded once, then cached

2. **IMDB dataset** (~84 MB for full, ~3 MB for sample):
   - Location: `~/.fastai/data/imdb/` or `imdb_sample/`
   - 50k reviews (full) or ~1k reviews (sample)

## Saved Files

After training, you'll find:

```
~/.fastai/data/imdb/
├── models/
│   └── imdb_classifier_simple.pth    # Model weights
└── imdb_classifier_export.pkl         # Exported for inference
```

## Using the Trained Model

### In Python:
```python
from fastai.text.all import *

# Load the exported model
learn = load_learner('~/.fastai/data/imdb/imdb_classifier_export.pkl')

# Predict
pred_class, pred_idx, probs = learn.predict("Great movie!")
print(f"Sentiment: {pred_class}, Confidence: {probs[pred_idx]:.1%}")
```

### Via Command Line:
```bash
# Interactive testing
python train_imdb_classifier_simple.py --test
```

## Troubleshooting

### Issue: Out of Memory (OOM)
```bash
# Reduce batch size
python train_imdb_classifier_simple.py --bs 32
```

### Issue: MPS/CUDA not detected
```python
# Check in Python:
import torch
print("CUDA:", torch.cuda.is_available())
print("MPS:", torch.backends.mps.is_available())
```

### Issue: Training too slow on CPU
```bash
# Use sample dataset only
python train_imdb_classifier_simple.py --sample --epochs 2
```

### Issue: Mixed precision errors on MPS
```bash
# Don't use --fp16 on M1 Macs (it's auto-disabled)
python train_imdb_classifier_simple.py  # FP16 auto-disabled on MPS
```

## Performance Tips

### For CUDA:
- ✅ Use `--fp16` for 2x speedup
- ✅ Increase batch size if you have VRAM: `--bs 128`
- ✅ Use full dataset for best accuracy

### For M1 Mac (MPS):
- ⚠️ Don't use `--fp16` (not well supported)
- ✅ Batch size 64 usually works well
- ✅ Full dataset works but is slower (~2-4 hours)
- 💡 Consider using sample for quick testing

### For CPU:
- ⚠️ Only use `--sample` dataset
- ⚠️ Reduce batch size: `--bs 16`
- ⚠️ Reduce epochs: `--epochs 2`
- 💡 Consider using Colab/cloud for full training

## Comparison: Simple vs Full ULMFit

| Method | Accuracy | Training Time | Complexity |
|--------|----------|---------------|------------|
| **This script** (Wikipedia only) | ~92-93% | 30-60 min GPU | ⭐ Simple |
| **Full ULMFit** (LM fine-tuning) | ~95% | 2-3 hours GPU | ⭐⭐⭐ Complex |

**Recommendation:** Start with this simple script. Only do full LM fine-tuning if you need that extra 2-3% accuracy.

## Next Steps

1. **Quick test**: Run with `--sample` to verify everything works
2. **Full training**: Run without `--sample` for best results
3. **Experiment**: Try different learning rates, epochs, batch sizes
4. **Full ULMFit**: See `ULMFIT_TRAINING_NOTES.md` for 2-stage training

## Code Structure

The script does:

```python
# 1. Load data
dls = TextDataLoaders.from_folder(path, valid='test')

# 2. Create learner (pretrained=True downloads Wikipedia weights)
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5)

# 3. Train with automatic gradual unfreezing
learn.fine_tune(epochs, lr)

# 4. Save
learn.export('imdb_classifier_export.pkl')
```

That's it! The `fine_tune()` method automatically:
- Freezes encoder and trains head (1 epoch)
- Unfreezes encoder and trains all (N epochs)
- Uses discriminative learning rates
- Uses one-cycle learning rate schedule

## Files in This Repo

- **`train_imdb_classifier_simple.py`** ⭐ - Use this for quick training
- **`test_ulmfit_inference.py`** - Basic inference testing
- **`ULMFIT_TRAINING_NOTES.md`** - Full 2-stage training details
- **`QUICK_START.md`** - This file

---

**Ready to train?** Run: `python train_imdb_classifier_simple.py --sample`
