#!/usr/bin/env python3
"""
Simple IMDB classifier training using Wikipedia pretrained weights.
Works on both CUDA and MPS (M1 Mac).

This skips LM fine-tuning and trains classifier directly on pretrained weights.
Expected accuracy: ~92-93% (vs ~95% with LM fine-tuning).
"""

from fastai.text.all import *
import argparse
import warnings
warnings.filterwarnings('ignore')

def train_imdb_classifier(
    bs=64,
    epochs=4,
    lr=1e-2,
    use_fp16=False,
    sample=False
):
    """
    Train IMDB classifier using pretrained Wikipedia weights.

    Args:
        bs: Batch size (default: 64)
        epochs: Number of epochs for fine_tune (default: 4)
        lr: Learning rate (default: 1e-2)
        use_fp16: Use mixed precision (works on CUDA, skip on MPS)
        sample: Use IMDB_SAMPLE for quick testing
    """
    print("=" * 70)
    print("🎬 IMDB Classifier Training (Wikipedia Pretrained)")
    print("=" * 70)

    # Download data
    print(f"\n📦 Downloading {'IMDB_SAMPLE' if sample else 'full IMDB'} dataset...")
    path = untar_data(URLs.IMDB_SAMPLE if sample else URLs.IMDB)
    print(f"✓ Data: {path}")

    # Check device
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🖥️  Device: CUDA ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"🖥️  Device: MPS (Apple Silicon)")
        use_fp16 = False  # MPS doesn't support fp16 well yet
    else:
        device = 'cpu'
        print(f"🖥️  Device: CPU")
        use_fp16 = False

    # Create data loaders
    print("\n🔧 Creating data loaders...")
    print(f"   Batch size: {bs}")
    dls = TextDataLoaders.from_folder(
        path,
        valid='test',
        bs=bs
    )
    print(f"✓ Train samples: {len(dls.train_ds)}")
    print(f"✓ Valid samples: {len(dls.valid_ds)}")
    print(f"✓ Vocab size: {len(dls.vocab)}")

    # Show sample
    print("\n📝 Sample data:")
    dls.show_batch(max_n=2)

    # Create learner with pretrained weights
    print("\n🧠 Creating classifier with Wikipedia pretrained weights...")
    print("   (This downloads pretrained weights automatically)")
    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        drop_mult=0.5,
        metrics=accuracy
    )

    # Apply mixed precision if requested
    if use_fp16:
        print("⚡ Using mixed precision (FP16)")
        learn = learn.to_fp16()
    else:
        print("💾 Using full precision (FP32)")

    print(f"✓ Model created with pretrained encoder")

    # Train with fine_tune (automatic gradual unfreezing)
    print("\n" + "=" * 70)
    print("🏋️  Training Classifier")
    print("=" * 70)
    print(f"\n📊 Hyperparameters:")
    print(f"   Learning rate: {lr}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {bs}")
    print(f"   Drop mult: 0.5")

    print("\n🚀 Starting training...")
    print("   Stage 1: Frozen encoder (1 epoch)")
    print(f"   Stage 2: Unfrozen encoder ({epochs} epochs)")

    learn.fine_tune(epochs, lr)

    # Show results
    print("\n" + "=" * 70)
    print("📊 Training Complete!")
    print("=" * 70)

    # Test predictions
    print("\n🎯 Testing Predictions:")
    print("-" * 70)

    test_reviews = [
        "This movie was absolutely fantastic! Best film I've seen all year.",
        "Terrible waste of time. The plot made no sense and the acting was awful.",
        "It was okay, nothing special. Watchable but forgettable.",
        "I loved every minute! The characters were so well developed.",
        "Boring and predictable. Would not recommend."
    ]

    for review in test_reviews:
        pred_class, pred_idx, probs = learn.predict(review)
        confidence = probs[pred_idx].item()
        print(f"\n📝 '{review[:60]}...'")
        print(f"   → {pred_class.upper()} ({confidence:.1%} confidence)")

    # Save model
    model_name = 'imdb_classifier_simple'
    print(f"\n💾 Saving model as '{model_name}'...")
    learn.save(model_name)
    print(f"✓ Model saved to: {learn.path/learn.model_dir/model_name}.pth")

    # Export for inference
    export_name = 'imdb_classifier_export.pkl'
    print(f"\n📦 Exporting for inference as '{export_name}'...")
    learn.export(export_name)
    print(f"✓ Exported to: {learn.path/export_name}")

    print("\n" + "=" * 70)
    print("✅ All Done!")
    print("=" * 70)

    return learn

def load_and_test(path=None, sample=False):
    """Load saved model and test it."""
    print("\n" + "=" * 70)
    print("🔄 Loading Saved Model")
    print("=" * 70)

    if path is None:
        path = untar_data(URLs.IMDB_SAMPLE if sample else URLs.IMDB)

    export_file = path / 'imdb_classifier_export.pkl'

    if not export_file.exists():
        print(f"❌ Model not found at: {export_file}")
        print("   Train the model first with --train")
        return None

    print(f"📂 Loading from: {export_file}")
    learn = load_learner(export_file)
    print("✓ Model loaded!")

    # Interactive testing
    print("\n🎯 Interactive Testing (Ctrl+C to exit)")
    print("-" * 70)

    try:
        while True:
            review = input("\nEnter movie review: ").strip()
            if not review:
                continue

            pred_class, pred_idx, probs = learn.predict(review)
            confidence = probs[pred_idx].item()

            print(f"→ {pred_class.upper()} ({confidence:.1%} confidence)")
            print(f"  [neg: {probs[0]:.1%}, pos: {probs[1]:.1%}]")

    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")

    return learn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train IMDB classifier with Wikipedia pretrained weights"
    )
    parser.add_argument(
        '--bs', type=int, default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--epochs', type=int, default=4,
        help='Number of epochs (default: 4)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-2,
        help='Learning rate (default: 1e-2)'
    )
    parser.add_argument(
        '--fp16', action='store_true',
        help='Use mixed precision (CUDA only)'
    )
    parser.add_argument(
        '--sample', action='store_true',
        help='Use IMDB_SAMPLE for quick testing'
    )
    parser.add_argument(
        '--test', action='store_true',
        help='Load saved model and test interactively'
    )

    args = parser.parse_args()

    if args.test:
        load_and_test(sample=args.sample)
    else:
        learn = train_imdb_classifier(
            bs=args.bs,
            epochs=args.epochs,
            lr=args.lr,
            use_fp16=args.fp16,
            sample=args.sample
        )

        print("\n💡 To test the saved model:")
        print(f"   python {__file__} --test")
