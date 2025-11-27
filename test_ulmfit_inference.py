#!/usr/bin/env python3
"""
Simple ULMFit inference test script for M1 Mac compatibility.
Tests the Wikipedia-pretrained model on movie review text.

Note: There's no pretrained IMDB classifier available.
This script demonstrates:
1. How to load the pretrained Wikipedia LM
2. Basic text generation
3. What you'd need to train for IMDB classification
"""

from fastai.text.all import *
import warnings
warnings.filterwarnings('ignore')

def test_language_model():
    """Test the pretrained language model with text generation."""
    print("=" * 70)
    print("Testing Language Model (Wikipedia-trained)")
    print("=" * 70)

    # Download IMDB sample data (smaller dataset for testing)
    print("\n📦 Downloading IMDB sample data...")
    path = untar_data(URLs.IMDB_SAMPLE)
    print(f"✓ Data downloaded to: {path}")

    # Create a simple language model dataloader
    print("\n🔧 Creating language model data loader...")
    dls_lm = TextDataLoaders.from_folder(
        path,
        is_lm=True,
        valid_pct=0.1,
        bs=8  # Small batch size for testing
    )

    # Create learner with pretrained weights (Wikipedia)
    print("\n🧠 Loading pretrained language model...")
    print("   (This downloads Wikipedia-trained weights automatically)")
    learn = language_model_learner(
        dls_lm,
        AWD_LSTM,
        metrics=[accuracy, Perplexity()],
        path=path
    )
    # Skip .to_fp16() for M1 Mac compatibility
    print("✓ Model loaded (running in FP32 for M1 Mac compatibility)")

    # Test text generation
    print("\n" + "=" * 70)
    print("🎬 Generating Movie Reviews")
    print("=" * 70)

    test_prompts = [
        "This movie was amazing because",
        "I didn't like this film because",
        "The acting in this movie"
    ]

    for prompt in test_prompts:
        print(f"\n📝 Prompt: '{prompt}'")
        try:
            generated = learn.predict(prompt, n_words=30, temperature=0.75)
            print(f"💭 Generated: {generated}")
        except Exception as e:
            print(f"❌ Error: {e}")

    return learn

def explain_imdb_training():
    """Explain what's needed to train an IMDB classifier."""
    print("\n" + "=" * 70)
    print("📚 How to Train IMDB Classifier (Paper Replication)")
    print("=" * 70)

    print("""
The ULMFit paper approach requires TWO stages:

STAGE 1: Fine-tune Language Model on IMDB corpus
-------------------------------------------------
• Uses ALL 100k IMDB texts (including 50k unlabeled)
• Adapts Wikipedia-pretrained LM to movie review language
• Training time: ~1-2 hours on GPU, longer on M1 Mac
• Command:
    path = untar_data(URLs.IMDB)
    texts = get_files(path, folders=['train','test','unsup'])
    dls_lm = TextDataLoaders.from_folder(path, is_lm=True)
    learn = language_model_learner(dls_lm, AWD_LSTM)
    learn.fit_one_cycle(1, 2e-2)
    learn.unfreeze()
    learn.fit_one_cycle(10, 2e-3)
    learn.save_encoder('finetuned')

STAGE 2: Train Classifier with Gradual Unfreezing
--------------------------------------------------
• Uses 50k labeled reviews (25k train, 25k test)
• Loads the fine-tuned encoder from Stage 1
• Training time: ~30-60 minutes on GPU
• Command:
    dls_clas = TextDataLoaders.from_folder(path, valid='test')
    learn = text_classifier_learner(dls_clas, AWD_LSTM)
    learn.load_encoder('finetuned')
    learn.fit_one_cycle(1, 1e-2)  # + gradual unfreezing steps

Expected Results:
• Paper reports: ~95% accuracy on IMDB test set
• Without LM fine-tuning: ~92-93% accuracy
""")

def test_quick_classifier():
    """Test a quick classifier without fine-tuning (for demo purposes)."""
    print("\n" + "=" * 70)
    print("🚀 Quick Classifier Test (No Fine-tuning)")
    print("=" * 70)

    print("\n📦 Downloading IMDB sample data...")
    path = untar_data(URLs.IMDB_SAMPLE)

    print("\n🔧 Creating classifier data loader...")
    dls = TextDataLoaders.from_folder(
        path,
        valid='test',
        bs=8  # Small batch size
    )

    print("\n🧠 Creating text classifier with pretrained encoder...")
    learn = text_classifier_learner(
        dls,
        AWD_LSTM,
        drop_mult=0.5,
        metrics=accuracy
    )
    # Skip .to_fp16() for M1 Mac
    print("✓ Classifier created")

    # Test on sample text
    print("\n" + "=" * 70)
    print("🎯 Testing Sentiment Prediction")
    print("=" * 70)

    test_reviews = [
        "This movie was absolutely fantastic! Best film I've seen all year.",
        "Terrible waste of time. The plot made no sense.",
        "It was okay, nothing special."
    ]

    for review in test_reviews:
        print(f"\n📝 Review: '{review}'")
        try:
            pred_class, pred_idx, probs = learn.predict(review)
            print(f"🎭 Prediction: {pred_class}")
            print(f"📊 Confidence: {probs[pred_idx]:.2%}")
        except Exception as e:
            print(f"❌ Error: {e}")

    return learn

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🎬 ULMFit Inference Test Script")
    print("=" * 70)
    print("\n⚠️  Note: This uses IMDB_SAMPLE (small dataset) for quick testing")
    print("    For full paper replication, use URLs.IMDB (100k reviews)")

    # Test 1: Language Model
    print("\n\n")
    try:
        lm_learn = test_language_model()
        print("\n✓ Language model test completed!")
    except Exception as e:
        print(f"\n❌ Language model test failed: {e}")
        import traceback
        traceback.print_exc()

    # Explain training process
    explain_imdb_training()

    # Test 2: Quick classifier (without full training)
    print("\n\n")
    try:
        clas_learn = test_quick_classifier()
        print("\n✓ Classifier test completed!")
        print("\n💡 This used Wikipedia-pretrained weights only.")
        print("   For better results, follow the 2-stage training above.")
    except Exception as e:
        print(f"\n❌ Classifier test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("✅ Tests complete!")
    print("=" * 70)
    print("\n📖 See nbs/examples/ulmfit.ipynb for full training example")
    print("📖 See nbs/38_tutorial.text.ipynb for detailed tutorial")
