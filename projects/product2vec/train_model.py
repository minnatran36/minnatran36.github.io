"""
================================================================================
TRAIN MODEL - PRODUCT2VEC USING WORD2VEC
================================================================================

Purpose:
    Train Word2Vec on shopping baskets to learn product embeddings.
    Products that appear together in baskets will have similar vectors.

Input:
    - baskets.pkl (from data_processor.py)

Output:
    - model.bin (trained Word2Vec model - for later use)
    - vectors.json (product vectors - for web export)
    - product_names.json (list of all product names)

The Word2Vec Analogy:
    NLP:        Sentences contain words      → Words in same sentence are related
    Shopping:   Baskets contain products     → Products in same basket are related

================================================================================
"""

import pickle
import json
import numpy as np
from gensim.models import Word2Vec
import time
import os


def load_baskets(baskets_path: str) -> list:
    """
    Load baskets from pickle file.
    
    Args:
        baskets_path: Path to baskets.pkl
        
    Returns:
        List of baskets (each basket is a list of product names)
    """
    
    print("=" * 60)
    print("STEP 1: LOADING BASKETS")
    print("=" * 60)
    
    print(f"\nLoading: {baskets_path}")
    
    with open(baskets_path, "rb") as f:
        baskets = pickle.load(f)
    
    print(f"  ✓ Loaded {len(baskets):,} baskets")
    
    # SAMPLE: Use only 100K baskets (server has 300s CPU limit)
    import random
    random.seed(42)
    if len(baskets) > 100000:
        print(f"  ⚠ Sampling 100,000 baskets (server CPU limit)")
        baskets = random.sample(baskets, 100000)
        print(f"  ✓ Sampled {len(baskets):,} baskets")
    
    # Show sample
    print(f"\nSample baskets:")
    for i, basket in enumerate(baskets[:3]):
        if len(basket) > 5:
            display = basket[:5] + [f"... (+{len(basket)-5} more)"]
        else:
            display = basket
        print(f"  {i+1}: {display}")
    
    return baskets


def train_word2vec(baskets: list, config: dict) -> Word2Vec:
    """
    Train Word2Vec model on baskets.
    
    Args:
        baskets: List of baskets (each basket = list of product names)
        config: Dictionary of hyperparameters
        
    Returns:
        Trained Word2Vec model
    
    =========================================================================
    WORD2VEC EXPLANATION
    =========================================================================
    
    Architecture: Skip-gram (sg=1)
    --------------------------------
    - Given a product (center), predict products around it (context)
    - Example: Given "Pasta", predict ["Sauce", "Parmesan", "Garlic"]
    - This learns: "Pasta" is related to "Sauce", "Parmesan", "Garlic"
    
    Why Skip-gram over CBOW?
    - Better for rare items (important for long-tail products)
    - The original Word2Vec paper found Skip-gram works better for semantics
    
    Negative Sampling (negative=10)
    --------------------------------
    - Problem: Computing softmax over 50K products is expensive
    - Solution: For each positive pair (Pasta, Sauce), sample 10 random
                "negative" products that WEREN'T in the basket
    - Train to distinguish real pairs from fake pairs
    - Much faster, works just as well!
    
    Hyperparameters:
    --------------------------------
    - vector_size (100): Dimensionality of product vectors
                         Higher = more expressive, but slower & more memory
                         100-300 is typical for Word2Vec
    
    - window (10): How many products around the center to consider as context
                   For baskets, we use larger window since all items are related
                   (unlike sentences where distant words are less related)
    
    - min_count (5): Ignore products appearing fewer than 5 times
                     Removes rare/noisy products
    
    - epochs (10): Number of passes through the data
                   More epochs = better convergence, but slower
    
    =========================================================================
    """
    
    print("\n" + "=" * 60)
    print("STEP 2: TRAINING WORD2VEC MODEL")
    print("=" * 60)
    
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  • {key}: {value}")
    
    print(f"\nTraining on {len(baskets):,} baskets...")
    print("(This may take 10-20 minutes for 3M+ baskets)\n")
    
    start_time = time.time()
    
    # -------------------------------------------------------------------------
    # Train Word2Vec
    # -------------------------------------------------------------------------
    # 
    # Key parameters:
    #   sentences = baskets       (our "sentences" are shopping baskets)
    #   vector_size = 100         (each product becomes a 100-dim vector)
    #   window = 10               (context window size)
    #   min_count = 5             (ignore rare products)
    #   sg = 1                    (use Skip-gram, not CBOW)
    #   negative = 10             (use negative sampling with 10 negatives)
    #   workers = 4               (parallelize across 4 CPU cores)
    #   seed = 42                 (for reproducibility)
    
    model = Word2Vec(
        sentences=baskets,
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        sg=1,                           # Skip-gram
        negative=config["negative"],     # Negative sampling
        workers=config["workers"],
        seed=config["seed"],
        epochs=config["epochs"]
    )
    
    elapsed = time.time() - start_time
    
    print(f"  ✓ Training complete!")
    print(f"  ✓ Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"  ✓ Vocabulary size: {len(model.wv):,} products")
    print(f"  ✓ Vector dimensions: {model.wv.vector_size}")
    
    return model


def explore_model(model: Word2Vec):
    """
    Explore the trained model - find similar products, test analogies.
    
    This is just for demonstration/debugging. Shows that the model learned
    meaningful relationships.
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORING LEARNED EMBEDDINGS")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Find similar products
    # -------------------------------------------------------------------------
    # 
    # model.wv.most_similar("Banana") returns products with highest
    # cosine similarity to "Banana" in the embedding space.
    #
    # If training worked, similar products should be things often
    # bought with bananas (yogurt, milk, other fruits, etc.)
    
    print("\n--- Similar Products ---")
    
    # Test with some common products (these should exist in the vocabulary)
    test_products = ["Banana", "Organic Whole Milk", "Bag of Organic Bananas"]
    
    for product in test_products:
        # Check if product exists in vocabulary
        if product in model.wv:
            print(f"\nSimilar to '{product}':")
            similar = model.wv.most_similar(product, topn=5)
            for similar_product, score in similar:
                print(f"  • {similar_product}: {score:.3f}")
        else:
            # Try to find a product that contains this word
            matching = [p for p in model.wv.index_to_key if product.lower() in p.lower()]
            if matching:
                product = matching[0]
                print(f"\nSimilar to '{product}':")
                similar = model.wv.most_similar(product, topn=5)
                for similar_product, score in similar:
                    print(f"  • {similar_product}: {score:.3f}")
    
    # -------------------------------------------------------------------------
    # Show vocabulary sample
    # -------------------------------------------------------------------------
    
    print("\n--- Vocabulary Sample (first 20 products) ---")
    for i, product in enumerate(model.wv.index_to_key[:20]):
        print(f"  {i+1}. {product}")


def save_model(model: Word2Vec, output_dir: str):
    """
    Save the trained model and export vectors.
    
    Saves:
    1. model.bin - Full Word2Vec model (can reload and continue training)
    2. vectors.json - Product vectors as JSON (for web/other languages)
    3. product_names.json - List of all product names in vocabulary
    """
    
    print("\n" + "=" * 60)
    print("STEP 4: SAVING MODEL AND VECTORS")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Save full model (binary format)
    # -------------------------------------------------------------------------
    # This preserves everything - can reload and:
    # - Continue training
    # - Access all model internals
    
    model_path = os.path.join(output_dir, "model.bin")
    print(f"\nSaving model to: {model_path}")
    model.save(model_path)
    print(f"  ✓ Model saved")
    
    # -------------------------------------------------------------------------
    # Export vectors to JSON
    # -------------------------------------------------------------------------
    # Convert numpy arrays to lists for JSON serialization
    # Format: { "product_name": [0.1, -0.2, 0.3, ...], ... }
    
    vectors_path = os.path.join(output_dir, "vectors.json")
    print(f"\nExporting vectors to: {vectors_path}")
    
    vectors_dict = {}
    for product in model.wv.index_to_key:
        # Convert numpy array to list of floats
        vector = model.wv[product].tolist()
        vectors_dict[product] = vector
    
    with open(vectors_path, "w") as f:
        json.dump(vectors_dict, f)
    
    file_size = os.path.getsize(vectors_path) / (1024 * 1024)
    print(f"  ✓ Exported {len(vectors_dict):,} product vectors")
    print(f"  ✓ File size: {file_size:.1f} MB")
    
    # -------------------------------------------------------------------------
    # Save product names list
    # -------------------------------------------------------------------------
    # Simple list of all products (useful for autocomplete, etc.)
    
    names_path = os.path.join(output_dir, "product_names.json")
    print(f"\nSaving product names to: {names_path}")
    
    product_names = list(model.wv.index_to_key)
    
    with open(names_path, "w") as f:
        json.dump(product_names, f)
    
    print(f"  ✓ Saved {len(product_names):,} product names")


def train_pipeline(baskets_path: str, output_dir: str, config: dict):
    """
    Main training pipeline.
    
    Args:
        baskets_path: Path to baskets.pkl
        output_dir: Directory to save outputs
        config: Hyperparameter configuration
    """
    
    print("\n" + "=" * 60)
    print("   PRODUCT2VEC TRAINING PIPELINE")
    print("=" * 60)
    print(f"\nInput: {baskets_path}")
    print(f"Output directory: {output_dir}")
    
    total_start = time.time()
    
    # Step 1: Load baskets
    baskets = load_baskets(baskets_path)
    
    # Step 2: Train Word2Vec
    model = train_word2vec(baskets, config)
    
    # Step 3: Explore model (optional, for verification)
    explore_model(model)
    
    # Step 4: Save model and vectors
    save_model(model, output_dir)
    
    # Done!
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("   TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\n  Output files:")
    print(f"    • {output_dir}/model.bin (full model)")
    print(f"    • {output_dir}/vectors.json (product vectors)")
    print(f"    • {output_dir}/product_names.json (product list)")
    print(f"\n  Next step: Run export_for_web.py to create web-ready files!")
    
    return model


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run this script from command line:
        python train_model.py
    
    Make sure baskets.pkl exists (run data_processor.py first)
    """
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    BASKETS_PATH = "./baskets.pkl"      # Input: baskets from data_processor.py
    OUTPUT_DIR = "./model_output"        # Output directory
    
    # Word2Vec hyperparameters
    CONFIG = {
        "vector_size": 100,    # Dimensionality of vectors
        "window": 5,          # Context window size
        "min_count": 20,        # Ignore products appearing < 5 times
        "negative": 5,        # Number of negative samples
        "epochs": 3,          # Training epochs
        "workers": 4,          # CPU threads for parallel training
        "seed": 42             # Random seed for reproducibility
    }
    
    # -------------------------------------------------------------------------
    # Run training
    # -------------------------------------------------------------------------
    
    model = train_pipeline(
        baskets_path=BASKETS_PATH,
        output_dir=OUTPUT_DIR,
        config=CONFIG
    )
