"""
================================================================================
EXPORT FOR WEB - PREPARE DATA FOR GITHUB PAGES WEBSITE
================================================================================

Purpose:
    Convert the trained model outputs into optimized files for a static website.
    The website can't run Python, so we pre-compute everything it needs.

Input:
    - model_output/vectors.json (product vectors from train_model.py)
    - model_output/product_names.json (product list)

Output:
    - web_data/products.json (product list with categories)
    - web_data/similarities.json (pre-computed similar products for each product)
    - web_data/config.json (metadata about the model)

Why Pre-compute?
    - Website loads JSON, can't run Word2Vec
    - Computing cosine similarity in JavaScript is slow for 50K products
    - Pre-compute top similar products for instant predictions

================================================================================
"""

import json
import numpy as np
import os
import time
from collections import defaultdict


def load_vectors(vectors_path: str) -> dict:
    """
    Load product vectors from JSON.
    
    Args:
        vectors_path: Path to vectors.json
        
    Returns:
        Dictionary of {product_name: vector_list}
    """
    
    print("=" * 60)
    print("STEP 1: LOADING VECTORS")
    print("=" * 60)
    
    print(f"\nLoading: {vectors_path}")
    
    with open(vectors_path, "r") as f:
        vectors = json.load(f)
    
    print(f"  ✓ Loaded {len(vectors):,} product vectors")
    print(f"  ✓ Vector dimension: {len(list(vectors.values())[0])}")
    
    return vectors


def load_product_categories(products_csv_path: str, aisles_csv_path: str, departments_csv_path: str) -> dict:
    """
    Load product category information from original CSV files.
    
    This adds aisle and department info to make the website more useful.
    
    Returns:
        Dictionary of {product_name: {"aisle": ..., "department": ...}}
    """
    
    print("\n" + "=" * 60)
    print("STEP 2: LOADING PRODUCT CATEGORIES (optional)")
    print("=" * 60)
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [products_csv_path, aisles_csv_path, departments_csv_path]):
        print("  ⚠ Category files not found, skipping categories")
        return {}
    
    try:
        import pandas as pd
        
        print(f"\nLoading category data...")
        
        products = pd.read_csv(products_csv_path)
        aisles = pd.read_csv(aisles_csv_path)
        departments = pd.read_csv(departments_csv_path)
        
        # Merge to get category names
        products = products.merge(aisles, on="aisle_id", how="left")
        products = products.merge(departments, on="department_id", how="left")
        
        # Create lookup dictionary
        categories = {}
        for _, row in products.iterrows():
            categories[row["product_name"]] = {
                "aisle": row.get("aisle", "unknown"),
                "department": row.get("department", "unknown")
            }
        
        print(f"  ✓ Loaded categories for {len(categories):,} products")
        return categories
        
    except Exception as e:
        print(f"  ⚠ Error loading categories: {e}")
        return {}


def cosine_similarity(vec1: list, vec2: list) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Cosine similarity = dot(A, B) / (||A|| * ||B||)
    
    Range: -1 to 1
        1 = identical direction (most similar)
        0 = perpendicular (unrelated)
       -1 = opposite direction (least similar)
    """
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_all_similarities(vectors: dict, top_k: int = 20) -> dict:
    """
    Pre-compute top-K similar products for every product.
    
    This is the most expensive operation, but we only do it once.
    The website can then just look up pre-computed results.
    
    Args:
        vectors: Dictionary of {product_name: vector}
        top_k: Number of similar products to store per product
        
    Returns:
        Dictionary of {product_name: [(similar_product, score), ...]}
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: COMPUTING SIMILARITIES")
    print("=" * 60)
    
    print(f"\nComputing top {top_k} similar products for each product...")
    print(f"This will take a few minutes for {len(vectors):,} products...\n")
    
    # Convert to numpy arrays for faster computation
    product_names = list(vectors.keys())
    vector_matrix = np.array([vectors[name] for name in product_names])
    
    # Normalize vectors (for faster cosine similarity computation)
    # Cosine similarity of normalized vectors = just dot product
    norms = np.linalg.norm(vector_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_matrix = vector_matrix / norms
    
    similarities = {}
    total = len(product_names)
    start_time = time.time()
    
    for i, product in enumerate(product_names):
        # Progress update every 5000 products
        if i % 5000 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  Processing {i:,}/{total:,} ({i/total*100:.1f}%) "
                  f"- {remaining:.0f}s remaining")
        
        # Compute similarity with all products (using dot product of normalized vectors)
        product_vector = normalized_matrix[i]
        all_similarities = np.dot(normalized_matrix, product_vector)
        
        # Get top K (excluding self)
        # argsort returns indices that would sort the array
        # We want highest similarity, so we negate and take first top_k+1
        top_indices = np.argsort(-all_similarities)[:top_k + 1]
        
        # Build result list (excluding self)
        similar_products = []
        for idx in top_indices:
            if idx != i:  # Exclude self
                similar_name = product_names[idx]
                score = float(all_similarities[idx])
                similar_products.append([similar_name, round(score, 4)])
                
                if len(similar_products) >= top_k:
                    break
        
        similarities[product] = similar_products
    
    elapsed = time.time() - start_time
    print(f"\n  ✓ Computed similarities in {elapsed:.1f} seconds")
    
    return similarities


def compute_similarities_optimized(vectors: dict, top_k: int = 20, batch_size: int = 1000) -> dict:
    """
    Memory-optimized version for large datasets.
    Processes in batches to avoid memory issues.
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: COMPUTING SIMILARITIES (OPTIMIZED)")
    print("=" * 60)
    
    print(f"\nComputing top {top_k} similar products for each product...")
    print(f"Processing {len(vectors):,} products in batches of {batch_size}...\n")
    
    product_names = list(vectors.keys())
    vector_matrix = np.array([vectors[name] for name in product_names], dtype=np.float32)
    
    # Normalize
    norms = np.linalg.norm(vector_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_matrix = vector_matrix / norms
    
    similarities = {}
    total = len(product_names)
    start_time = time.time()
    
    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        
        # Progress update
        elapsed = time.time() - start_time
        rate = batch_start / elapsed if elapsed > 0 else 0
        remaining = (total - batch_start) / rate if rate > 0 else 0
        print(f"  Processing {batch_start:,}-{batch_end:,}/{total:,} "
              f"- {remaining:.0f}s remaining")
        
        # Compute similarities for this batch
        batch_vectors = normalized_matrix[batch_start:batch_end]
        batch_similarities = np.dot(batch_vectors, normalized_matrix.T)
        
        # Process each product in batch
        for i, global_idx in enumerate(range(batch_start, batch_end)):
            product = product_names[global_idx]
            all_sims = batch_similarities[i]
            
            # Get top K+1 indices (to exclude self)
            top_indices = np.argpartition(-all_sims, top_k + 1)[:top_k + 1]
            top_indices = top_indices[np.argsort(-all_sims[top_indices])]
            
            # Build result
            similar_products = []
            for idx in top_indices:
                if idx != global_idx:
                    similar_products.append([
                        product_names[idx],
                        round(float(all_sims[idx]), 4)
                    ])
                    if len(similar_products) >= top_k:
                        break
            
            similarities[product] = similar_products
    
    elapsed = time.time() - start_time
    print(f"\n  ✓ Computed similarities in {elapsed:.1f} seconds")
    
    return similarities


def create_products_json(vectors: dict, categories: dict, output_path: str):
    """
    Create products.json - list of all products with metadata.
    
    Format:
    [
        {"name": "Banana", "aisle": "fresh fruits", "department": "produce"},
        {"name": "Organic Milk", "aisle": "milk", "department": "dairy eggs"},
        ...
    ]
    """
    
    print("\n" + "=" * 60)
    print("STEP 4: CREATING PRODUCTS.JSON")
    print("=" * 60)
    
    products_list = []
    
    # Group products by department for easier browsing
    by_department = defaultdict(list)
    
    for product_name in vectors.keys():
        cat = categories.get(product_name, {})
        
        product_entry = {
            "name": product_name,
            "aisle": cat.get("aisle", "unknown"),
            "department": cat.get("department", "unknown")
        }
        
        products_list.append(product_entry)
        by_department[product_entry["department"]].append(product_name)
    
    # Sort by name
    products_list.sort(key=lambda x: x["name"])
    
    print(f"\nSaving to: {output_path}")
    
    with open(output_path, "w") as f:
        json.dump(products_list, f)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved {len(products_list):,} products")
    print(f"  ✓ File size: {file_size:.2f} MB")
    
    # Show department breakdown
    print(f"\nProducts by department:")
    for dept, prods in sorted(by_department.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  • {dept}: {len(prods):,} products")


def create_similarities_json(similarities: dict, output_path: str):
    """
    Create similarities.json - pre-computed similar products.
    
    Format:
    {
        "Banana": [["Organic Banana", 0.95], ["Strawberries", 0.82], ...],
        "Milk": [["Organic Milk", 0.93], ["Eggs", 0.78], ...],
        ...
    }
    """
    
    print("\n" + "=" * 60)
    print("STEP 5: CREATING SIMILARITIES.JSON")
    print("=" * 60)
    
    print(f"\nSaving to: {output_path}")
    
    with open(output_path, "w") as f:
        json.dump(similarities, f)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Saved similarities for {len(similarities):,} products")
    print(f"  ✓ File size: {file_size:.2f} MB")


def create_config_json(vectors: dict, output_path: str):
    """
    Create config.json - metadata about the model.
    
    Useful for the website to display info about the model.
    """
    
    print("\n" + "=" * 60)
    print("STEP 6: CREATING CONFIG.JSON")
    print("=" * 60)
    
    config = {
        "model_name": "Product2Vec",
        "description": "Word2Vec trained on Instacart shopping baskets",
        "num_products": len(vectors),
        "vector_dimensions": len(list(vectors.values())[0]),
        "algorithm": "Skip-gram with Negative Sampling",
        "training_data": "Instacart Online Grocery Shopping Dataset 2017",
        "num_baskets": "~3.2 million",
        "paper": "Mikolov et al. (2013) - Efficient Estimation of Word Representations"
    }
    
    print(f"\nSaving to: {output_path}")
    
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"  ✓ Config saved")
    
    for key, value in config.items():
        print(f"  • {key}: {value}")


def export_for_web(
    vectors_path: str,
    output_dir: str,
    products_csv: str = None,
    aisles_csv: str = None,
    departments_csv: str = None,
    top_k: int = 20
):
    """
    Main export pipeline.
    
    Args:
        vectors_path: Path to vectors.json from train_model.py
        output_dir: Directory to save web files
        products_csv: Optional path to products.csv (for categories)
        aisles_csv: Optional path to aisles.csv
        departments_csv: Optional path to departments.csv
        top_k: Number of similar products to pre-compute
    """
    
    print("\n" + "=" * 60)
    print("   EXPORT FOR WEB PIPELINE")
    print("=" * 60)
    print(f"\nInput: {vectors_path}")
    print(f"Output directory: {output_dir}")
    
    total_start = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Load vectors
    vectors = load_vectors(vectors_path)
    
    # Step 2: Load categories (optional)
    categories = {}
    if products_csv and aisles_csv and departments_csv:
        categories = load_product_categories(products_csv, aisles_csv, departments_csv)
    
    # Step 3: Compute similarities (this takes the longest)
    similarities = compute_similarities_optimized(vectors, top_k=top_k)
    
    # Step 4: Create products.json
    create_products_json(
        vectors, 
        categories, 
        os.path.join(output_dir, "products.json")
    )
    
    # Step 5: Create similarities.json
    create_similarities_json(
        similarities,
        os.path.join(output_dir, "similarities.json")
    )
    
    # Step 6: Create config.json
    create_config_json(
        vectors,
        os.path.join(output_dir, "config.json")
    )
    
    # Done!
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("   EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\n  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\n  Output files (copy these to your website):")
    print(f"    • {output_dir}/products.json")
    print(f"    • {output_dir}/similarities.json")
    print(f"    • {output_dir}/config.json")
    print(f"\n  Next step: Copy web_data/ to your GitHub Pages repo!")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run this script from command line:
        python export_for_web.py
    
    Make sure vectors.json exists (run train_model.py first)
    """
    
    # Configuration
    VECTORS_PATH = "./model_output/vectors.json"
    OUTPUT_DIR = "./web_data"
    
    # Optional: paths to original CSVs for category info
    PRODUCTS_CSV = "./archive/products.csv"
    AISLES_CSV = "./archive/aisles.csv"
    DEPARTMENTS_CSV = "./archive/departments.csv"
    
    # Number of similar products to pre-compute per product
    TOP_K = 20
    
    # Run export
    export_for_web(
        vectors_path=VECTORS_PATH,
        output_dir=OUTPUT_DIR,
        products_csv=PRODUCTS_CSV,
        aisles_csv=AISLES_CSV,
        departments_csv=DEPARTMENTS_CSV,
        top_k=TOP_K
    )
