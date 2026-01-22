"""
================================================================================
DATA PROCESSOR FOR PRODUCT2VEC
================================================================================

Purpose:
    Transform raw Instacart CSV files into a simple list of baskets (shopping trips)
    that can be used to train Word2Vec.

Input Files (from Instacart dataset):
    - order_products__prior.csv  (32M rows: which products are in which orders)
    - products.csv               (50K rows: product ID to name mapping)

Output File:
    - baskets.pkl  (list of lists: each inner list = one shopping basket)

Example Output:
    [
        ["Organic Eggs", "Whole Milk", "Banana"],           # Order 1
        ["Pasta", "Marinara Sauce", "Parmesan Cheese"],     # Order 2
        ["Chicken Breast", "Broccoli", "Brown Rice"],       # Order 3
        ...
        # ~3.2 million baskets total
    ]

Why This Format?
    Word2Vec needs "sentences" to learn from. In NLP, a sentence is a list of words.
    For us, a "sentence" is a basket (list of product names).
    
    Word2Vec will learn: products appearing in the same basket are related.

================================================================================
"""

import pandas as pd
import pickle
import os
import time


def load_raw_data(data_dir: str) -> tuple:
    """
    Load the two CSV files we need.
    
    Args:
        data_dir: Path to folder containing the CSV files
        
    Returns:
        Tuple of (order_products DataFrame, products DataFrame)
    
    We only need 2 of the 6 files:
    - order_products__prior.csv: Links order_id to product_id (32M rows)
    - products.csv: Maps product_id to product_name (50K rows)
    
    We DON'T need:
    - orders.csv (has user_id, time — we don't care)
    - aisles.csv (product categories — we don't need)
    - departments.csv (product categories — we don't need)
    - order_products__train.csv (smaller dataset for ML competition)
    """
    
    print("=" * 60)
    print("STEP 1: LOADING RAW CSV FILES")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Load order_products__prior.csv
    # -------------------------------------------------------------------------
    # This file tells us: which products are in which orders
    #
    # Structure:
    #   order_id | product_id | add_to_cart_order | reordered
    #   ---------|------------|-------------------|----------
    #      2     |   33120    |        1          |    1
    #      2     |   28985    |        2          |    1
    #      2     |    9327    |        3          |    0
    #
    # We only need: order_id and product_id (to group products by order)
    # We ignore: add_to_cart_order, reordered (not needed for Word2Vec)
    
    order_products_path = os.path.join(data_dir, "order_products__prior.csv")
    print(f"\nLoading: {order_products_path}")
    
    # Only load the columns we need (saves memory!)
    order_products = pd.read_csv(
        order_products_path,
        usecols=["order_id", "product_id"]  # Ignore add_to_cart_order, reordered
    )
    
    print(f"  ✓ Loaded {len(order_products):,} rows")
    print(f"  ✓ Unique orders: {order_products['order_id'].nunique():,}")
    print(f"  ✓ Unique products: {order_products['product_id'].nunique():,}")
    
    # -------------------------------------------------------------------------
    # Load products.csv
    # -------------------------------------------------------------------------
    # This file tells us: what is the name of each product
    #
    # Structure:
    #   product_id | product_name                    | aisle_id | department_id
    #   -----------|--------------------------------|----------|---------------
    #      1       | Chocolate Sandwich Cookies      |    61    |      19
    #     24       | Organic Bananas                 |    24    |       4
    #
    # We only need: product_id and product_name (to convert IDs to names)
    # We ignore: aisle_id, department_id (categories — not needed)
    
    products_path = os.path.join(data_dir, "products.csv")
    print(f"\nLoading: {products_path}")
    
    products = pd.read_csv(
        products_path,
        usecols=["product_id", "product_name"]  # Ignore aisle_id, department_id
    )
    
    print(f"  ✓ Loaded {len(products):,} products")
    
    return order_products, products


def join_tables(order_products: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Join order_products with products to get product names.
    
    BEFORE (order_products):
        order_id | product_id
        ---------|----------
           2     |   33120      ← Just IDs, not meaningful!
           2     |   28985
           
    AFTER (joined):
        order_id | product_id | product_name
        ---------|------------|------------------
           2     |   33120    | Organic Eggs       ← Now we have names!
           2     |   28985    | Michigan Apples
    
    This is a LEFT JOIN on product_id.
    """
    
    print("\n" + "=" * 60)
    print("STEP 2: JOINING TABLES")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Merge (JOIN) the two DataFrames
    # -------------------------------------------------------------------------
    # 
    # SQL equivalent:
    #   SELECT op.order_id, op.product_id, p.product_name
    #   FROM order_products op
    #   LEFT JOIN products p ON op.product_id = p.product_id
    #
    # Why LEFT JOIN?
    #   - Keep all rows from order_products
    #   - Add product_name where we can match product_id
    #   - If a product_id has no name, it becomes NaN (we'll handle this)
    
    print("\nJoining order_products with products on 'product_id'...")
    
    joined = pd.merge(
        order_products,           # Left table (32M rows)
        products,                 # Right table (50K rows)
        on="product_id",          # Join column
        how="left"                # Keep all orders, even if product name missing
    )
    
    print(f"  ✓ Joined table has {len(joined):,} rows")
    
    # -------------------------------------------------------------------------
    # Handle missing product names (if any)
    # -------------------------------------------------------------------------
    # Some product_ids might not have a name in products.csv
    # We'll drop these rows (or you could keep them as "Unknown")
    
    missing_count = joined["product_name"].isna().sum()
    if missing_count > 0:
        print(f"  ⚠ Found {missing_count:,} rows with missing product names")
        print(f"    Dropping these rows...")
        joined = joined.dropna(subset=["product_name"])
        print(f"  ✓ After dropping: {len(joined):,} rows")
    else:
        print(f"  ✓ No missing product names!")
    
    return joined


def create_baskets(joined_df: pd.DataFrame) -> list:
    """
    Group products by order_id to create baskets.
    
    BEFORE (joined_df):
        order_id | product_name
        ---------|------------------
           2     | Organic Eggs
           2     | Michigan Apples
           2     | Garlic Powder
           3     | Organic Banana
           3     | Whole Milk
           
    AFTER (baskets):
        [
            ["Organic Eggs", "Michigan Apples", "Garlic Powder"],  # Order 2
            ["Organic Banana", "Whole Milk"],                       # Order 3
        ]
    
    This is the KEY step — converting relational data to "sentences" for Word2Vec.
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: CREATING BASKETS")
    print("=" * 60)
    
    # -------------------------------------------------------------------------
    # Group by order_id, collect all product_names into a list
    # -------------------------------------------------------------------------
    #
    # SQL equivalent:
    #   SELECT order_id, ARRAY_AGG(product_name) as basket
    #   FROM joined_df
    #   GROUP BY order_id
    #
    # Pandas groupby().apply(list) does exactly this:
    #   - Group all rows with the same order_id
    #   - Collect the product_name values into a Python list
    
    print("\nGrouping products by order_id...")
    start_time = time.time()
    
    # Group by order_id and convert each group's product names to a list
    baskets_series = joined_df.groupby("order_id")["product_name"].apply(list)
    
    # Convert from pandas Series to Python list
    baskets = baskets_series.tolist()
    
    elapsed = time.time() - start_time
    print(f"  ✓ Created {len(baskets):,} baskets in {elapsed:.1f} seconds")
    
    # -------------------------------------------------------------------------
    # Show statistics about basket sizes
    # -------------------------------------------------------------------------
    
    basket_sizes = [len(b) for b in baskets]
    avg_size = sum(basket_sizes) / len(basket_sizes)
    min_size = min(basket_sizes)
    max_size = max(basket_sizes)
    
    print(f"\nBasket Statistics:")
    print(f"  • Total baskets: {len(baskets):,}")
    print(f"  • Average size: {avg_size:.1f} products per basket")
    print(f"  • Smallest basket: {min_size} products")
    print(f"  • Largest basket: {max_size} products")
    
    # -------------------------------------------------------------------------
    # Show sample baskets
    # -------------------------------------------------------------------------
    
    print(f"\nSample Baskets:")
    for i, basket in enumerate(baskets[:3]):
        # Truncate long baskets for display
        if len(basket) > 5:
            display = basket[:5] + [f"... (+{len(basket)-5} more)"]
        else:
            display = basket
        print(f"  Basket {i+1}: {display}")
    
    return baskets


def filter_baskets(baskets: list, min_size: int = 2, max_size: int = 100) -> list:
    """
    Filter out baskets that are too small or too large.
    
    Why filter?
    - Baskets with 1 item: Can't learn co-occurrence (need at least 2 items)
    - Baskets with 100+ items: Might be bulk/wholesale orders (outliers)
    
    Args:
        baskets: List of baskets
        min_size: Minimum items per basket (default: 2)
        max_size: Maximum items per basket (default: 100)
        
    Returns:
        Filtered list of baskets
    """
    
    print("\n" + "=" * 60)
    print("STEP 4: FILTERING BASKETS")
    print("=" * 60)
    
    original_count = len(baskets)
    
    # Keep only baskets within size range
    filtered = [b for b in baskets if min_size <= len(b) <= max_size]
    
    removed_count = original_count - len(filtered)
    
    print(f"\n  • Original baskets: {original_count:,}")
    print(f"  • Filter: {min_size} ≤ basket size ≤ {max_size}")
    print(f"  • Removed: {removed_count:,} baskets")
    print(f"  • Remaining: {len(filtered):,} baskets")
    
    return filtered


def save_baskets(baskets: list, output_path: str):
    """
    Save baskets to a pickle file.
    
    Why pickle?
    - Fast to save and load
    - Preserves Python list structure exactly
    - Compact file size
    
    Alternative formats:
    - JSON: Human-readable but slower and larger file
    - CSV: Doesn't handle lists-of-lists well
    """
    
    print("\n" + "=" * 60)
    print("STEP 5: SAVING BASKETS")
    print("=" * 60)
    
    print(f"\nSaving to: {output_path}")
    
    with open(output_path, "wb") as f:
        pickle.dump(baskets, f)
    
    # Check file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
    print(f"  ✓ Saved {len(baskets):,} baskets")
    print(f"  ✓ File size: {file_size:.1f} MB")


def process_data(data_dir: str, output_path: str, min_basket_size: int = 2, max_basket_size: int = 100):
    """
    Main function: Run the entire data processing pipeline.
    
    Args:
        data_dir: Path to folder containing Instacart CSV files
        output_path: Path to save the output baskets.pkl file
        min_basket_size: Minimum products per basket (default: 2)
        max_basket_size: Maximum products per basket (default: 100)
    
    Pipeline:
        CSV files → Load → Join → Group → Filter → Save → baskets.pkl
    """
    
    print("\n" + "=" * 60)
    print("   PRODUCT2VEC DATA PROCESSOR")
    print("=" * 60)
    print(f"\nInput directory: {data_dir}")
    print(f"Output file: {output_path}")
    
    start_time = time.time()
    
    # Step 1: Load CSV files
    order_products, products = load_raw_data(data_dir)
    
    # Step 2: Join tables to get product names
    joined = join_tables(order_products, products)
    
    # Step 3: Create baskets (group by order_id)
    baskets = create_baskets(joined)
    
    # Step 4: Filter baskets by size
    baskets = filter_baskets(baskets, min_basket_size, max_basket_size)
    
    # Step 5: Save to pickle file
    save_baskets(baskets, output_path)
    
    # Done!
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("   PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\n  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Output: {output_path}")
    print(f"  Baskets: {len(baskets):,}")
    print(f"\n  Next step: Run train_model.py to train Word2Vec!")
    
    return baskets


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Run this script from command line:
        python data_processor.py
    
    Make sure you have these files in the 'archive' folder:
        - order_products__prior.csv
        - products.csv
    """
    
    # Configuration
    DATA_DIR = "./archive"              # Folder containing CSV files
    OUTPUT_PATH = "./baskets.pkl"       # Output file
    MIN_BASKET_SIZE = 2                 # Minimum products per basket
    MAX_BASKET_SIZE = 100               # Maximum products per basket
    
    # Run the pipeline
    baskets = process_data(
        data_dir=DATA_DIR,
        output_path=OUTPUT_PATH,
        min_basket_size=MIN_BASKET_SIZE,
        max_basket_size=MAX_BASKET_SIZE
    )
