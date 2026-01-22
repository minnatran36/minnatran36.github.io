# Product2Vec: Market Basket Prediction

A Word2Vec-based product recommendation system trained on the Instacart dataset.

## 🎯 Project Overview

This project applies the Word2Vec algorithm (Mikolov et al., 2013) to predict what products a customer will buy next based on their current shopping basket.

**Core Insight:** Products appearing in the same shopping basket are related, just like words appearing in the same sentence are related.

## 📁 Project Structure

```
product2vec/
├── archive/                    # Instacart CSV files (from Kaggle)
│   ├── orders.csv
│   ├── order_products__prior.csv
│   ├── order_products__train.csv
│   ├── products.csv
│   ├── aisles.csv
│   └── departments.csv
│
├── data_processor.py           # Step 1: CSV → baskets.pkl
├── train_model.py              # Step 2: Train Word2Vec
├── export_for_web.py           # Step 3: Export for website
│
├── baskets.pkl                 # (generated) Shopping baskets
├── model_output/               # (generated) Trained model
│   ├── model.bin
│   ├── vectors.json
│   └── product_names.json
│
├── web_data/                   # (generated) Website data
│   ├── products.json
│   ├── similarities.json
│   └── config.json
│
├── index.html                  # Website frontend
├── app.js                      # Website JavaScript
└── README.md                   # This file
```

## 🚀 Quick Start

### Step 1: Set Up on Server

```bash
# SSH into your server
ssh nt1112@agate.cs.unh.edu

# Create project folder
mkdir -p ~/product2vec
cd ~/product2vec

# Upload Instacart data (from your Mac)
# On your Mac: scp -r archive nt1112@agate.cs.unh.edu:~/product2vec/

# Upload Python scripts (from your Mac)
# On your Mac: scp *.py nt1112@agate.cs.unh.edu:~/product2vec/
```

### Step 2: Install Dependencies

```bash
# On the server
pip3 install --user pandas numpy gensim
```

### Step 3: Run the Pipeline

```bash
# Step 1: Process data (creates baskets.pkl)
python3 data_processor.py
# Time: ~2 minutes

# Step 2: Train model (creates model_output/)
python3 train_model.py
# Time: ~10-20 minutes

# Step 3: Export for web (creates web_data/)
python3 export_for_web.py
# Time: ~5-10 minutes
```

### Step 4: Download for Website

```bash
# On your Mac
scp -r nt1112@agate.cs.unh.edu:~/product2vec/web_data ./

# Also get index.html and app.js if not already
scp nt1112@agate.cs.unh.edu:~/product2vec/index.html ./
scp nt1112@agate.cs.unh.edu:~/product2vec/app.js ./
```

### Step 5: Deploy to GitHub Pages

```bash
# Your GitHub Pages repo structure:
your-username.github.io/
├── product2vec/           # or at root
│   ├── index.html
│   ├── app.js
│   └── web_data/
│       ├── products.json
│       ├── similarities.json
│       └── config.json
```

Then visit: `https://your-username.github.io/product2vec/`

## 🧠 How It Works

### The Word2Vec Analogy

| NLP (Text) | Product2Vec (Shopping) |
|------------|------------------------|
| Sentence | Shopping basket |
| Word | Product |
| Words in same sentence = related | Products in same basket = related |
| Word embeddings | Product embeddings |

### Pipeline

```
┌─────────────────────┐
│   Instacart CSVs    │
│   (713MB, 32M rows) │
└──────────┬──────────┘
           │
           ▼  data_processor.py
┌─────────────────────┐
│    baskets.pkl      │
│  (3.2M baskets)     │
└──────────┬──────────┘
           │
           ▼  train_model.py
┌─────────────────────┐
│   vectors.json      │
│ (50K product vectors)│
└──────────┬──────────┘
           │
           ▼  export_for_web.py
┌─────────────────────┐
│    web_data/        │
│ (pre-computed sims) │
└──────────┬──────────┘
           │
           ▼  GitHub Pages
┌─────────────────────┐
│  Interactive Demo   │
└─────────────────────┘
```

### Prediction Method

1. User adds products to basket: `["Pasta", "Tomato Sauce"]`
2. Look up pre-computed similar products for each
3. Aggregate scores (products similar to MULTIPLE items rank higher)
4. Return top recommendations: `["Parmesan", "Garlic", "Ground Beef", ...]`

## 📊 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Skip-gram with Negative Sampling |
| Vector Dimensions | 100 |
| Window Size | 10 |
| Min Count | 5 |
| Negative Samples | 10 |
| Training Data | 3.2M baskets |
| Vocabulary | ~50K products |

## 📚 References

1. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). **Efficient Estimation of Word Representations in Vector Space**. arXiv:1301.3781.

2. **Instacart Online Grocery Shopping Dataset 2017**. Available at: https://www.kaggle.com/c/instacart-market-basket-analysis

## 🎓 Skills Demonstrated

- **Machine Learning**: Word2Vec embeddings
- **Data Engineering**: Processing 32M rows efficiently
- **Full Stack Development**: Python backend + JavaScript frontend
- **Research Application**: Implementing academic paper concepts

## 📝 License

MIT License - Feel free to use for your own portfolio!
