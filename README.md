# 🛒 Trendyol E-Commerce Hackathon 2025 — CatBoost Baseline Solution

> **Team:** CatBoost Baseline  
> **Public LB:** `0.70423` &nbsp;|&nbsp; **Private LB:** `0.70590`  
> **Competition:** [Trendyol E-Ticaret Hackathonu 2025 — Kaggle](https://www.kaggle.com/competitions/trendyol-e-ticaret-hackathonu-2025-kaggle)

---

## 📌 Task Overview

The competition goal was to predict **which products a user will purchase (order) or click** within a search session on Trendyol, Turkey's leading e-commerce platform. Given a session with a search query and a set of candidate products, the model must rank them so that purchased items appear at the top.

The evaluation metric is a custom **Group AUC** (`trendyol_metric_group_auc`) that measures ranking quality at the session level, rewarding models that correctly distinguish ordered and clicked items from the rest.

---

## 🗂️ Data Sources

The competition provided rich multi-table data:

| File | Description |
|---|---|
| `train_sessions.parquet` / `test_sessions.parquet` | Session-level rows with user, content, and query information |
| `content/metadata.parquet` | Product attributes, categories, price info, review and rating counts |
| `content/price_rate_review_data.parquet` | Daily price, discount, and review snapshots per product |
| `user/metadata.parquet` | User demographics (age, gender, tenure) |
| `content/search_log.parquet` | Historical impression and click counts for products in search |
| `content/sitewide_log.parquet` | Sitewide click, cart, favorite, and order counts per product |
| `user/search_log.parquet` / `user/sitewide_log.parquet` | Same aggregates at the user level |
| `term/search_log.parquet` | Query-level impression and click statistics |
| `content/top_terms_log.parquet` | Per-product, per-query impression/click history |
| `user/fashion_search_log.parquet` / `user/fashion_sitewide_log.parquet` | Fine-grained user–product interaction history |
| `user/top_terms_log.parquet` | Per-user, per-query search history |

---

## ⚙️ Solution Pipeline

### 1. Feature Engineering

Feature engineering is the core of this solution. All features were built using **Polars** for fast, memory-efficient processing.

#### 1.1 Timestamp Features
Extracted `hour_of_day`, `day_of_week`, `month`, and `year` from session timestamps. Created binary flags `is_weekend` and `is_holiday_season` (November–December).

#### 1.2 Recency Weighting
Applied **exponential decay** to time-series log data before aggregation, using a half-life of 14 days:

```
decay_weight = exp(-log(2) / 14 × days_since_event)
```

This up-weights recent impressions and clicks when computing CTR/CVR statistics, capturing trend shifts in product popularity.

#### 1.3 Content-Level CTR & CVR Features
From search and sitewide logs, computed per-product statistics:
- Sum and mean impression/click counts
- Raw CTR (`clicks / impressions`) and CVR (`orders / clicks`)
- **Recency-weighted** variants of all the above

#### 1.4 User-Level Engagement Features
Symmetrically, computed per-user search impressions, clicks, CTR, and sitewide CVR, both raw and recency-weighted.

#### 1.5 Query (Term) Level Features
Per search term: total impressions, clicks, and CTR across the entire catalog — both historical totals and recency-weighted.

#### 1.6 Query × Product Interaction Features
From `content/top_terms_log`, computed per `(content_id, search_term)` pair:
- Impression/click sums and CTR (historical and recency-weighted)
- **`ct_over_term_ctr`**: ratio of the product's term-specific CTR over the global term CTR, capturing whether a product outperforms the average for a given query.

#### 1.7 User × Product & User × Query Interaction Features
From fashion logs and top-terms logs, captured per `(user_id, content_id)` and per `(user_id, search_term)` behavior, including sitewide CVR and search CTR.

#### 1.8 CV Tags — TF-IDF Term Match
Tokenized each product's `cv_tags` field and computed per-token IDF scores across the catalog. For each session row, created:
- `term_in_cv_tags`: binary flag indicating if the search term appears in the product's tags
- `term_in_cv_tags_idf`: IDF weight of the matching token, rewarding rare but precise tag matches

#### 1.9 Derived Product Features
From price, review, and rating fields:
- `discount_rate`, `price_ratio`, `absolute_discount`
- `engagement_score` = `rating_avg × rating_count × review_count`
- `media_review_ratio`, `avg_options_per_attribute`, `total_attributes`
- `content_age_days` (days since product creation)
- Laplace-smoothed CTR/CVR estimates (e.g., `(clicks + 0.5) / (impressions + 1.0)`)

#### 1.10 Session-Level Aggregates
Computed within each session:
- `session_length`, price stats (`avg_price`, `min_price`, `max_price`, `price_std`), discount and review stats, average rating and engagement
- Diversity measures: `session_unique_items`, `session_unique_cat1`, `session_unique_leaf`

#### 1.11 Session-Relative Features
For each item in a session, computed its position relative to session-level statistics:
- `price_vs_session_avg`, `discount_vs_session_avg`, `reviews_vs_session_avg`, etc.
- `price_percentile_in_session`: normalized price rank within the session's price range
- `category_share_in_session`: share of the session occupied by the product's top-level category

#### 1.12 Category-Level Statistics and Z-Scores
Computed mean and standard deviation of price, discount, reviews, rating, and engagement at both **level-1 category** and **leaf category** granularity (from the training data). Then derived ratio and z-score features for each product against its category baseline (e.g., `price_vs_cat1_avg`, `rating_z_leaf`).

#### 1.13 Content-Frequency Features
Counted how many times each product appeared in training sessions, and computed per-content average price, discount, and engagement.

---

### 2. Modeling

Two separate **CatBoost** classifiers were trained:
- **Order model**: predicts whether a product will be purchased (`ordered = 1`)
- **Click model**: predicts whether a product will be clicked (`clicked = 1`)

Both models natively handle the categorical features (`user_id`, `content_id`, `search_term`, category hierarchy) without manual encoding.

#### Hyperparameters (Optuna-tuned, 10 trials)

| Parameter | Order Model | Click Model |
|---|---|---|
| `iterations` | 1924 | 986 |
| `learning_rate` | 0.02451 | 0.03031 |
| `depth` | 4 | 7 |
| `l2_leaf_reg` | 4.44 | 15.56 |
| `subsample` | 0.759 | 0.905 |
| `bootstrap_type` | Bernoulli | Bernoulli |
| `auto_class_weights` | Balanced | Balanced |

#### Cross-Validation Strategy

Used **10-fold GroupKFold** grouped by `session_id` to prevent data leakage between folds (all rows belonging to the same session stay in the same split).

In each fold:
1. Train both models on the training split.
2. Generate OOF predictions for the validation split.
3. Search for the optimal blend weight `w` in `[0.50, 0.80]` (step 0.02) that maximizes the competition metric on the fold's validation data.

After all folds, a global blend weight search over `[0.40, 0.90]` (step 0.01) is performed on the full OOF predictions.

#### Final Ensemble Score

```
final_score = w × pred_ordered + (1 − w) × pred_clicked
```

Items within each session are ranked by `final_score` in descending order to produce the submission.

#### Final Model Training

Final models are trained on the full training data. The number of iterations is clipped to the mean of per-fold best iterations (within `[800, 2200]`) to avoid overfitting.

---

### 3. Submission Ensemble

The highest-scoring submission was produced by ensembling **6 submission files** (generated from slight variations of the same pipeline) using **Reciprocal Rank Fusion (RRF)**.

#### RRF Formula

For each `(session, item)` pair across all submissions:

```
RRF_score(item) = Σ  weight_i / (k + rank_i(item))
```

with `k = 60`.

Items are re-ranked by their aggregated RRF score within each session.

#### Ensemble Methods Explored

| Method | Description |
|---|---|
| **RRF k=60** ⭐ Best | Reciprocal Rank Fusion with k=60 — **Public LB: 0.70423, Private LB: 0.70590** |
| RRF k=50 | Slightly sharper top-rank weighting |
| Exp α=0.05 | Exponential decay score: `exp(-0.05 × (rank - 1))` |
| MedRank λ=50 | Minimizes sum of ranks; missing items penalized by +50 |
| Weighted RRF k=60 | RRF with weight 1.2 on the blended submission |

#### Running the Ensemble

Save the ensemble script as `ensemble_submissions.py` and run:

```bash
python3 ensemble_submissions.py \
  --subs sub1.csv sub2.csv sub3.csv sub4.csv sub5.csv sub6.csv \
  --method rrf \
  --k 60 \
  --output ensemble_rrf_k60.csv
```

Ensemble files are available as a Kaggle dataset:

```bash
kaggle datasets download -d ahmeterdempamuk/catboost-baseline-trendyolhackathon-ensemble-files
```

---

## 📦 Requirements

```
catboost
lightgbm
optuna
polars
shap
scikit-learn
pandas
numpy
```

Install with:

```bash
pip install catboost lightgbm optuna polars shap scikit-learn pandas numpy
```

---

## 🚀 Reproducing the Solution

1. Set up Kaggle API credentials and download the competition data:
   ```bash
   kaggle competitions download -c trendyol-e-ticaret-hackathonu-2025-kaggle
   unzip trendyol-e-ticaret-hackathonu-2025-kaggle.zip -d ./data
   ```

2. Set the data path in the notebook:
   ```python
   DATA_PATH = "./data"
   ```

3. Run all cells in the notebook top-to-bottom. GPU is recommended (`CB_TASK_TYPE = "GPU"`).

4. Run the ensemble script on the generated submission files.

---

## 🏆 Results

| Split | Score |
|---|---|
| **Public Leaderboard** | **0.70423** |
| **Private Leaderboard** | **0.70590** |

The near-identical public/private scores indicate a stable, well-generalizing solution with no significant overfitting to the public leaderboard.

---

## 💡 Key Takeaways

- **Recency-weighted aggregations** effectively captured short-term trend shifts in product and query popularity.
- **Cross-granularity ratio features** (product vs. session, product vs. category, product vs. term) gave the model strong relative-ranking signals within each context.
- **CV tag IDF matching** provided a lightweight relevance signal between the product's visual/semantic tags and the user's search query.
- **Dual-target training** (order + click) with per-fold weight optimization yielded better ranking than a single target.
- **Submission-level RRF ensemble** provided a consistent, reliable boost over any single model run.

---

## 🙏 Acknowledgements

We would like to sincerely thank **TEKNOFEST** and the **Trendyol Data Science team** for organizing this competition and giving us the opportunity to work with real-world e-commerce data. This was an incredibly educational and competitive experience over a two-week period.

— *Team CatBoost Baseline*
