"""
Perfume Recommender — Pro Edition (Python + Streamlit)
=====================================================

What this version adds vs the Starter:
- **Data model**: Uses real CSVs (ratings, catalog, synonyms). Optional web/ETL hook.
- **Note ontology**: Canonicalization + synonym folding + tier weights (top/heart/base).
- **Modeling**:
  - ElasticNet with cross‑validation to learn note preferences (regresses 0–10 ratings).
  - Optional binary “Like” classifier (>=7) with probability calibration.
  - **Target encoding** for brand & family priors (cold‑start help).
  - **Isotonic calibration** to keep predictions in human‑meaningful 0–10 space.
  - **Bootstrap ensembles** for a **confidence** interval on predictions.
- **Explainability**: Per‑note contributions (like SHAP-lite) for any prediction.
- **Fuzzy name resolution**: Type imperfect names; it finds best match.
- **Streamlit UI**: Upload CSVs, train, visualize inclination chart, score & recommend.

Quickstart
----------
1) Install deps:
   pip install pandas numpy scikit-learn matplotlib rapidfuzz streamlit
2) Prepare CSVs (or use the inlined fallbacks):
   - ratings.csv: name,rating (0–10)
   - catalog.csv: name,brand,family,notes_top,notes_heart,notes_base
       * Each notes_* column is a pipe‑separated list (e.g., "bergamot|grapefruit").
   - synonyms.csv: alias,canonical  (e.g., "lily of the valley,muguet")
3) Run training & CLI demo:
   python perfume_recommender_pro.py --train
4) Launch UI:
   streamlit run perfume_recommender_pro.py

Files
-----
This is a single file that works as both a CLI and a Streamlit app.

"""
from __future__ import annotations
import argparse
import io
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

# ---- Streamlit optional import ----
try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# ==========================================================
# 1) Data loading
# ==========================================================

def read_csv_safely(path: str, **kwargs) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        return pd.read_csv(path, **kwargs)
    return None


def load_data(
    ratings_path: Optional[str] = None,
    catalog_path: Optional[str] = None,
    synonyms_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ratings, catalog, synonyms — fallback to embedded samples."""
    # If no paths provided, try default filenames in current directory
    if ratings_path is None:
        ratings_path = "ratings.csv"
    if catalog_path is None:
        catalog_path = "catalog.csv"
    if synonyms_path is None:
        synonyms_path = "synonyms.csv"
    
    ratings = read_csv_safely(ratings_path)
    catalog = read_csv_safely(catalog_path)
    synonyms = read_csv_safely(synonyms_path)

    if ratings is None:
        ratings = pd.DataFrame([
    # Designer / niche full bottles
    {"name": "Terre d’Hermès Vetiver", "rating": 9.5},
    {"name": "Théorème Rue Broca", "rating": 9.0},
    {"name": "Nishane Vetiver", "rating": 9.5},
    {"name": "Armaf Club de Nuit Untold", "rating": 9.5},
    {"name": "Versace Eros", "rating": 7.5},
    {"name": "YSL Y", "rating": 7.5},
    {"name": "Paco Rabanne 1 Million Elixir", "rating": 7.0},
    {"name": "Davidoff Cool Water", "rating": 8.0},
    {"name": "Rasasi Hawas", "rating": 6.5},

    # Studio West
    {"name": "Studio West Bora Bora", "rating": 9.0},
    {"name": "Studio West Man", "rating": 9.0},

    # Lattafa / Ajmal / Afnan etc.
    {"name": "Khamrah Kahwah", "rating": 7.0},
    {"name": "Silver Shade", "rating": 8.0},
    {"name": "Afnan Snoi", "rating": 7.0},
    {"name": "Afnan Rare Reef", "rating": 9.0},
    {"name": "Afnan 9pm", "rating": 3.0},
    {"name": "Lattafa Kingdom", "rating": None},   # not rated explicitly
    {"name": "Maison Alhambra Yeah Man", "rating": 8.0},

    # Inspired / clones
    {"name": "Fragrance World Mark & Victor", "rating": 7.0},
    {"name": "Club de Nuit Sillage", "rating": 8.0},
    {"name": "Blu by Ahmad", "rating": 7.0},

    # Scentedelic decants
    {"name": "1200 Micrograms", "rating": 9.5},
    {"name": "Time & Space 2.0", "rating": 9.0},
    {"name": "Komorebi (Mridul)", "rating": 8.0},
    {"name": "Patchouli Rum (Mridul)", "rating": 8.0},
    {"name": "Dusk Till Dawn (Mridul)", "rating": 7.5},
    {"name": "Ambergris Gold (Mridul)", "rating": None},  # not rated explicitly yet
    {"name": "Sacred Temptation (Mridul)", "rating": 8.0},
    {"name": "Santal 1 (Mridul)", "rating": None},  # disliked, but no numeric given
    {"name": "Nile to Milky Way (Mridul)", "rating": 8.0},
    {"name": "Labyrinth Potion (Mridul)", "rating": 7.0},
    {"name": "Bourbon Silk (Mridul)", "rating": 8.5},

    # Other Scentedelic
    {"name": "Petals of Desire 8.0", "rating": 8.0},
    {"name": "Bazinga Absolu", "rating": 7.0},
    {"name": "Magic Mushroom", "rating": 8.0},
    {"name": "Dopamine Absolu 2.0", "rating": 8.0},
    {"name": "Long Island Ice Tea 2.0", "rating": 8.0},
    {"name": "Monkey D. Luffy", "rating": 8.5},
    {"name": "Call of da Wild", "rating": None},   # not numerically rated
    {"name": "Equilibrium Aqua", "rating": 9.0},
    {"name": "Fractal Cologne", "rating": None},  # not yet scored
    {"name": "Kakashi 33", "rating": None},       # not yet scored
    {"name": "Yellow Flash", "rating": None},     # not yet scored
    {"name": "Timeless 9.0", "rating": 7.0},
    {"name": "Timeless 2.0", "rating": 7.0},

    # Raahi Parfums
    {"name": "Gulab (Raahi)", "rating": 8.5},
    {"name": "Vétiveria (Raahi)", "rating": 6.5},
    {"name": "Dahn al Oud (Raahi)", "rating": 7.0},
    {"name": "Champa Muse (Raahi)", "rating": 8.0},
    {"name": "Sambac Noix (Raahi)", "rating": 7.5},

    # Ahmed Al Maghribi
    {"name": "Kaaf (Ahmed Al Maghribi)", "rating": 6.5},
    {"name": "Okami Umi", "rating": 6.5},

    # Others
    {"name": "Albait Prive", "rating": 2.0},
])
    if catalog is None:
        # Minimal embedded catalog with tiers. Expand with your CSV or ETL.
        catalog = pd.DataFrame([
            {"name": "Terre d’Hermès Vetiver", "brand": "Hermès", "family": "Woody", "notes_top": "grapefruit|citrus", "notes_heart": "vetiver", "notes_base": "oakmoss|patchouli|woody"},
            {"name": "Nishane Vetiver", "brand": "Nishane", "family": "Woody", "notes_top": "citrus", "notes_heart": "vetiver|spice", "notes_base": "woody"},
            {"name": "Théorème Rue Broca", "brand": "Rue Broca", "family": "Amber Woody", "notes_top": "citrus", "notes_heart": "spice", "notes_base": "amber|woody"},
            {"name": "Armaf Club de Nuit Untold", "brand": "Armaf", "family": "Amber Floral", "notes_top": "saffron|jasmine", "notes_heart": "amberwood|cedar", "notes_base": "moss"},
            {"name": "1200 Micrograms", "brand": "Scentedelic", "family": "Citrus Amber", "notes_top": "grapefruit|citrus", "notes_heart": "amber", "notes_base": "woody|musky"},
            {"name": "Rare Reef", "brand": "Afnan", "family": "Citrus Aromatic", "notes_top": "orange|citrus", "notes_heart": "aromatic", "notes_base": "woody"},
            {"name": "Khamrah Kahwah", "brand": "Lattafa", "family": "Amber Spicy", "notes_top": "coffee|spice", "notes_heart": "amber", "notes_base": "sweet"},
            {"name": "Silver Shade", "brand": "Ajmal", "family": "Citrus Musky", "notes_top": "citrus|fruity", "notes_heart": "musk", "notes_base": "woody"},
            {"name": "Studio West Bora Bora", "brand": "Studio West", "family": "Aromatic Citrus", "notes_top": "citrus|green", "notes_heart": "aromatic", "notes_base": ""},
            {"name": "Studio West Man", "brand": "Studio West", "family": "Chypre", "notes_top": "bergamot", "notes_heart": "", "notes_base": "oakmoss"},
            {"name": "Cool Water", "brand": "Davidoff", "family": "Aromatic Aquatic", "notes_top": "aquatic|lavender", "notes_heart": "musk", "notes_base": "woody"},
            {"name": "Afnan 9pm", "brand": "Afnan", "family": "Amber Vanilla", "notes_top": "sweet|vanilla", "notes_heart": "amber", "notes_base": "warm spicy"},
            {"name": "Komorebi (Mridul)", "brand": "Mridul", "family": "Green Citrus", "notes_top": "fig|bergamot", "notes_heart": "neroli|african orange", "notes_base": "mimosa|amber|cedar"},
            {"name": "Time & Space 2.0", "brand": "Scentedelic", "family": "Citrus Aromatic", "notes_top": "citrus", "notes_heart": "aromatic", "notes_base": "woody|ambroxan"},
            {"name": "Gulab (Raahi)", "brand": "Raahi", "family": "Floral", "notes_top": "rose", "notes_heart": "floral|resin", "notes_base": "cedarwood|musk"},
            {"name": "Vétiveria (Raahi)", "brand": "Raahi", "family": "Woody", "notes_top": "bergamot|lemon", "notes_heart": "incense|woody", "notes_base": "vetiver|earthy"},
            {"name": "Dahn al Oud (Raahi)", "brand": "Raahi", "family": "Amber Woody", "notes_top": "saffron|warm spices", "notes_heart": "oud|balsamic", "notes_base": "woods|amber|musk"},
            {"name": "Champa Muse (Raahi)", "brand": "Raahi", "family": "Floral", "notes_top": "magnolia|citrus", "notes_heart": "champa|muguet", "notes_base": "soft woods|musk"},
            {"name": "Kaaf (Ahmed Al Maghribi)", "brand": "Ahmed Al Maghribi", "family": "Aromatic Aquatic", "notes_top": "red fruits|watermelon|lavender|orange", "notes_heart": "lotus|jasmine|muguet|sea accord", "notes_base": "sandalwood|ambroxan|white musk"},
        ])

    if synonyms is None:
        synonyms = pd.DataFrame([
            {"alias": "lily of the valley", "canonical": "muguet"},
            {"alias": "soft woods", "canonical": "woody"},
            {"alias": "woods", "canonical": "woody"},
            {"alias": "sicilian orange", "canonical": "orange"},
            {"alias": "grapefruit", "canonical": "grapefruit"},
            {"alias": "lavandin", "canonical": "lavender"},
        ])

    return ratings, catalog, synonyms


# ==========================================================
# 2) Note normalization & feature building
# ==========================================================

def make_synonym_map(syn_df: pd.DataFrame) -> Dict[str, str]:
    mp = {}
    for _, r in syn_df.iterrows():
        a = str(r["alias"]).strip().lower()
        c = str(r["canonical"]).strip().lower()
        if a:
            mp[a] = c
    return mp


def normalize_notes(note_str: str, syn_map: Dict[str, str]) -> List[str]:
    if not isinstance(note_str, str) or not note_str:
        return []
    raw = [x.strip().lower() for x in note_str.split("|") if x.strip()]
    out = []
    for n in raw:
        n = syn_map.get(n, n)
        # drop purely descriptive tokens that aren't notes
        if n in {"soft", "warm", "dark"}:  # tweakable
            continue
        out.append(n)
    return out


TIER_WEIGHTS = {"top": 0.9, "heart": 1.0, "base": 1.2}


def perfume_row_to_text(row: pd.Series, syn_map: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
    """Return a weighted note bag + per-note weights used (for explanations)."""
    notes = []
    per_note_w = {}
    for tier, col in [("top", "notes_top"), ("heart", "notes_heart"), ("base", "notes_base")]:
        lst = normalize_notes(row.get(col, ""), syn_map)
        w = TIER_WEIGHTS[tier]
        for n in lst:
            notes.append(n)
            per_note_w[n] = per_note_w.get(n, 0.0) + w
    return " ".join(notes), per_note_w


def build_feature_corpus(catalog: pd.DataFrame, syn_map: Dict[str, str]) -> Tuple[pd.Series, List[Dict[str, float]]]:
    texts = []
    weights = []
    for _, r in catalog.iterrows():
        t, w = perfume_row_to_text(r, syn_map)
        texts.append(t)
        weights.append(w)
    return pd.Series(texts, index=catalog.index), weights


# ==========================================================
# 3) Modeling — ElasticNet + calibration + target encodings
# ==========================================================

@dataclass
class TrainedModel:
    catalog: pd.DataFrame
    syn_map: Dict[str, str]
    vec: TfidfVectorizer
    reg: ElasticNet
    iso: Optional[IsotonicRegression]
    brand_means: Dict[str, float]
    family_means: Dict[str, float]
    note_coef: Dict[str, float]  # TF-IDF feature name -> coef
    bootstrap_preds: Optional[np.ndarray]  # for CI


def fit_model(ratings: pd.DataFrame, catalog: pd.DataFrame, synonyms: pd.DataFrame, n_bootstrap: int = 40) -> TrainedModel:
    # Resolve synonyms & build TF-IDF
    syn_map = make_synonym_map(synonyms)
    texts, _ = build_feature_corpus(catalog, syn_map)
    # Use individual words as features instead of multi-word phrases
    vec = TfidfVectorizer(token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b', min_df=1)
    X_text = vec.fit_transform(texts)

    # Join ratings to catalog
    df = ratings.merge(catalog, on="name", how="left")
    
    # Remove rows where catalog info is missing
    initial_len = len(df)
    df = df.dropna(subset=['brand', 'family'])
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} rows due to missing catalog info")
    
    # Remove rows with missing ratings (NaN values)
    initial_len = len(df)
    df = df.dropna(subset=['rating'])
    if len(df) < initial_len:
        print(f"Warning: Dropped {initial_len - len(df)} rows due to missing ratings")
    
    if len(df) == 0:
        raise ValueError("No valid training data remaining after filtering. Please check your ratings and catalog data.")

    # Brand/family target encoding (simple mean encoding)
    global_mean = df["rating"].mean()
    brand_means = df.groupby("brand")["rating"].mean().to_dict()
    family_means = df.groupby("family")["rating"].mean().to_dict()

    def enc_brand(x):
        return [brand_means.get(b, global_mean) for b in x]

    def enc_family(x):
        return [family_means.get(f, global_mean) for f in x]

    brand_feat = np.array(enc_brand(df["brand"].fillna("?"))).reshape(-1, 1)
    family_feat = np.array(enc_family(df["family"].fillna("?"))).reshape(-1, 1)

    # Get text features for the perfumes that have ratings
    # Find the catalog row indices that correspond to perfumes in df
    catalog_indices = []
    valid_df_indices = []
    for i, name in enumerate(df['name']):
        # Find the catalog row with this perfume name
        catalog_matches = catalog[catalog['name'] == name]
        if len(catalog_matches) > 0:
            catalog_idx = catalog_matches.index[0]
            catalog_indices.append(catalog_idx)
            valid_df_indices.append(i)
    
    # Filter df to only include rows we found in catalog
    df = df.iloc[valid_df_indices].reset_index(drop=True)
    brand_feat = brand_feat[valid_df_indices]
    family_feat = family_feat[valid_df_indices]
    
    # Get the corresponding text features
    X_text_aligned = X_text[catalog_indices]

    # Combine features
    X = np.hstack([X_text_aligned.toarray(), brand_feat, family_feat])
    y = df["rating"].values

    # ElasticNet CV
    alphas = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    l1s = [0.0, 0.1, 0.3, 0.5, 0.8]
    best = None
    best_mae = 1e9
    kf = KFold(n_splits=min(5, len(df)), shuffle=True, random_state=42)
    for a in alphas:
        for l1 in l1s:
            maes = []
            for tr, va in kf.split(X):
                reg = ElasticNet(alpha=a, l1_ratio=l1, random_state=42, max_iter=10000)
                reg.fit(X[tr], y[tr])
                pred = np.clip(reg.predict(X[va]), 0, 10)
                maes.append(mean_absolute_error(y[va], pred))
            m = float(np.mean(maes))
            if m < best_mae:
                best_mae = m
                best = (a, l1)
    a_opt, l1_opt = best
    reg = ElasticNet(alpha=a_opt, l1_ratio=l1_opt, random_state=42, max_iter=10000)
    reg.fit(X, y)

    # Isotonic calibration (maps reg output to 0–10 more faithfully)
    raw_pred = reg.predict(X)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=10.0)
    iso.fit(raw_pred, y)

    # Note coefficients
    feature_names = list(vec.get_feature_names_out()) + ["brand_te", "family_te"]
    coefs = reg.coef_
    note_coef = {feature_names[i]: float(coefs[i]) for i in range(min(len(coefs), len(feature_names)))}

    # Bootstrap predictions for confidence estimation
    rng = np.random.RandomState(123)
    preds = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(X), size=len(X), replace=True)
        reg_b = ElasticNet(alpha=a_opt, l1_ratio=l1_opt, random_state=42, max_iter=10000)
        reg_b.fit(X[idx], y[idx])
        preds.append(iso.predict(reg_b.predict(X)))
    bootstrap_preds = np.vstack(preds) if preds else None

    return TrainedModel(
        catalog=catalog.reset_index(drop=True),
        syn_map=syn_map,
        vec=vec,
        reg=reg,
        iso=iso,
        brand_means=brand_means,
        family_means=family_means,
        note_coef=note_coef,
        bootstrap_preds=bootstrap_preds,
    )


def save_model(model: TrainedModel, filepath: str = "model.pkl"):
    """Save trained model to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")


def load_model(filepath: str = "model.pkl") -> Optional[TrainedModel]:
    """Load trained model from disk."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None


def should_retrain(model_path: str = "model.pkl") -> bool:
    """Check if model needs retraining based on file modification times."""
    if not os.path.exists(model_path):
        return True
    
    model_time = os.path.getmtime(model_path)
    
    # Check if any data files are newer than the model
    data_files = ["ratings.csv", "catalog.csv", "synonyms.csv"]
    for file in data_files:
        if os.path.exists(file) and os.path.getmtime(file) > model_time:
            print(f"Model is outdated due to changes in {file}")
            return True
    
    return False


def get_or_train_model(
    ratings: pd.DataFrame, 
    catalog: pd.DataFrame, 
    synonyms: pd.DataFrame,
    force_retrain: bool = False
) -> TrainedModel:
    """Get existing model or train a new one if needed."""
    model_path = "model.pkl"
    
    if not force_retrain and not should_retrain(model_path):
        model = load_model(model_path)
        if model is not None:
            return model
    
    print("Training new model...")
    print(f"Processing {len(catalog)} perfumes with {len(ratings)} ratings...")
    model = fit_model(ratings, catalog, synonyms)
    save_model(model, model_path)
    return model


# ==========================================================
# 4) Scoring, recommending & explanations
# ==========================================================

def _perfume_index_by_name(name: str, catalog: pd.DataFrame) -> Optional[int]:
    """Improved fuzzy matching that handles data quality issues."""
    choices = list(catalog["name"].values)
    
    # Try multiple matching strategies to handle different name formats
    strategies = [
        (name, fuzz.WRatio),
        (name.lower(), fuzz.WRatio), 
        (name.replace(' ', '-').lower(), fuzz.WRatio),
        (name.replace(' ', '').lower(), fuzz.partial_ratio),
    ]
    
    best_match = None
    best_score = 0
    
    for query, scorer in strategies:
        match, score, _ = process.extractOne(query, choices, scorer=scorer)
        
        # Prefer longer matches to avoid short name pollution
        # and require a reasonable minimum score
        if score > best_score and len(match) > 3 and score >= 70:
            best_match = match
            best_score = score
    
    if best_match and best_score >= 70:
        return int(catalog.index[catalog["name"] == best_match][0])
    return None


def _vectorize_row(row: pd.Series, model: TrainedModel) -> np.ndarray:
    text, _ = perfume_row_to_text(row, model.syn_map)
    X_text = model.vec.transform([text]).toarray()
    brand_te = np.array([[model.brand_means.get(row.get("brand"), np.nan)]])
    family_te = np.array([[model.family_means.get(row.get("family"), np.nan)]])
    # Fallback to global mean if unseen
    if np.isnan(brand_te).any():
        brand_te[:] = np.nanmean(list(model.brand_means.values()))
    if np.isnan(family_te).any():
        family_te[:] = np.nanmean(list(model.family_means.values()))
    return np.hstack([X_text, brand_te, family_te])


def predict_like_score(name: str, model: TrainedModel) -> Tuple[float, Tuple[float, float], Dict[str, float]]:
    idx = _perfume_index_by_name(name, model.catalog)
    if idx is None:
        return float("nan"), (float("nan"), float("nan")), {}
    row = model.catalog.iloc[idx]
    X = _vectorize_row(row, model)
    raw = model.reg.predict(X)[0]
    pred = float(model.iso.predict([raw])[0]) if model.iso else float(np.clip(raw, 0, 10))

    # Confidence via bootstrap variance on training distribution (proxy)
    if model.bootstrap_preds is not None:
        bs_std = float(np.std(model.bootstrap_preds))
        ci = (max(0.0, pred - 1.28 * bs_std), min(10.0, pred + 1.28 * bs_std))
    else:
        ci = (max(0.0, pred - 0.6), min(10.0, pred + 0.6))

    # Explain note contributions (coef * tfidf weight)
    text, _ = perfume_row_to_text(row, model.syn_map)
    x_text = model.vec.transform([text]).toarray()[0]
    expl = {}
    vocab = list(model.vec.get_feature_names_out())
    for i, val in enumerate(x_text):
        if val != 0:
            note = vocab[i]
            w = model.reg.coef_[i]
            if abs(w) > 1e-6:
                expl[note] = float(w * val)
    expl = dict(sorted(expl.items(), key=lambda t: abs(t[1]), reverse=True)[:15])

    return pred, ci, expl


def recommend_top_k(candidates: Iterable[str], model: TrainedModel, k: int = 10) -> List[Tuple[str, float]]:
    scored = []
    for name in candidates:
        s, _, _ = predict_like_score(name, model)
        if not math.isnan(s):
            scored.append((name, s))
    return sorted(scored, key=lambda t: t[1], reverse=True)[:k]


def top_bottom_notes(model: TrainedModel, top_n: int = 15) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    items = [(n, w) for n, w in model.note_coef.items() if n not in {"brand_te", "family_te"}]
    items.sort(key=lambda t: t[1], reverse=True)
    top = items[:top_n]
    bottom = items[-top_n:]
    return top, bottom


# ==========================================================
# 5) Streamlit UI
# ==========================================================

def run_streamlit_app():
    st.set_page_config(page_title="Perfume Recommender — Pro", layout="wide")
    st.title("Perfume Recommender — Pro Edition")

    st.sidebar.header("Data Inputs")
    ratings_file = st.sidebar.file_uploader("ratings.csv", type=["csv"])
    catalog_file = st.sidebar.file_uploader("catalog.csv", type=["csv"])
    synonyms_file = st.sidebar.file_uploader("synonyms.csv", type=["csv"])

    ratings, catalog, synonyms = load_data()
    if ratings_file:
        ratings = pd.read_csv(ratings_file)
    if catalog_file:
        catalog = pd.read_csv(catalog_file)
    if synonyms_file:
        synonyms = pd.read_csv(synonyms_file)

    st.sidebar.write("**Records:**", len(catalog), "perfumes;", len(ratings), "ratings")
    
    # App status for cloud deployment
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 App Status")
    
    # Check if model file exists (indicates warm state)
    import os
    if os.path.exists("model.pkl"):
        st.sidebar.success("✅ Model cached (fast mode)")
        # Get file age
        import time
        age = time.time() - os.path.getmtime("model.pkl")
        if age < 3600:  # Less than 1 hour
            st.sidebar.info(f"🕐 Cached {int(age/60)} min ago")
        else:
            st.sidebar.info(f"🕐 Cached {int(age/3600)} hrs ago")
    else:
        st.sidebar.warning("⏳ Cold start (training required)")
    
    # Performance tip
    if not os.path.exists("model.pkl"):
        st.sidebar.info("💡 **Tip**: First load takes 2-3 minutes. Subsequent loads are instant!")

    with st.spinner("Loading model... (This may take 2-3 minutes on first load)"):
        try:
            # Show progress for cloud users
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text('🔍 Checking for cached model...')
            progress_bar.progress(10)
            
            model = get_or_train_model(ratings, catalog, synonyms)
            
            progress_bar.progress(100)
            status_text.text('✅ Model ready!')
            
            # Clear progress indicators after a moment
            import time
            time.sleep(1)
            progress_bar.progress(100)  # Keep at 100% briefly
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error("Model loading/training failed. See details below.")
            st.exception(e)
            st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Fragrance Inclination (top/negative notes)")
        top, bottom = top_bottom_notes(model, 15)
        t_labels, t_vals = zip(*top) if top else ([], [])
        b_labels, b_vals = zip(*bottom) if bottom else ([], [])
        st.bar_chart(pd.DataFrame({"weight": list(t_vals)}, index=list(t_labels)))
        st.bar_chart(pd.DataFrame({"weight": list(b_vals)}, index=list(b_labels)))

    with col2:
        st.subheader("Score a Perfume")
        name_in = st.text_input("Enter perfume name")
        if name_in:
            pred, ci, expl = predict_like_score(name_in, model)
            st.markdown(f"**Predicted rating:** {pred:.2f} (CI ~ {ci[0]:.2f}–{ci[1]:.2f})")
            if expl:
                st.write("Top note contributions (+/−):")
                st.dataframe(pd.DataFrame(sorted(expl.items(), key=lambda t: -abs(t[1])), columns=["note","contribution"]))

    st.subheader("Recommend from Catalog")
    n = st.slider("How many?", min_value=5, max_value=30, value=10)
    recs = []
    for nm in catalog["name"].tolist():
        s, _, _ = predict_like_score(nm, model)
        if not math.isnan(s):
            recs.append((nm, s))
    recs = sorted(recs, key=lambda t: t[1], reverse=True)[:n]
    st.dataframe(pd.DataFrame(recs, columns=["name","predicted_rating"]))
    
    # Footer with deployment info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
        🌟 <strong>Perfume Recommender</strong> | 
        Deployed on Streamlit Cloud | 
        <a href='https://github.com/parasgupta/perfume-recommender' target='_blank'>View Source</a>
        <br>
        💡 <em>App sleeps after 15min inactivity. First wake-up may take 2-3 minutes for model training.</em>
    </div>
    """, unsafe_allow_html=True)


# ==========================================================
# 6) CLI
# ==========================================================

def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--train", action="store_true", help="Train and print summary")
    ap.add_argument("--retrain", action="store_true", help="Force retrain model")
    ap.add_argument("--ratings", type=str, default=None)
    ap.add_argument("--catalog", type=str, default=None)
    ap.add_argument("--synonyms", type=str, default=None)
    ap.add_argument("--score", type=str, default=None, help="Perfume name to score")
    # Streamlit passes its own args; don't crash on unknowns
    args, _unknown = ap.parse_known_args()

    ratings, catalog, synonyms = load_data(args.ratings, args.catalog, args.synonyms)
    
    # Use cached model unless force retrain or explicit training requested
    force_retrain = args.retrain or args.train
    model = get_or_train_model(ratings, catalog, synonyms, force_retrain)

    if args.train:
        print(f"Training set size: {len(ratings)} ratings; Catalog: {len(catalog)} items")
        top, bottom = top_bottom_notes(model, 15)
        print("Top positive notes:")
        for n, w in top:
            print(f"  {n:18s} {w:+.3f}")
        print("Most negative notes:")
        for n, w in bottom:
            print(f"  {n:18s} {w:+.3f}")

    if args.score:
        pred, ci, expl = predict_like_score(args.score, model)
        print(f"{args.score}\nPredicted rating: {pred:.2f} (CI ~ {ci[0]:.2f}–{ci[1]:.2f})")
        if expl:
            print("Top contributions:")
            for n, v in sorted(expl.items(), key=lambda t: -abs(t[1])):
                print(f"  {n:18s} {v:+.3f}")

    if not args.train and not args.score and _HAS_ST:
        # If no CLI action requested, run the Streamlit app when available.
        run_streamlit_app()


if __name__ == "__main__":
    main()
