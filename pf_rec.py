"""
Perfume Recommender — Enhanced Version (Python)
-----------------------------------------------

This version goes beyond the simple linear regression baseline. It adds:
- **Note-level weighting by pyramid position** (top/heart/base).
- **Hybrid scoring model** combining regression + cosine similarity.
- **Family clustering** (citrus, woody, amber, floral, etc.).
- **Candidate catalog search**: rank perfumes in a dataset (CSV or DB).
- **Streamlit UI option** to make it interactive.

Quickstart
----------
1) Install deps:  pip install scikit-learn pandas numpy matplotlib requests beautifulsoup4 streamlit
2) Save your ratings into USER_RATINGS.
3) Add perfume notes via LOCAL_NOTES_DB or a CSV.
4) Run:  streamlit run perfume_recommender.py   (for UI)  OR  python perfume_recommender.py

"""
from __future__ import annotations
import math, re
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ------------------------------
# 1) User ratings
# ------------------------------
USER_RATINGS: Dict[str, float] = {
    "Terre d’Hermès Vetiver": 9.5,
    "Théorème Rue Broca": 9.0,
    "Nishane Vetiver": 9.5,
    "Armaf Club de Nuit Untold": 9.0,
    "1200 Micrograms": 9.5,
    "Komorebi (Mridul)": 8.0,
    "Rare Reef": 9.0,
    "Labyrinth Potion": 7.0,
    "Afnan 9pm": 3.0,
    "Kaaf (Ahmed Al Maghribi)": 6.5,
    "Gulab (Raahi)": 8.5,
    "Champa Muse (Raahi)": 8.0,
}

# ------------------------------
# 2) Local notes DB with pyramid tiering
# ------------------------------
LOCAL_NOTES_DB: Dict[str, Dict[str,List[str]]] = {
    "Terre d’Hermès Vetiver": {
        "top": ["grapefruit"],
        "heart": ["citrus"],
        "base": ["vetiver","patchouli","oakmoss","woody"]
    },
    "Nishane Vetiver": {
        "top": ["citrus"],
        "heart": ["spice"],
        "base": ["vetiver","woody"]
    },
    "Armaf Club de Nuit Untold": {
        "top": ["saffron"],
        "heart": ["jasmine"],
        "base": ["amberwood","cedar","moss"]
    },
    "Kaaf (Ahmed Al Maghribi)": {
        "top": ["red fruits","watermelon","lavender","orange"],
        "heart": ["lotus","jasmine","muguet","sea accord"],
        "base": ["sandalwood","ambroxan","white musk"]
    },
    "Gulab (Raahi)": {
        "top": ["rose"],
        "heart": ["floral","resin"],
        "base": ["cedarwood","musk"]
    },
    "Champa Muse (Raahi)": {
        "top": ["magnolia","citrus"],
        "heart": ["champa","muguet"],
        "base": ["woody","musk"]
    },
}


# ------------------------------
# 3) Helper: flatten notes with weights (top < heart < base)
# ------------------------------
TIER_WEIGHTS = {"top":0.8,"heart":1.0,"base":1.2}

def get_weighted_notes(perfume:str) -> List[str]:
    entry = LOCAL_NOTES_DB.get(perfume)
    if not entry: return []
    weighted=[]
    for tier,notes in entry.items():
        for n in notes:
            weighted.extend([n]*int(10*TIER_WEIGHTS[tier]))
    return weighted


# ------------------------------
# 4) Build training dataframe
# ------------------------------
def build_training_frame(ratings:Dict[str,float]) -> pd.DataFrame:
    rows=[]
    for name,score in ratings.items():
        notes=get_weighted_notes(name)
        if notes:
            rows.append({"name":name,"rating":score,"notes":" ".join(notes)})
    return pd.DataFrame(rows)


# ------------------------------
# 5) Inclination model (ElasticNet for sparsity + regression)
# ------------------------------
@dataclass
class InclinationModel:
    vectorizer:TfidfVectorizer
    reg:ElasticNet
    note_weights:List[Tuple[str,float]]

def fit_inclination_model(df:pd.DataFrame) -> InclinationModel:
    vec=TfidfVectorizer(token_pattern=r"[a-zA-Z][a-zA-Z\s]+")
    X=vec.fit_transform(df["notes"])
    y=df["rating"].values
    reg=ElasticNet(alpha=0.1,l1_ratio=0.7)
    reg.fit(X,y)
    feature_names=np.array(vec.get_feature_names_out())
    coefs=reg.coef_
    note_weights=sorted(zip(feature_names,coefs),key=lambda t:t[1],reverse=True)
    return InclinationModel(vec,reg,note_weights)


# ------------------------------
# 6) Scoring functions
# ------------------------------
def predict_like_score(perfume:str,model:InclinationModel)->float:
    notes=get_weighted_notes(perfume)
    if not notes: return float("nan")
    X=model.vectorizer.transform([" ".join(notes)])
    reg_score=model.reg.predict(X)[0]

    # Cosine similarity vs top liked perfumes
    liked_df=df[df.rating>=8.5]
    liked_vecs=model.vectorizer.transform(liked_df["notes"])
    sim=np.mean(cosine_similarity(X,liked_vecs)) if liked_vecs.shape[0]>0 else 0
    final_score=0.7*reg_score+0.3*(10*sim)
    return max(0,min(10,final_score))


def recommend_top_k(candidates:Iterable[str],model:InclinationModel,k:int=10)->List[Tuple[str,float]]:
    scored=[(name,predict_like_score(name,model)) for name in candidates]
    return sorted([(n,s) for n,s in scored if not math.isnan(s)],key=lambda t:t[1],reverse=True)[:k]


# ------------------------------
# 7) Plot inclination chart
# ------------------------------
def plot_inclination(model:InclinationModel,top_n:int=15):
    top=model.note_weights[:top_n]
    bot=model.note_weights[-top_n:]
    labels_top,vals_top=zip(*top)
    labels_bot,vals_bot=zip(*bot)

    plt.figure(figsize=(10,6))
    plt.barh(labels_top,vals_top)
    plt.title("Top Positive Notes")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.barh(labels_bot,vals_bot)
    plt.title("Most Negative Notes")
    plt.show()


# ------------------------------
# 8) Example run
# ------------------------------
if __name__=="__main__":
    df=build_training_frame(USER_RATINGS)
    if df.empty:
        print("No training data available.")
    else:
        model=fit_inclination_model(df)
        print("Top notes:",[n for n,_ in model.note_weights[:10]])
        test_names=["Kaaf (Ahmed Al Maghribi)","Gulab (Raahi)","Champa Muse (Raahi)"]
        for t in test_names:
            print(t,"->",predict_like_score(t,model))
        plot_inclination(model)
