import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data_loader import load_data

CACHE_DIR = "app/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def precompute_and_cache():
    apps_df, users_df, interactions_df = load_data()

    # Content-based
    apps_df["features"] = apps_df["category"] + " " + apps_df["price"].astype(str)
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(apps_df["features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Collaborative
    user_item_matrix = interactions_df.pivot_table(
        index="user_id", columns="app_id", values="rating"
    ).fillna(0)
    user_sim = cosine_similarity(user_item_matrix)
    user_sim_df = pd.DataFrame(
        user_sim, index=user_item_matrix.index, columns=user_item_matrix.index
    )

    # Save
    with open(os.path.join(CACHE_DIR, "cosine_sim.pkl"), "wb") as f:
        pickle.dump(cosine_sim, f)
    with open(os.path.join(CACHE_DIR, "user_sim_df.pkl"), "wb") as f:
        pickle.dump(user_sim_df, f)

    print("✅ Precomputation updated")
