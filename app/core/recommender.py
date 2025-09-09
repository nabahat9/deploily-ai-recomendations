import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .data_loader import load_data

# Load datasets
apps_df, users_df, interactions_df = load_data()

# =========================
# 1. Content-Based Filtering
# =========================
apps_df["features"] = apps_df["category"] + " " + apps_df["price"].astype(str)
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(apps_df["features"])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_content(app_id, top_n=5):
    idx = apps_df.index[apps_df["app_id"] == app_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    return apps_df.iloc[[i[0] for i in sim_scores]][["app_id", "app_name", "category", "price"]]


# =========================
# 2. Collaborative Filtering
# =========================
user_item_matrix = interactions_df.pivot_table(
    index="user_id", columns="app_id", values="rating"
).fillna(0)

user_sim = cosine_similarity(user_item_matrix)
user_sim_df = pd.DataFrame(
    user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)


def recommend_collaborative(user_id, top_n=5):
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[
        1:6].index
    similar_users_ratings = user_item_matrix.loc[similar_users].mean(
    ).sort_values(ascending=False)
    user_rated = set(
        interactions_df[interactions_df["user_id"] == user_id]["app_id"])
    recs = similar_users_ratings.drop(user_rated).head(top_n).index
    return apps_df[apps_df["app_id"].isin(recs)][["app_id", "app_name", "category", "price"]]

# =========================
# 3. Hybrid Recommendation
# =========================


def recommend_hybrid(user_id=None, app_id=None, top_n=5, alpha=0.6):
    if app_id is None:
        raise ValueError("Need at least app_id for recommendations")

    if (user_id is None) or (user_id not in user_item_matrix.index) or (user_item_matrix.loc[user_id].sum() == 0):
        return recommend_content(app_id=app_id, top_n=top_n)

    similar_users = user_sim_df[user_id].sort_values(ascending=False)[
        1:6].index
    similar_users_ratings = user_item_matrix.loc[similar_users].mean()
    user_rated = set(
        interactions_df[interactions_df["user_id"] == user_id]["app_id"])
    similar_users_ratings = similar_users_ratings.drop(
        user_rated, errors="ignore")

    cf_scores = (similar_users_ratings - similar_users_ratings.min()) / (
        similar_users_ratings.max() - similar_users_ratings.min() + 1e-9
    )

    idx = apps_df.index[apps_df["app_id"] == app_id][0]
    sim_scores = cosine_sim[idx]
    content_scores = pd.Series(sim_scores, index=apps_df["app_id"])
    content_scores = content_scores.drop(user_rated, errors="ignore")
    content_scores = (content_scores - content_scores.min()) / (
        content_scores.max() - content_scores.min() + 1e-9
    )

    hybrid_scores = alpha * \
        cf_scores.add(0, fill_value=0) + (1 - alpha) * \
        content_scores.add(0, fill_value=0)
    recs = hybrid_scores.sort_values(ascending=False).head(top_n).index
    return apps_df[apps_df["app_id"].isin(recs)][["app_id", "app_name", "category", "price"]]
