import pandas as pd
from pathlib import Path

# Base directory = project_root/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_data():
    apps_df = pd.read_csv(DATA_DIR / "apps.csv")
    users_df = pd.read_csv(DATA_DIR / "users.csv")
    interactions_df = pd.read_csv(DATA_DIR / "interactions.csv")
    return apps_df, users_df, interactions_df
