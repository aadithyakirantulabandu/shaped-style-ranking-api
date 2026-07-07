"""Small public interaction dataset (MovieLens ml-latest-small, ~100k ratings),
reshaped to the generic event schema in api/schemas.py.
"""
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ML_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def _ensure_downloaded(cache_dir: Path) -> Path:
    extracted_dir = cache_dir / "ml-latest-small"
    if not (extracted_dir / "ratings.csv").exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = cache_dir / "ml-latest-small.zip"
        urllib.request.urlretrieve(ML_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(cache_dir)
    return extracted_dir


def load_movielens_events(cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Returns columns: user_id, item_id, timestamp, event_type."""
    extracted_dir = _ensure_downloaded(cache_dir)
    ratings = pd.read_csv(extracted_dir / "ratings.csv")
    return pd.DataFrame({
        "user_id": ratings["userId"].astype(str),
        "item_id": ratings["movieId"].astype(str),
        "timestamp": ratings["timestamp"],
        "event_type": ratings["rating"].apply(lambda r: "like" if r >= 4.0 else "view"),
    })


def load_movielens_item_text(cache_dir: Path = CACHE_DIR) -> dict:
    """item_id -> "title genre1 genre2 ..." for content-based embedding."""
    extracted_dir = _ensure_downloaded(cache_dir)
    movies = pd.read_csv(extracted_dir / "movies.csv")
    text = movies["title"] + " " + movies["genres"].str.replace("|", " ", regex=False)
    return dict(zip(movies["movieId"].astype(str), text))
