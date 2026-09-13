from pathlib import Path
import pandas as pd


def load_bandung_google(path: str | Path) -> pd.DataFrame:
    """Load the immutable Google edge-sample table."""
    return pd.read_csv(path)


def load_sampled_edges(path: str | Path) -> pd.DataFrame:
    """Load sampled OSM edge metadata."""
    return pd.read_csv(path)
