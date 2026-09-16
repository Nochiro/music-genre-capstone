"""Clean and consolidate the FMA genre taxonomy.

The raw FMA metadata contains 161 genre tags. This module maps them to a
smaller set of top-level genre categories and drops tags that have too few
tracks to be useful for training.
"""

from typing import Dict, Optional

import pandas as pd


def load_genre_metadata(metadata_path: str) -> pd.DataFrame:
    """Load FMA track/genre metadata from disk.

    Args:
        metadata_path: Path to the FMA metadata CSV (e.g. ``tracks.csv``).

    Returns:
        DataFrame with track IDs and raw genre labels.
    """
    ...


def clean_genre_taxonomy(
    df: pd.DataFrame,
    min_tracks: int = 100,
) -> pd.DataFrame:
    """Map raw genre tags to top-level categories and drop rare genres.

    Args:
        df: DataFrame produced by :func:`load_genre_metadata`.
        min_tracks: Minimum number of tracks a top-level genre must have
                    to be retained.

    Returns:
        Filtered DataFrame with a ``genre_top`` column.
    """
    ...


def get_genre_mapping() -> Dict[str, str]:
    """Return a dictionary mapping raw genre strings to top-level categories.

    Returns:
        Mapping of ``{raw_genre: top_level_genre}``.
    """
    ...
