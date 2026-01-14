import os
from typing import Optional

import pandas as pd
import numpy as np
import scipy.io

DEFAULT_DATASET_DIR = "C:\\Users\\yuans\\Desktop\\Mayo_dataset\\DATASET_MAYO"
DEFAULT_SEGMENTS_CSV = os.path.join(DEFAULT_DATASET_DIR, "segments.csv")

class MayoDataset:
    """Convenient accessor for Mayo dataset files and metadata.

    Provides default paths, cached metadata, and helper methods to retrieve
    segment raw arrays and metadata rows.
    """

    def __init__(self, base_dir: Optional[str] = None, csv_path: Optional[str] = None):
        self.base_dir = base_dir or DEFAULT_DATASET_DIR
        self.csv_path = csv_path or os.path.join(self.base_dir, "segments.csv")
        self._segments: Optional[pd.DataFrame] = None

    @property
    def segments(self) -> pd.DataFrame:
        """Pandas DataFrame of all segments metadata (lazy-loaded and cached)."""
        if self._segments is None:
            self._segments = pd.read_csv(self.csv_path)
        return self._segments
    
    def print_columns(self):
        """Print the column names of the segments DataFrame."""
        print(self.segments.columns)
    
    def print_statistics(self):
        """Print basic statistics of the segments DataFrame."""
        df = self.segments
        print("Segments DataFrame Statistics:")
        print(f"Total segments: {len(df)}")
        print("\nColumn-wise statistics:")
        for column in df.columns:
            # print number of unique values
            unique_values = df[column].nunique()
            print(f"- {column}: {unique_values} unique values")
            # if unique values < 10, print the unique values
            if unique_values < 10:
                print(f"  Unique values: {df[column].unique()}")

    def reload_segments(self) -> pd.DataFrame:
        """Force re-read of the segments CSV and update cache."""
        self._segments = pd.read_csv(self.csv_path)
        return self._segments

    def get_raw(self, segment_id: str) -> np.ndarray:
        """Return raw data array for a given `segment_id` (1D np.array)."""
        mat_path = os.path.join(self.base_dir, f"{segment_id}.mat")
        mat = scipy.io.loadmat(mat_path)
        data = np.array(mat.get("data"))
        return data.flatten()

    def get_metadata(self, segment_id: str) -> pd.Series:
        """Return metadata row for a given `segment_id` as a pandas Series.

        Raises KeyError if not found.
        """
        df = self.segments
        match = df[df["segment_id"].astype(str) == str(segment_id)]
        if match.empty:
            raise KeyError(f"segment_id not found: {segment_id}")
        return match.iloc[0]
    
    def get_random_k_segments(self, k: int) -> pd.DataFrame:
        """Return a DataFrame of k random segments from the dataset."""
        return self.segments.sample(n=k).reset_index(drop=True)
    
    def get_random_k_segments_by_category(self, k: int, category_col: str, category_value) -> pd.DataFrame:
        """Return a DataFrame of k random segments filtered by a specific category."""
        filtered_df = self.segments[self.segments[category_col] == category_value]
        return filtered_df.sample(n=k).reset_index(drop=True)


__all__ = [
    "MayoDataset",
]