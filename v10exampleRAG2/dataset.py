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
    
    #print all the columns of segments dataframe
    def print_columns(self):
        print(self.segments.columns)

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
        if "segment_id" not in df.columns:
            raise KeyError("segments metadata missing 'segment_id' column")
        match = df[df["segment_id"].astype(str) == str(segment_id)]
        if match.empty:
            raise KeyError(f"segment_id not found: {segment_id}")
        return match.iloc[0]


__all__ = [
    "MayoDataset",
]