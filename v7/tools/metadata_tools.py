# tools/metadata_tools.py
from typing import Dict, Any, List, Sequence

import numpy as np

from ._utils import DEFAULT_PLOT_DIR  # kept for consistency, though unused


def metadata_snapshot_tool(
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return metadata so that interpreters can reason about it numerically.

    Returns
    -------
    {
      "numeric_results": {"metadata_snapshot": metadata_dict},
      "image_paths": []
    }
    """
    return {
        "numeric_results": {"metadata_snapshot": metadata},
        "image_paths": [],
    }


def metadata_completeness_tool(
    metadata: Dict[str, Any],
    required_fields: Sequence[str],
) -> Dict[str, Any]:
    """
    Simple metadata quality checker.

    Parameters
    ----------
    metadata : dict
        Arbitrary metadata dictionary.
    required_fields : sequence of str
        Keys that are expected to be present in metadata.

    Returns
    -------
    {
      "numeric_results": {
         "metadata_completeness": {
             "n_required": int,
             "n_present": int,
             "n_missing": int,
             "missing_fields": [str, ...]
         }
      },
      "image_paths": []
    }
    """
    missing = [k for k in required_fields if k not in metadata]
    n_required = len(required_fields)
    n_present = n_required - len(missing)

    summary = {
        "n_required": int(n_required),
        "n_present": int(n_present),
        "n_missing": int(len(missing)),
        "missing_fields": missing,
    }
    return {
        "numeric_results": {"metadata_completeness": summary},
        "image_paths": [],
    }
