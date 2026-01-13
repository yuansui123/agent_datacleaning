
# Given metadata, plot_path and return json file describing the saved plot
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import json

def save_example_json(metadata, plot_path, plot_id):
    """Save example metadata and plot info to a JSON file."""

    example_data = {
        "metadata": metadata.to_dict(),
        "plot": {
            "path": plot_path,
            "id": plot_id
        }
    }
    json_path = os.path.splitext(plot_path)[0] + ".json"
    with open(json_path, 'w') as f:
        json.dump(example_data, f, indent=4)
    return json_path