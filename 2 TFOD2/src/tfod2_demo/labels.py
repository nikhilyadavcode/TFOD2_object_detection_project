from __future__ import annotations

import re
from pathlib import Path


def load_label_map(path: Path) -> dict[int, str]:
    """Parse a TF Object Detection API .pbtxt label map."""
    text = path.read_text(encoding="utf-8")
    labels: dict[int, str] = {}

    for item in re.findall(r"item\s*{(.*?)}", text, flags=re.DOTALL):
        id_match = re.search(r"\bid\s*:\s*(\d+)", item)
        name_match = re.search(r"\b(?:display_name|name)\s*:\s*['\"]([^'\"]+)['\"]", item)
        if id_match and name_match:
            labels[int(id_match.group(1))] = name_match.group(1)

    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels

