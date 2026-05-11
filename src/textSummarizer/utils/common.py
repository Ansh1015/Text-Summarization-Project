import os
from pathlib import Path
from typing import Any

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations

from textSummarizer.logging import logger


@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns a ConfigBox for dot-notation access."""
    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
        if content is None:
            raise ValueError(f"YAML file is empty: {path_to_yaml}")
        logger.info(f"YAML file loaded: {path_to_yaml}")
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"YAML file is empty: {path_to_yaml}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {path_to_yaml}")


@ensure_annotations
def create_directories(path_to_directories: list, verbose: bool = True):
    """Creates a list of directories if they don't exist."""
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the file size in KB."""
    size_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_kb} KB"
