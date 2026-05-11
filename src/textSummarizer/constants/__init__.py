from pathlib import Path

# Anchored to the package root, independent of working directory
_PKG_ROOT = Path(__file__).parent.parent.parent.parent

CONFIG_FILE_PATH = _PKG_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = _PKG_ROOT / "params.yaml"
