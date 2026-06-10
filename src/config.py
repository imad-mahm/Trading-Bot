"""Tiny helper to load config.yaml into a plain dictionary.

Keeping this in one place means every other module reads settings the same way.
"""

from pathlib import Path
import yaml


# The project root is the folder that contains config.yaml (one level above src/).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read the YAML config file and return it as a dictionary.

    Parameters
    ----------
    path : str or Path
        Location of the config file. Defaults to ``config.yaml`` in the project
        root, so you normally don't pass anything.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find config file at {path}. "
            "Make sure you run commands from the project folder."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
