"""Locate bcr-mistral data, model weights and outputs.

Three roots, each overridable by an environment variable:
  BCRM_DATA    input data (sequence tables, metadata, TRUST4 output)   default: <repo>/data
  BCRM_MODELS  foundation-model weights and HF configs                  default: <repo>/models
  BCRM_OUT     everything the pipeline regenerates (embeddings,
               classifier output, figures)                              default: <repo>/results
See docs/DATA_LAYOUT.md for what lives under each root.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _root(var, default):
    return Path(os.environ.get(var, REPO_ROOT / default)).expanduser()


def repo_path(rel):
    return str(REPO_ROOT / rel)


def data_path(rel=""):
    return str(_root("BCRM_DATA", "data") / rel)


def model_path(rel=""):
    return str(_root("BCRM_MODELS", "models") / rel)


def out_path(rel=""):
    """Output location; its parent directory is created if missing."""
    p = _root("BCRM_OUT", "results") / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)
