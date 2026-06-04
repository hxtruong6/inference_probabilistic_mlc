"""Project-level constants and paths."""
import os

# Repo root = parent of the package dir (datasets/ and result/ live there).
# Overridable via DACAF_BASE_DIR so the CLI can be run from anywhere.
BASE_DIR = os.environ.get(
    "DACAF_BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

# Default experimental protocol.
SEED = 6
KFOLD_SPLIT_NUMBER = 10
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
