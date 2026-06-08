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

# Cap on samples held in the 2^L joint at once during inference. None = no
# chunking (one batch; the exact paradigm used for the paper's small/medium
# datasets). Set a positive int (e.g. via DACAF_INFERENCE_BATCH_SIZE) to bound
# peak memory on large-N datasets such as the chest-X-ray features; chunking is
# numerically equivalent up to float associativity and does not change predictions.
_bs = os.environ.get("DACAF_INFERENCE_BATCH_SIZE")
INFERENCE_BATCH_SIZE = int(_bs) if _bs else None
