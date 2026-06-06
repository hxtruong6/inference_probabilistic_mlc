# Reproducibility-capsule image for DaCaF (Code Ocean / Software Impacts).
# Reproduces the paper's tabular CHD-49 result table on CPU in seconds.
FROM python:3.10-slim

# Deterministic + quiet. DACAF_BASE_DIR pins the repo root so the CLI finds
# datasets/ regardless of where the package is installed (the default derives
# the root from the package location, which is wrong for a site-packages install).
ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DACAF_BASE_DIR=/app

WORKDIR /app

# Build tools for scientific wheels (most install as wheels, but be safe).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran \
    && rm -rf /var/lib/apt/lists/*

# Pin the core (tabular) scientific stack first, so the package install below
# does not re-resolve to newer versions. The heavy [image] stack is excluded.
COPY requirements-core.txt /app/requirements-core.txt
RUN pip install -r /app/requirements-core.txt

# Install the package itself without touching the pinned dependencies.
COPY . /app
RUN pip install --no-deps -e .

# Default reproducible run: CHD-49 table -> /results (Code Ocean mounts /results).
# Override with e.g. `bash scripts/reproduce_tabular.sh /results` for all 5 datasets.
CMD ["bash", "scripts/reproduce_capsule.sh", "/results"]
