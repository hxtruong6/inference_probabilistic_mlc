FROM python:3.10-slim

WORKDIR /app

# Build tools for scientific wheels (most install as wheels, but be safe).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran git \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Core (tabular) install. For the ChestX-ray image experiments use:
#   pip install ".[image]"
RUN pip install --no-cache-dir .

# Default: run the full local evaluation sweep (see `dacaf-mlc --help`).
CMD ["dacaf-mlc"]
