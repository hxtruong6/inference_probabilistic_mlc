"""Backward-compatible shim.

The evaluation driver now lives in the package as ``dacaf_mlc.evaluate``.
Prefer the console command ``dacaf-mlc`` or ``python -m dacaf_mlc.evaluate``.
This shim keeps ``python inference_evaluate_models.py ...`` working (e.g. for
existing Slurm scripts).
"""
from dacaf_mlc.evaluate import main

if __name__ == "__main__":
    main()
