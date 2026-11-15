#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_all.py — Execute the complete DSAI 4207 Data Engineering pipeline (00 -> 07) in order.
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    "00_download.py",
    "01_canonicalize.py",
    "03_testset.py",
    "02_decontaminate_and_dedup.py",
    "04_split.py",
    "05_warmup.py",
    "06_format_arms.py",
    "07_validate.py",
]


def main() -> None:
    t0 = time.time()
    for script_name in PIPELINE_STEPS:
        script_path = SCRIPTS_DIR / script_name
        print(f"\n{'='*72}\n>>> Running {script_name} ...\n{'='*72}", flush=True)
        step_t0 = time.time()
        res = subprocess.run([sys.executable, str(script_path)], check=False)
        if res.returncode != 0:
            print(f"[FATAL] {script_name} failed with exit code {res.returncode}!", file=sys.stderr)
            sys.exit(res.returncode)
        print(f">>> {script_name} completed in {time.time() - step_t0:.2f}s.", flush=True)

    print(f"\n{'='*72}\n>>> ALL DATA ENGINEERING STEPS COMPLETED IN {time.time() - t0:.2f}s!\n{'='*72}")


if __name__ == "__main__":
    main()
