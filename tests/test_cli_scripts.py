import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / "experiments"

SCRIPTS = [
    "train_yolov8.py",
    "train_yolov11.py",
    "train_rtdetr_l.py",
    "eval_grounding_dino.py",
    "eval_florence2_vlm.py",
    "deim_pipeline.py",
    "run_all_experiments.py",
]


def run_help(script: str) -> int:
    script_path = EXPERIMENTS / script
    assert script_path.exists(), f"Missing script: {script_path}"
    cmd = [sys.executable, str(script_path), "--help"]
    env = os.environ.copy()
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
    except subprocess.TimeoutExpired:
        return 1
    return proc.returncode


def test_all_cli_scripts_have_help():
    failures = []
    for s in SCRIPTS:
        rc = run_help(s)
        if rc != 0:
            failures.append((s, rc))
    assert not failures, f"Help check failed for: {failures}"