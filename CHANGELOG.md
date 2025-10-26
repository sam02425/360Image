# Changelog

All notable changes to this repository will be documented in this file.

## [Unreleased]
- Planned: Consolidate top-level academic experiment scripts into `docs/legacy/`.
- Planned: Add CI workflow to run smoke tests on push.

## [2025-10-26] Repository Reorganization and Hygiene
- Added `README.md` with updated structure, capabilities, and usage examples.
- Created `MIGRATION_GUIDE.md` documenting file moves, canonical entrypoints, and path updates.
- Added `requirements.txt` consolidating core dependencies and `pytest` for smoke tests.
- Introduced `tests/test_cli_scripts.py` to invoke `--help` across main CLI scripts.
- Refactored `experiments/deim_pipeline.py` to lazy-load heavy dependencies so `--help` exits quickly.
- Confirmed all CLI help commands pass (`pytest -q` → 1 test passed).
- Moved legacy research scripts and documents into `docs/legacy/` and images into `docs/images/`.
- Removed duplicate top-level `run_all_experiments.py` in favor of `experiments/run_all_experiments.py`.