# Repository Guidelines

## Project Structure & Module Organization
Season scripts such as `s3.py`, `s4.py`, `s5.py`, and `s6.py` pull their inputs from `data/` and export charts into `fig/`. Use the notebook set (`season_*.ipynb`, `test.ipynb`) for exploratory work and keep outputs cleared before committing. Shared fonts for Farsi labels live under `fonts/ttf/`, prompt references stay in `docs/`, and any map overlays must be stored and versioned in `iran-geojson/` alongside the figures that depend on them.

## Build, Test, and Development Commands
Create a fresh virtual environment before running code:
```bash
python -m venv .venv
.\.venv\Scripts\Activate
pip install pandas numpy matplotlib seaborn arabic-reshaper python-bidi
```
Regenerate season visuals locally with `python s5.py` (swap in other season files as needed), and launch `jupyter notebook` when reviewing or iterating on figures interactively.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and `snake_case` names for modules, functions, and variables. Group imports by standard library, third-party, then project modules. Add docstrings for reusable helpers and only include inline comments to clarify non-obvious steps such as custom reshaping or bidi handling. Save figures with descriptive names like `fig/season5_turnout.png` to match their dataset and season.

## Testing Guidelines
We treat each script execution as a smoke test. Log key DataFrame shapes, verify expected columns before plotting, and raise when inputs are missing. For new utilities, include an `if __name__ == "__main__":` block that exercises a representative path. Notebooks should record validation notes in markdown cells and keep execution counts reset before review.

## Commit & Pull Request Guidelines
Use Conventional Commit prefixes (`feat:`, `fix:`, `data:`) and keep subject lines under 72 characters. Reference the affected season or dataset in the body and list reproduction steps (e.g., `python s4.py`). Pull requests must summarize the change, attach regenerated figures from `fig/` when visuals update, and link coordination tickets so reviewers can rerun the exact command sequence.

## Data & Asset Handling
Treat everything under `data/` as authoritative and never commit personally identifiable information. When introducing new datasets, document provenance in the pull request. Include `fonts/ttf/Vazirmatn-Regular.ttf` with any shared outputs to preserve Farsi text, and confirm generated charts render correctly on both Windows and macOS before publishing.