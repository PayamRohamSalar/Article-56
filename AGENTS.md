# Repository Guidelines

## Project Structure & Module Organization
The repository centers on season-specific Python scripts (`s3.py`, `s4.py`, `s5.py`, `s5_v2.py`) that load source tables from `data/` and render charts saved into `fig/`. Interactive exploration and prototyping live in the Jupyter notebooks (`season_*.ipynb`, `test.ipynb`). Fonts required for Farsi visualizations are stored in `fonts/`, while prompt references stay under `docs/`. Geospatial overlays belong in `iran-geojson/` and should be versioned alongside matching figures.

## Build, Test, and Development Commands
Create an isolated environment before running scripts:
```bash
python -m venv .venv
.\.venv\Scripts\Activate
pip install pandas numpy matplotlib seaborn arabic-reshaper python-bidi
```
Run a season script locally to regenerate outputs:
```bash
python s5.py
```
Use notebooks for iterative analysis via `jupyter notebook` when reviewing visual changes.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and `snake_case` for variables, functions, and file names (e.g., `season_summary.py`). Keep imports grouped by standard library, third-party, and project modules. Provide docstrings for reusable helpers, and reserve inline comments for non-obvious logic. Persist figure exports using descriptive filenames such as `fig/season5_turnout.png`.

## Testing Guidelines
Automated tests are not yet present; treat each script run as a smoke test. Log key intermediate DataFrame shapes and assert expected column presence before plotting. When adding analytical utilities, include lightweight checks (e.g., `if __name__ == "__main__":`) that call representative scenarios. For notebooks, clear outputs before commit and keep validation notes in markdown cells.

## Commit & Pull Request Guidelines
Adopt Conventional Commit style (`feat:`, `fix:`, `data:`) with succinct 72-character subjects. Reference impacted season or dataset in the body and list reproducible steps (e.g., `python s4.py`). Pull requests should summarize motivation, attach regenerated figures from `fig/` when visuals change, and link coordination tickets. Request review whenever data sources or visualization logic change so reviewers can rerun the exact command sequence.

## Data & Asset Handling
Treat files in `data/` as authoritative inputs; never commit personally identifiable information. Document provenance for new datasets in the PR description. To keep Farsi text legible, ensure `fonts/ttf/Vazirmatn-Regular.ttf` ships with rendered figures and verify plots on both Windows and macOS before publishing.
