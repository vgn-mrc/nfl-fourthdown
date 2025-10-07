# Repository Structure

```text
nfl-fourthdown/
├── LICENSE
├── README.md
├── configs/
├── data/
├── main.py
├── notebooks/
├── pyproject.toml
├── src/
│   └── data/
│       ├── make_dataset.py
│       └── play_by_play_download.py
└── uv.lock
```

- `README.md` — project overview and setup instructions.
- `configs/` — placeholder for configuration files (currently empty).
- `data/` — storage for raw and processed datasets.
- `main.py` — script entry point for running project logic.
- `notebooks/` — exploratory analysis notebooks.
- `pyproject.toml` & `uv.lock` — project packaging and dependency lock.
- `src/` — project source code. Current `data/` module contains dataset preparation scripts.
