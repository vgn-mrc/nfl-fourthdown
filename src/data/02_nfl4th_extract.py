from __future__ import annotations

import argparse
import os
import sys
import tempfile
import subprocess
from pathlib import Path
from typing import List

import pandas as pd

REQ_COLS = [
    # game identifiers (kept for joins / sanity)
    "play_id", "game_id",
    # team + season
    "home_team", "away_team", "posteam", "season", "season_type",
    # core situation
    "qtr", "quarter_seconds_remaining", "ydstogo", "yardline_100",
    "score_differential",
    # misc required by nfl4th examples
    "home_opening_kickoff",
    "posteam_timeouts_remaining", "defteam_timeouts_remaining",
]

MIN_SEASON = 2014

def _validate_prereqs():
    # Optional: rpy2 path first for in-process speed
    try:
        import rpy2  # noqa: F401
        return "rpy2"
    except Exception:
        pass
    # Fallback: Rscript must be callable
    if subprocess.call(["which", "Rscript"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0:
        return "rscript"
    raise SystemExit(
        "No R bridge found. Install either:\n"
        "  • rpy2 (Python)  OR\n"
        "  • R + Rscript and the R package nfl4th\n\n"
        "R quickstart:\n"
        "  install.packages('nfl4th')  # installs deps (nflfastR, nflreadr, mgcv, xgboost, ...)"
    )

def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)

def _prep_input(df: pd.DataFrame) -> pd.DataFrame:
    # Only 4th downs that are actual decision points; keep wide to avoid over-filtering
    mask = (df["down"] == 4) & (df["play_deleted"] != 1)
    cand = df.loc[mask, :].copy()

    # Ensure required columns exist
    missing = sorted(set(REQ_COLS) - set(cand.columns))
    if missing:
        raise SystemExit(f"Missing required columns for nfl4th: {missing}")

    # Map season_type (REG/POST) -> type (reg/post) as expected by nfl4th docs
    cand["type"] = cand["season_type"].str.lower().map({"reg": "reg", "post": "post"})
    # Keep minimal frame for nfl4th, plus join keys
    use = cand[REQ_COLS + ["type"]].copy()

    # nfl4th examples use integer qtr and numeric others; coerce conservatively
    use["qtr"] = use["qtr"].astype("Int64")
    use["quarter_seconds_remaining"] = pd.to_numeric(use["quarter_seconds_remaining"], errors="coerce")
    use["ydstogo"] = pd.to_numeric(use["ydstogo"], errors="coerce")
    use["yardline_100"] = pd.to_numeric(use["yardline_100"], errors="coerce")
    use["score_differential"] = pd.to_numeric(use["score_differential"], errors="coerce")
    use["posteam_timeouts_remaining"] = pd.to_numeric(use["posteam_timeouts_remaining"], errors="coerce")
    use["defteam_timeouts_remaining"] = pd.to_numeric(use["defteam_timeouts_remaining"], errors="coerce")
    use["season"] = pd.to_numeric(use["season"], errors="coerce")

    # Drop rows with any critical NaNs for the computation
    crit = [
        "home_team","away_team","posteam","season","type","qtr",
        "quarter_seconds_remaining","ydstogo","yardline_100","score_differential",
        "posteam_timeouts_remaining","defteam_timeouts_remaining","home_opening_kickoff",
    ]
    before = len(use)
    use = use.dropna(subset=crit)
    dropped = before - len(use)
    if dropped:
        print(f"  - Dropped {dropped} plays lacking required fields for nfl4th")

    return use

def _run_nfl4th_rpy2(df_in: pd.DataFrame) -> pd.DataFrame:
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri

    # Load nfl4th once
    ro.r("suppressMessages(library(nfl4th))")
    add_4th_probs = ro.r("nfl4th::add_4th_probs")

    # Use local converter so inputs/outputs are handled automatically
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(df_in)
        r_res = add_4th_probs(r_df) # type: ignore
        # r_res is already a pandas.DataFrame under this converter;
        # if not, convert once.
        if isinstance(r_res, pd.DataFrame):
            out = r_res
        else:
            out = ro.conversion.rpy2py(r_res)

    return out.reset_index(drop=True)



def _run_nfl4th_rscript(df_in: pd.DataFrame) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as td:
        in_csv = Path(td) / "in.csv"
        out_csv = Path(td) / "out.csv"
        rfile = Path(td) / "run.R"
        df_in.to_csv(in_csv, index=False)
        r_code = """
suppressMessages({
  library(nfl4th)
  library(readr)
})
args <- commandArgs(trailingOnly=TRUE)
infile <- args[1]; outfile <- args[2]
df <- readr::read_csv(infile, show_col_types = FALSE)
res <- nfl4th::add_4th_probs(df)
readr::write_csv(res, outfile)
"""
        rfile.write_text(r_code)
        subprocess.run(["Rscript", str(rfile), str(in_csv), str(out_csv)], check=True)
        return pd.read_csv(out_csv)

def _enrich_and_write(full_df: pd.DataFrame, probs_df: pd.DataFrame, out_path: Path):
    # Merge back by (game_id, play_id) for safety
    keep_cols = [c for c in probs_df.columns if c not in full_df.columns]
    merged = full_df.merge(
        probs_df[["game_id","play_id"] + keep_cols],
        on=["game_id","play_id"],
        how="left",
        validate="one_to_one",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    return merged

def process_season(year: int, in_dir: Path, out_dir: Path, bridge: str):
    if year < MIN_SEASON:
        print(f"Skipping {year}: nfl4th supports seasons >= {MIN_SEASON}.")
        return

    in_path = in_dir / f"pbp_{year}.parquet"
    if not in_path.exists():
        print(f"  ! Missing {in_path}, skipping.")
        return

    print(f"Season {year}: reading {in_path} …")
    df = _read_parquet(in_path)
    use = _prep_input(df)

    print(f"  -> Running nfl4th::add_4th_probs on {len(use):,} 4th-down plays via {bridge} …")
    if bridge == "rpy2":
        probs = _run_nfl4th_rpy2(use)
    else:
        probs = _run_nfl4th_rscript(use)

    out_path = out_dir / f"fourth_probs_{year}.parquet"
    _enrich_and_write(df, probs, out_path)
    print(f"  -> Wrote {out_path}")

def parse_seasons(spec: str) -> List[int]:
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.replace(",", " ").split() if s]

def main():
    ap = argparse.ArgumentParser(description="Add nfl4th 4th-down probabilities to PBP Parquets.")
    ap.add_argument("--seasons", required=True, help='e.g. "2016-2024" or "2019,2021,2023"')
    ap.add_argument("--in-dir", default="data/raw", help="Directory with pbp_<YEAR>.parquet")
    ap.add_argument("--out-dir", default="data/processed", help="Directory for fourth_probs_<YEAR>.parquet")
    args = ap.parse_args()

    bridge = _validate_prereqs()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    years = parse_seasons(args.seasons)
    for y in years:
        process_season(y, in_dir, out_dir, bridge)

if __name__ == "__main__":
    main()
