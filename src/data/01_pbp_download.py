"""
Fetch play-by-play for full NFL seasons using nfl_data_py.
- Output: Saves one Parquet file per season (data/raw/pbp_<YEAR>.parquet)
"""

import argparse, os, re
import pandas as pd
import nfl_data_py as nfl

def parse_seasons(spec: str):
    # Examples: "2016-2024", "2019,2021,2023", "2020 2022"
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in re.split(r"[,\s]+", spec) if x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", required=True, help='e.g., "2016-2024" or "2019,2021,2023"')
    ap.add_argument("--outdir", default="data/raw")
    ap.add_argument("--engine", default="pyarrow", choices=["pyarrow", "fastparquet"],
                    help="Parquet engine (default: pyarrow)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    years = parse_seasons(args.seasons)
    print(f"Importing play-by-play for seasons: {years}")

    total_rows = 0
    for yr in years:
        print(f"  • Season {yr}: downloading…")
        # downcast=True keeps memory in check and speeds IO
        df = nfl.import_pbp_data([yr], downcast=True)

        out_parq = os.path.join(args.outdir, f"pbp_{yr}.parquet")

        # Try preferred engine; on ImportError, attempt the alternative once.
        try:
            df.to_parquet(out_parq, index=False, engine=args.engine)
        except ImportError as e:
            alt = "fastparquet" if args.engine == "pyarrow" else "pyarrow"
            print(f"    Engine '{args.engine}' not available ({e}). Trying '{alt}'…")
            try:
                df.to_parquet(out_parq, index=False, engine=alt)
            except ImportError as e2:
                raise SystemExit(
                    f"Neither '{args.engine}' nor '{alt}' is installed. "
                    f"Install one, e.g.: `uv add {args.engine}` or `uv add {alt}`"
                ) from e2

        print(f"    Saved {out_parq} with {len(df):,} rows.")
        total_rows += len(df)

    print(f"Done. Wrote {len(years)} files totaling {total_rows:,} rows.")

if __name__ == "__main__":
    main()
