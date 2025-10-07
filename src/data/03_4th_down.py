from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np


# -----------------------------
# Column groups to KEEP (raw)
# -----------------------------
ID_COLS = [
    "play_id","game_id","nfl_api_id","nflverse_game_id",
    "season","season_type","week","game_date","start_time","time_of_day",
]

TEAM_COLS = [
    "home_team","away_team","posteam","defteam","possession_team",
    "posteam_type","side_of_field",
]

STATE_COLS = [
    "qtr","quarter_seconds_remaining","half_seconds_remaining","game_seconds_remaining",
    "game_half","down","ydstogo","yardline_100","goal_to_go","goaltogo",
    "score_differential","play_clock",
    "posteam_timeouts_remaining","defteam_timeouts_remaining",
    "home_timeouts_remaining","away_timeouts_remaining",
]

ENV_COLS = ["roof","surface","temp","wind","weather","stadium_id","game_stadium","stadium","location"]

NFL4TH_COLS = [
    # decision WPs + components
    "go_boost","first_down_prob","wp_fail","wp_succeed",
    "go_wp","fg_make_prob","miss_fg_wp","make_fg_wp","fg_wp","punt_wp",
]

EDA_HELPERS = [
    "ep","wp","xpass","pass_oe",
    "kick_distance","play_type","play_type_nfl","special_teams_play",
    "desc",
]

DECISION_FLAGS = [
    "field_goal_attempt","punt_attempt","rush_attempt","pass_attempt","sack",
    "aborted_play","penalty","play_deleted","yards_gained",
]

DRIVE_SERIES = [
    "series","series_success","series_result","order_sequence",
    "drive","fixed_drive","fixed_drive_result",
    "drive_play_count","drive_first_downs","drive_inside20",
    "drive_ended_with_score","drive_quarter_start","drive_quarter_end",
    "drive_start_yard_line","drive_end_yard_line",
]

SCORE_COLS = [
    "total_home_score","total_away_score",
    "posteam_score","defteam_score",
    "posteam_score_post","defteam_score_post","score_differential_post",
    "home_score","away_score",
]

# all-NA we explicitly drop if present (seen in your run)
KNOWN_ALL_NA = {"st_play_type", "end_yard_line", "old_game_id"}


def parse_seasons(spec: str | None) -> list[int] | None:
    if not spec:
        return None
    spec = spec.strip()
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.replace(",", " ").split() if s]


def find_parquets(in_dir: str, seasons: list[int] | None) -> list[Path]:
    base = Path(in_dir)
    pats = ["fourth_probs_*.parquet", "ourth_probs_*.parquet"]  # accept typo
    candidates: list[Path] = []
    for p in pats:
        candidates.extend(sorted(base.glob(p)))
    if seasons:
        year_re = re.compile(r"(\d{4})")
        filtered = []
        for c in candidates:
            m = year_re.search(c.name)
            if m and int(m.group(1)) in seasons:
                filtered.append(c)
        return filtered
    return candidates


def _select_keep(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        ID_COLS + TEAM_COLS + STATE_COLS + ENV_COLS +
        NFL4TH_COLS + EDA_HELPERS + DECISION_FLAGS + DRIVE_SERIES + SCORE_COLS +
        ["home_opening_kickoff","end_clock_time"]  # useful context if present
    )
    cols = [c for c in keep if c in df.columns]
    return df[cols]


def load_year(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # keep only true decision points (at least one option available)
    mask = df[["go_wp","fg_wp","punt_wp"]].notna().any(axis=1)
    df = df.loc[mask, :]

    df = _select_keep(df)

    # drop known all-NA early if present in this year
    drop_now = [c for c in KNOWN_ALL_NA if c in df.columns and df[c].isna().all()]
    if drop_now:
        df = df.drop(columns=drop_now)

    # normalize a few integer-ish fields to pandas nullable ints (no cleaning)
    for c in ["posteam_timeouts_remaining","defteam_timeouts_remaining","home_timeouts_remaining","away_timeouts_remaining"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # availability flags
    out["available_go"]   = out["go_wp"].notna().astype("Int8")
    out["available_fg"]   = out["fg_wp"].notna().astype("Int8")
    out["available_punt"] = out["punt_wp"].notna().astype("Int8")

    # actual observed choice (no cleaning; NA if ambiguous)
    def _actual_choice(row):
        if row.get("field_goal_attempt", 0) == 1: return "FG"
        if row.get("punt_attempt", 0) == 1: return "PUNT"
        if (row.get("rush_attempt", 0) == 1) or (row.get("pass_attempt", 0) == 1) or (row.get("sack", 0) == 1):
            return "GO"
        return pd.NA
    out["actual_choice"] = out.apply(_actual_choice, axis=1)

    # model recommendation (argmax over available)
    def _model_choice(row):
        opts = {k: row[k] for k in ["go_wp","fg_wp","punt_wp"] if pd.notna(row[k])}
        if not opts:
            return pd.NA
        best = max(opts, key=opts.get) #type: ignore
        return best.replace("_wp","").upper()
    out["model_choice"] = out.apply(_model_choice, axis=1)

    # margins / aggressiveness
    def _top2_margin(row):
        vals = [row[k] for k in ["go_wp","fg_wp","punt_wp"] if pd.notna(row[k])]
        if len(vals) < 2: return pd.NA
        s = sorted(vals, reverse=True)
        return float(s[0] - s[1])
    out["margin_top2"] = out.apply(_top2_margin, axis=1)

    def _aggr(row):
        others = [row[k] for k in ["fg_wp","punt_wp"] if pd.notna(row[k])]
        if pd.isna(row["go_wp"]) or not others:
            return pd.NA
        return float(row["go_wp"] - max(others))
    out["go_aggressiveness"] = out.apply(_aggr, axis=1)

    # estimated FG distance from LOS (yards): 17 + (100 - yardline_100)
    y = out["yardline_100"].clip(lower=1, upper=99)
    out["est_fg_distance"] = 17 + (100 - y)

    # bins (EDA-friendly)
    out["ytg_bin"] = pd.cut(out["ydstogo"],
                            bins=[0,1,3,6,10,99],
                            labels=["1","2-3","4-6","7-10","11+"],
                            include_lowest=True)

    # Corrected field position labels for yardline_100
    out["yardline_zone"] = pd.cut(out["yardline_100"],
                                  bins=[0,20,40,60,80,100],
                                  labels=["Opp 1-20 (RZ)","Opp 21-40","Midfield 41-60","Own 21-40","Own 1-20"],
                                  include_lowest=True)

    # score buckets (clean labels)
    out["score_bucket"] = pd.cut(out["score_differential"],
                                 bins=[-100,-9,-4,-1,0,1,4,8,100],
                                 labels=["≤-9","-8..-4","-3..-1","Tie","1..3","4..7","8..99","100+"],
                                 include_lowest=True)

    out["quarter_clock_min"] = (out["quarter_seconds_remaining"] / 60.0)
    out["clock_bucket_qtr"] = pd.cut(out["quarter_seconds_remaining"],
                                     bins=[-1,30,120,300,600,900],
                                     labels=["≤0:30","0:31–2:00","2:01–5:00","5:01–10:00","10:01–15:00"])

    out["endgame_flag"] = ((out["qtr"] >= 4) & (out["game_seconds_remaining"] <= 120)).astype("Int8")
    out["two_minute_drill"] = (out["quarter_seconds_remaining"] <= 120).astype("Int8")

    out["timeouts_net"] = (out["posteam_timeouts_remaining"] - out["defteam_timeouts_remaining"]).astype("Int8")
    out["off_no_timeouts"] = (out["posteam_timeouts_remaining"] == 0).astype("Int8")
    out["def_no_timeouts"] = (out["defteam_timeouts_remaining"] == 0).astype("Int8")

    # FG range bins (wider upper bound to avoid NA)
    out["fg_range_bin"] = pd.cut(
        out["est_fg_distance"],
        bins=[0,34,39,44,49,54,59,130],
        labels=["<35","35-39","40-44","45-49","50-54","55-59","60+"],
        include_lowest=True
    )

    # availability string + size
    def _avail(row):
        return ",".join([lbl for lbl, f in
                        [("GO",row["available_go"]),("FG",row["available_fg"]),("PUNT",row["available_punt"])]
                        if f == 1]) or "None"
    out["available_actions"] = out.apply(_avail, axis=1)
    out["n_actions_available"] = (
        out["available_go"].astype("Int8") +
        out["available_fg"].astype("Int8") +
        out["available_punt"].astype("Int8")
    )

    # agreement (pure pandas cast → nullable Int8)
    out["agreement"] = (
        out["actual_choice"].notna()
        & out["model_choice"].notna()
        & (out["actual_choice"].str.upper() == out["model_choice"].str.upper())
    ).astype("Int8")

    return out


def main():
    ap = argparse.ArgumentParser(description="Build tidy 4th-down EDA dataset (no cleaning).")
    ap.add_argument("--in-dir", default="data/processed", help="dir with fourth_probs_<YEAR>.parquet")
    ap.add_argument("--seasons", default=None, help='e.g. "2014-2024" or "2019,2021"')
    ap.add_argument("--out-parquet", default="data/processed/4th_down_data.parquet")
    args = ap.parse_args()

    years = parse_seasons(args.seasons)
    files = find_parquets(args.in_dir, years)
    if not files:
        raise SystemExit("No input files matching *ourth_probs_<YEAR>.parquet or fourth_probs_<YEAR>.parquet found.")

    frames = [derive_features(load_year(p)) for p in files]
    tidy = pd.concat(frames, ignore_index=True)

    # final pass: drop columns that are all-NA across ALL seasons loaded
    all_na_cols = [c for c in tidy.columns if tidy[c].isna().all()]
    if all_na_cols:
        tidy = tidy.drop(columns=all_na_cols)

    Path(args.out_parquet).parent.mkdir(parents=True, exist_ok=True)
    tidy.to_parquet(args.out_parquet, index=False)

    # quick console summary
    print(f"Wrote {args.out_parquet} with {len(tidy):,} rows")
    avail = tidy[["available_go","available_fg","available_punt"]].mean().round(3).to_dict()
    print("Availability (any season):", avail)
    if "agreement" in tidy.columns:
        print("Agreement (actual vs model):",
              tidy.loc[tidy["actual_choice"].notna(), "agreement"].mean().round(3)) # type: ignore


if __name__ == "__main__":
    main()
