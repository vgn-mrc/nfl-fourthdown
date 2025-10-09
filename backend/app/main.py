from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sklearn.base import BaseEstimator

try:
    from joblib import load as joblib_load
except Exception:  # fallback via sklearn
    from sklearn.externals.joblib import load as joblib_load  # type: ignore

from tensorflow import keras
from tensorflow.keras import layers as L


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parents[2]
ART_WP = BASE_DIR / "notebooks" / "artifacts_wp"
ART_COMP = BASE_DIR / "notebooks" / "artifacts_comp"
ART_COACH = BASE_DIR / "notebooks" / "artifacts_coach"


# ---------- Utility: environment mapping & feature engineering ----------
ROOF_MAP = {
    "closed": "indoor",
    "dome": "indoor",
    "indoor": "indoor",
    "outdoors": "open_air",
    "open": "open_air",
    "open_air": "open_air",
}

SURFACE_MAP = {
    "grass": "natural",
    "grass ": "natural",
    "natural": "natural",
    "fieldturf": "artificial",
    "sportturf": "artificial",
    "matrixturf": "artificial",
    "astroturf": "artificial",
    "astroplay": "artificial",
    "a_turf": "artificial",
    "artificial": "artificial",
}

TEMP_BINS = [-np.inf, 32, 45, 60, 75, np.inf]
TEMP_LABELS = ["Frigid", "Cold", "Cool", "Mild", "Warm/Hot"]

WIND_BINS = [-1, 0, 5, 10, 15, np.inf]
WIND_LABELS = ["Calm", "Light", "Moderate", "Windy", "VeryWindy"]


def _safe_lower(x: Optional[str]) -> Optional[str]:
    return x.lower() if isinstance(x, str) else x


def map_roof(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    return ROOF_MAP.get(val.lower(), None)


def map_surface(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    return SURFACE_MAP.get(val.lower(), None)


def bin_temp(temp_f: Optional[float], roof: Optional[str]) -> Optional[str]:
    # Indoors default: 70°F if missing
    if roof == "indoor" and temp_f is None:
        temp_f = 70.0
    if temp_f is None or np.isnan(temp_f):
        return None
    idx = np.digitize([temp_f], TEMP_BINS)[0] - 1
    idx = int(np.clip(idx, 0, len(TEMP_LABELS) - 1))
    return TEMP_LABELS[idx]


def bin_wind(wind_mph: Optional[float], roof: Optional[str]) -> Optional[str]:
    # Indoors default: 0 mph if missing
    if roof == "indoor" and wind_mph is None:
        wind_mph = 0.0
    if wind_mph is None or np.isnan(wind_mph):
        return None
    idx = np.digitize([wind_mph], WIND_BINS)[0] - 1
    idx = int(np.clip(idx, 0, len(WIND_LABELS) - 1))
    return WIND_LABELS[idx]


def derive_clock(qtr: int, quarter_seconds_remaining: Optional[int], game_seconds_remaining: Optional[int]) -> Tuple[int, int]:
    if game_seconds_remaining is not None and quarter_seconds_remaining is not None:
        return int(quarter_seconds_remaining), int(game_seconds_remaining)
    if quarter_seconds_remaining is None:
        raise ValueError("quarter_seconds_remaining is required if game_seconds_remaining not provided")
    q = int(qtr)
    qsr = int(quarter_seconds_remaining)
    qsr = int(np.clip(qsr, 0, 900))
    # Regulation assumption: 15 min quarters; OT treat as remaining seconds in current period
    if q <= 4:
        gsr = (4 - q) * 900 + qsr
    else:
        gsr = qsr
    return qsr, gsr


def compute_flags(row: Dict[str, Any]) -> Dict[str, Any]:
    yardline_100 = float(row["yardline_100"])  # 1..99
    ydstogo = float(row["ydstogo"])  # >=1
    qtr = int(row["qtr"])  # 1..5
    game_seconds_remaining = int(row["game_seconds_remaining"])  # >=0
    score_diff = int(row["score_differential"])  # posteam - defteam

    quarter_seconds_remaining = int(row["quarter_seconds_remaining"])  # >=0
    two_minute_drill = 1 if quarter_seconds_remaining <= 120 else 0

    fg_dist_yd = yardline_100 + 17.0
    is_fg_range = 1 if (fg_dist_yd <= 67.0) else 0
    fg_puts_ties_or_leads = 1 if (-3 <= score_diff <= 0) else 0
    is_game_deciding = 1 if (abs(score_diff) <= 8 and game_seconds_remaining <= 120 and qtr == 4) else 0
    is_losing_by_one_score = 1 if (score_diff < 0 and score_diff >= -8) else 0

    # bins for comp model
    # yardline_zone bins=[0,20,40,60,80,100] → labels as in training
    yl = yardline_100
    if yl <= 20:
        yardline_zone = "RZ 0-20"
    elif yl <= 40:
        yardline_zone = "20-40"
    elif yl <= 60:
        yardline_zone = "40-60"
    elif yl <= 80:
        yardline_zone = "60-80"
    else:
        yardline_zone = "80-100"

    # ytg_bin bins=[0,2,5,10,99]
    g = ydstogo
    if g <= 2:
        ytg_bin = "1-2"
    elif g <= 5:
        ytg_bin = "3-5"
    elif g <= 10:
        ytg_bin = "6-10"
    else:
        ytg_bin = "11+"

    return {
        "two_minute_drill": two_minute_drill,
        "fg_dist_yd": fg_dist_yd,
        "is_fg_range": is_fg_range,
        "fg_puts_ties_or_leads": fg_puts_ties_or_leads,
        "is_game_deciding": is_game_deciding,
        "is_losing_by_one_score": is_losing_by_one_score,
        "yardline_zone": yardline_zone,
        "ytg_bin": ytg_bin,
    }


# ---------- Pydantic schemas ----------
class PlayInput(BaseModel):
    # Basic situation
    qtr: int = Field(..., ge=1, le=5)
    quarter_seconds_remaining: Optional[int] = Field(None, ge=0, le=900)
    game_seconds_remaining: Optional[int] = Field(None, ge=0)
    yardline_100: float = Field(..., ge=1, le=99)
    ydstogo: float = Field(..., ge=1)
    score_differential: int  # posteam - defteam
    posteam_timeouts_remaining: int = Field(..., ge=0, le=3)
    defteam_timeouts_remaining: int = Field(..., ge=0, le=3)

    # Season & environment
    season_type: Optional[Literal["REG", "POST"]] = None
    roof: Optional[str] = None  # accepts raw; mapped to {indoor, open_air}
    surface: Optional[str] = None  # accepts raw; mapped to {natural, artificial}
    temp_f: Optional[float] = None
    wind_mph: Optional[float] = None

    @validator("season_type")
    def _upper(cls, v):
        if v is None:
            return v
        return v.upper()

    def to_feature_row(self) -> Dict[str, Any]:
        # Map env
        roof_m = map_roof(self.roof) if self.roof is not None else None
        surface_m = map_surface(self.surface) if self.surface is not None else None
        temp_cat = bin_temp(self.temp_f, roof_m)
        wind_cat = bin_wind(self.wind_mph, roof_m)

        qsr, gsr = derive_clock(self.qtr, self.quarter_seconds_remaining, self.game_seconds_remaining)

        base = {
            "qtr": int(self.qtr),
            "quarter_seconds_remaining": int(qsr),
            "game_seconds_remaining": int(gsr),
            "yardline_100": float(self.yardline_100),
            "ydstogo": float(self.ydstogo),
            "score_differential": int(self.score_differential),
            "posteam_timeouts_remaining": int(self.posteam_timeouts_remaining),
            "defteam_timeouts_remaining": int(self.defteam_timeouts_remaining),
            "season_type": self.season_type or "REG",
            "roof": roof_m or "open_air",
            "surface": surface_m or "natural",
            "temp": temp_cat or "Mild",
            "wind": wind_cat or "Light",
        }
        base.update(compute_flags(base))
        return base


class WpResponse(BaseModel):
    go_wp: float
    fg_wp: float
    punt_wp: float
    best_action: Literal["GO", "FIELD_GOAL", "PUNT"]


class CompResponse(BaseModel):
    fg_make_prob: float
    first_down_prob: float


class CoachResponse(BaseModel):
    policy: Literal["GO", "FIELD_GOAL", "PUNT"]
    probs: Dict[str, float]


class AllResponse(BaseModel):
    wp: WpResponse
    comp: CompResponse
    coach: CoachResponse


# ---------- Artifact loading ----------
class InferenceBundle:
    def __init__(self, name: str, dir_path: Path) -> None:
        self.name = name
        self.dir = dir_path
        self.preprocess: BaseEstimator = joblib_load(self.dir / "preprocess.joblib")
        with open(self.dir / "feature_meta.json", "r") as f:
            self.meta = json.load(f)
        with open(self.dir / "preprocess_feature_names.json", "r") as f:
            self.feature_names = json.load(f)

    def build_model(self) -> keras.Model:
        input_dim = int(self.meta.get("input_dim", len(self.feature_names)))
        inp = L.Input(shape=(input_dim,), name="features")

        if self.name == "wp":
            x = L.Dense(128, activation=_activation(), kernel_regularizer=keras.regularizers.l2(1e-5))(inp)
            x = L.BatchNormalization()(x)
            x = L.Dropout(0.15)(x)
            x = L.Dense(64, activation=_activation(), kernel_regularizer=keras.regularizers.l2(1e-5))(x)
            x = L.BatchNormalization()(x)
            x = L.Dropout(0.10)(x)
            x = L.Dense(32, activation=_activation())(x)
            go_out = L.Dense(1, activation="sigmoid", name="go")(x)
            fg_out = L.Dense(1, activation="sigmoid", name="fg")(x)
            punt_out = L.Dense(1, activation="sigmoid", name="punt")(x)
            model = keras.Model(inp, {"go": go_out, "fg": fg_out, "punt": punt_out}, name="wp_multitask")

        elif self.name == "comp":
            h = L.Dense(96, activation=_activation(), kernel_regularizer=keras.regularizers.l2(1e-5))(inp)
            h = L.BatchNormalization()(h)
            h = L.Dropout(0.10)(h)
            h = L.Dense(48, activation=_activation(), kernel_regularizer=keras.regularizers.l2(1e-5))(h)
            h = L.BatchNormalization()(h)
            h = L.Dropout(0.05)(h)
            fg_out = L.Dense(1, activation="sigmoid", name="fg_make")(h)
            go_out = L.Dense(1, activation="sigmoid", name="go_conv")(h)
            model = keras.Model(inp, {"fg_make": fg_out, "go_conv": go_out}, name="comp_multitask")

        elif self.name == "coach":
            x = L.Dense(128, activation=_activation(), kernel_regularizer=keras.regularizers.l2(1e-5))(inp)
            x = L.BatchNormalization()(x)
            x = L.Dropout(0.20)(x)
            x = L.Dense(64, activation=_activation(), kernel_regularizer=keras.regularizers.l2(1e-5))(x)
            x = L.BatchNormalization()(x)
            x = L.Dropout(0.15)(x)
            x = L.Dense(32, activation=_activation())(x)
            out = L.Dense(3, activation="softmax", name="coach_policy")(x)
            model = keras.Model(inp, out, name="coach_policy_classifier")

        else:
            raise ValueError(f"Unknown bundle name: {self.name}")

        # load weights
        weight_glob = {
            "wp": "wp_multitask.best.weights.h5",
            "comp": "comp_multitask.best.weights.h5",
            "coach": "coach_policy.best.weights.h5",
        }
        weights_path = self.dir / weight_glob[self.name]
        model.load_weights(str(weights_path))
        return model


def _activation():
    try:
        # TF >= 2.13 has gelu
        _ = keras.activations.gelu
        return "gelu"
    except Exception:
        return "relu"


def _make_frame(pi: PlayInput, meta: Dict[str, Any], for_name: str) -> pd.DataFrame:
    row = pi.to_feature_row()
    num_cols = meta.get("numeric_features", [])
    cat_cols = meta.get("categorical_features", [])
    cols = num_cols + cat_cols
    # subset row to available keys; ensure types
    data: Dict[str, Any] = {}
    for c in cols:
        v = row.get(c)
        data[c] = v
    df = pd.DataFrame([data])
    # ensure object dtype for categorical features
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object")
    return df


# ---------- App init ----------
app = FastAPI(title="NFL 4th Down Inference API", version="0.1.0")

# CORS for local Next.js dev and public demo; tighten in prod as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class _State:
    wp: Optional[InferenceBundle] = None
    comp: Optional[InferenceBundle] = None
    coach: Optional[InferenceBundle] = None
    model_wp: Optional[keras.Model] = None
    model_comp: Optional[keras.Model] = None
    model_coach: Optional[keras.Model] = None


S = _State()


@app.on_event("startup")
def _load_all():
    # Load preprocessors + models
    S.wp = InferenceBundle("wp", ART_WP)
    S.comp = InferenceBundle("comp", ART_COMP)
    S.coach = InferenceBundle("coach", ART_COACH)
    S.model_wp = S.wp.build_model()
    S.model_comp = S.comp.build_model()
    S.model_coach = S.coach.build_model()


@app.get("/health")
def health():
    ok = all([S.wp, S.comp, S.coach, S.model_wp, S.model_comp, S.model_coach])
    return {"ok": ok}


@app.get("/metadata")
def metadata():
    return {
        "wp": {"numeric": S.wp.meta.get("numeric_features"), "categorical": S.wp.meta.get("categorical_features"), "input_dim": S.wp.meta.get("input_dim")},
        "comp": {"numeric": S.comp.meta.get("numeric_features"), "categorical": S.comp.meta.get("categorical_features"), "input_dim": S.comp.meta.get("input_dim")},
        "coach": {"numeric": S.coach.meta.get("numeric_features"), "categorical": S.coach.meta.get("categorical_features"), "classes": S.coach.meta.get("class_order"), "input_dim": S.coach.meta.get("input_dim")},
    }


@app.post("/predict/wp", response_model=WpResponse)
def predict_wp(pi: PlayInput):
    df = _make_frame(pi, S.wp.meta, "wp")
    X = S.wp.preprocess.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)
    preds: Dict[str, np.ndarray] = S.model_wp.predict(X, verbose=0)  # type: ignore
    go = float(preds["go"].ravel()[0])
    fg = float(preds["fg"].ravel()[0])
    punt = float(preds["punt"].ravel()[0])
    options = {"GO": go, "FIELD_GOAL": fg, "PUNT": punt}
    best = max(options, key=options.get)
    return WpResponse(go_wp=go, fg_wp=fg, punt_wp=punt, best_action=best)  # type: ignore


@app.post("/predict/comp", response_model=CompResponse)
def predict_comp(pi: PlayInput):
    # season_type required by this model; default set in to_feature_row
    df = _make_frame(pi, S.comp.meta, "comp")
    X = S.comp.preprocess.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)
    preds: Dict[str, np.ndarray] = S.model_comp.predict(X, verbose=0)  # type: ignore
    fg_make = float(preds["fg_make"].ravel()[0])
    go_conv = float(preds["go_conv"].ravel()[0])
    return CompResponse(fg_make_prob=fg_make, first_down_prob=go_conv)


@app.post("/predict/coach", response_model=CoachResponse)
def predict_coach(pi: PlayInput):
    df = _make_frame(pi, S.coach.meta, "coach")
    X = S.coach.preprocess.transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(np.float32)
    probs: np.ndarray = S.model_coach.predict(X, verbose=0)  # type: ignore
    probs = probs.reshape(-1, 3)[0]
    classes = S.coach.meta.get("class_order", ["GO", "FIELD_GOAL", "PUNT"])  # index order
    out = {cls: float(probs[i]) for i, cls in enumerate(classes)}
    policy = max(out, key=out.get)
    return CoachResponse(policy=policy, probs=out)


@app.post("/predict/all", response_model=AllResponse)
def predict_all(pi: PlayInput):
    return AllResponse(wp=predict_wp(pi), comp=predict_comp(pi), coach=predict_coach(pi))
