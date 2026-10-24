"""Shared streaming pipeline and compact exploratory diagnostics for PA43.

Importing this module never trains models. All notebooks share the same current
prediction store. Only an explicit experiment command creates new histories.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pandas as pd
from river import ensemble, forest, linear_model, naive_bayes, tree

from .features import add_representation_b, add_target

ROOT = Path(__file__).resolve().parents[1]
FINAL_OUTPUT = ROOT / "results" / "final"
EXPLORATORY_OUTPUT = ROOT / "results" / "exploratory"
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2025-12-31 23:00", tz="UTC")
FINAL_START = START + pd.Timedelta(days=30)
COMMON_START = pd.Timestamp("2022-01-01", tz="UTC")
HISTORY_HOURS = 24
SEED = 42
# These were the effective depths in the original runs. Pin them explicitly
# instead of deriving a default from the process recursion limit.
MAX_DEPTH = 980
WEATHER = ["temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
FEATURES_A = ["temperature_lag1", "humidity_lag1", "precipitation_lag1", "wind_lag1"]
FEATURES_B = FEATURES_A + ["temperature_delta", "humidity_mean_6h", "precipitation_std_6h", "rain_events_24h"]
CALENDAR_FEATURES = ["local_hour_sin", "local_hour_cos", "annual_phase_sin", "annual_phase_cos"]
FEATURES = {"A": FEATURES_A, "B": FEATURES_B, "B_calendar": FEATURES_B + CALENDAR_FEATURES}
VARIANTS = list(FEATURES)
CITIES = {"Faro": "faro", "Angra": "angra"}
ZONES = {"Faro": "Europe/Lisbon", "Angra": "Atlantic/Azores"}
GROUPS = ["city", "representation", "model"]
SCORES = ["accuracy", "macro_f1", "precision", "recall"]
CANDIDATES = {
    "0 days": START + pd.Timedelta(hours=24),
    "30 days": FINAL_START,
    "90 days": START + pd.Timedelta(days=90),
    "180 days": START + pd.Timedelta(days=180),
    "Calendar year 2021": COMMON_START,
}
TRANSITIONS = {
    "Rain onset": (0, 1), "Continued rain": (1, 1),
    "Rain cessation": (1, 0), "Continued dry weather": (0, 0),
}
COUNTS = ["n_events", "tn", "fp", "fn", "tp", "n_errors"]


def model_factories():
    """Fresh instances; River uses seed, not sklearn's random_state argument."""
    return {
        "GaussianNB": lambda: naive_bayes.GaussianNB(),
        "LogisticRegression": lambda: linear_model.LogisticRegression(),
        "HoeffdingTree": lambda: tree.HoeffdingTreeClassifier(max_depth=MAX_DEPTH),
        "OnlineBagging": lambda: ensemble.BaggingClassifier(
            model=tree.HoeffdingTreeClassifier(max_depth=MAX_DEPTH), n_models=10, seed=SEED),
        "AdaptiveRandomForest": lambda: forest.ARFClassifier(n_models=10, max_depth=MAX_DEPTH, seed=SEED),
    }



def add_calendar_features(frame, city):
    """Add only calendar information known before the target is observed.

    The annual phase uses the local date, with January 1 at phase zero and a
    denominator of 365 or 366 for that local year. It is constant within a local
    day. Hourly phase follows local civil time, including daylight saving time.
    """
    result = frame.copy()
    times = pd.to_datetime(result["time"], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("Calendar features require valid prediction timestamps")
    local = times.dt.tz_convert(ZONES[city])
    hour = local.dt.hour + local.dt.minute / 60 + local.dt.second / 3600
    hour_phase = 2 * np.pi * hour / 24
    year_phase = 2 * np.pi * (local.dt.dayofyear - 1) / np.where(local.dt.is_leap_year, 366, 365)
    result["local_hour_sin"] = np.sin(hour_phase)
    result["local_hour_cos"] = np.cos(hour_phase)
    result["annual_phase_sin"] = np.sin(year_phase)
    result["annual_phase_cos"] = np.cos(year_phase)
    return result


def validate_raw(frame, city):
    """Reject invalid inputs; never silently sort, impute, or remove weather."""
    if not set(["time"] + WEATHER).issubset(frame.columns):
        raise ValueError(f"{city}: missing required columns")
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    expected = pd.date_range(START, END, freq="h")
    if not pd.DatetimeIndex(frame["time"]).equals(expected):
        raise ValueError(f"{city}: expected unique, ordered, complete hourly UTC timestamps")
    if not np.isfinite(frame[WEATHER].to_numpy(dtype=float)).all():
        raise ValueError(f"{city}: missing or non-finite weather values")
    if not frame["relative_humidity_2m"].between(0, 100).all():
        raise ValueError(f"{city}: relative humidity outside [0, 100]")
    if frame[["precipitation", "wind_speed_10m"]].lt(0).any().any():
        raise ValueError(f"{city}: negative precipitation or wind speed")
    return frame


def load_streams(root=ROOT):
    """Build all representations from raw data using shared eligibility."""
    streams, checks = {}, []
    for city, slug in CITIES.items():
        raw = validate_raw(pd.read_csv(Path(root) / "data" / "raw" / f"{slug}_2021_2025.csv"), city)
        full = add_calendar_features(add_representation_b(add_target(raw)), city)
        eligible = full.iloc[HISTORY_HOURS:].reset_index(drop=True)
        if not np.isfinite(eligible[FEATURES["B_calendar"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{city}: incomplete features after the shared history interval")
        streams[city] = {
            rep: eligible[["time", "rain_event"] + columns].copy()
            for rep, columns in FEATURES.items()
        }
        checks.append({
            "city": city, "raw_hours": len(raw), "feature_history_hours": HISTORY_HOURS,
            "eligible_hours": len(eligible), "first_prediction": eligible["time"].iloc[0],
            "last_prediction": eligible["time"].iloc[-1],
            "rain_hours": int(eligible["rain_event"].sum()),
            "rain_rate": eligible["rain_event"].mean(), "quality_checks_passed": True,
            "representations_aligned": streams[city]["A"]["time"].equals(streams[city]["B"]["time"]),
        })
    return streams, pd.DataFrame(checks)


def run_stream(frame, features, model, score_from=None):
    """Predict, evaluate, then learn once per eligible hour.

    score_from only marks records for scoring; prediction and learning are
    unchanged. File writes and aggregation are excluded from model timings.
    """
    records = []
    for row in frame[["time", "rain_event"] + features].itertuples(index=False, name=None):
        timestamp, actual, *values = row
        x, actual = dict(zip(features, values)), int(actual)
        start = perf_counter()
        predicted = model.predict_one(x)
        prediction_seconds = perf_counter() - start
        fallback = predicted is None
        predicted = 0 if fallback else int(predicted)
        if predicted not in (0, 1):
            raise ValueError(f"Unexpected prediction: {predicted}")
        # Evaluate before exposing the current label to learn_one.
        outcome = ("tn" if predicted == 0 else "fp") if actual == 0 else (
            "fn" if predicted == 0 else "tp"
        )
        scored = score_from is None or timestamp >= score_from
        start = perf_counter()
        model.learn_one(x, actual)
        update_seconds = perf_counter() - start
        records.append((timestamp, actual, predicted, fallback, outcome,
                        prediction_seconds, update_seconds, scored))
    return pd.DataFrame(records, columns=[
        "time", "actual", "predicted", "fallback", "outcome",
        "prediction_seconds", "update_seconds", "scored",
    ])


def package_versions():
    names = ["river", "pandas", "numpy", "scipy", "matplotlib", "nbformat", "nbclient", "ipykernel"]
    return {"python": platform.python_version(), **{
        name: importlib.metadata.version(name) for name in names
    }}


def summarize(frame):
    """Compute binary metrics without selecting a winning warm-up."""
    if frame.empty:
        raise ValueError("Cannot score an empty interval")
    y, p = frame["actual"].to_numpy(), frame["predicted"].to_numpy()
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    tp = int(((y == 1) & (p == 1)).sum())

    def ratio(a, b):
        return a / b if b else 0.0

    return {
        "evaluation_start": frame["time"].min(), "evaluation_end": frame["time"].max(),
        "n_events": len(frame), "rain_hours": tp + fn, "dry_hours": tn + fp,
        "rain_rate": (tp + fn) / len(frame), "fallback_count": int(frame["fallback"].sum()),
        "accuracy": (tp + tn) / len(frame),
        "macro_f1": (ratio(2 * tp, 2 * tp + fp + fn) + ratio(2 * tn, 2 * tn + fp + fn)) / 2,
        "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "both_classes_present": bool(tp + fn and tn + fp),
        "recall_display": ratio(tp, tp + fn) if tp + fn else np.nan,
        "precision_display": ratio(tp, tp + fp) if tp + fp else np.nan,
        "macro_f1_display": ((ratio(2 * tp, 2 * tp + fp + fn) + ratio(2 * tn, 2 * tn + fp + fn)) / 2)
            if tp + fn and tn + fp else np.nan,
        "prediction_time_per_event": frame["prediction_seconds"].mean(),
        "update_time_per_event": frame["update_seconds"].mean(),
    }


def summarize_periods(frame, period):
    """Recompute metrics for UTC calendar months or years, preserving support."""
    if period not in ("month", "year"):
        raise ValueError("period must be month or year")
    rows = []
    for key, group in frame.groupby(GROUPS, sort=False):
        times = group["time"]
        labels = times.dt.strftime("%Y-%m" if period == "month" else "%Y")
        for label, selected in group.groupby(labels, sort=True):
            period_start = pd.Timestamp(label + ("-01" if period == "month" else "-01-01"), tz="UTC")
            period_end = period_start + (pd.offsets.MonthBegin(1) if period == "month" else pd.offsets.YearBegin(1))
            expected_hours = int((period_end - period_start) / pd.Timedelta(hours=1))
            rows.append({
                **dict(zip(GROUPS, key)), "period": label, "year": int(label[:4]),
                **({"month": int(label[5:7])} if period == "month" else {}),
                "complete_period": len(selected) == expected_hours,
                **summarize(selected),
            })
    return pd.DataFrame(rows)


def paired_deltas(overall):
    """Subtract B from B_calendar, pairing exactly by city and learner."""
    numeric = SCORES + ["tn", "fp", "fn", "tp", "fallback_count",
                        "prediction_time_per_event", "update_time_per_event"]
    reference = overall.loc[overall["representation"].eq("B")].set_index(["city", "model"])
    calendar = overall.loc[overall["representation"].eq("B_calendar")].set_index(["city", "model"])
    if not reference.index.equals(calendar.index):
        raise AssertionError("Unpaired comparison")
    for name in ["n_events", "rain_hours", "dry_hours", "evaluation_start", "evaluation_end"]:
        pd.testing.assert_series_equal(reference[name], calendar[name])
    result = calendar[numeric].subtract(reference[numeric]).add_prefix("delta_")
    for name in ["n_events", "rain_hours", "dry_hours"]:
        result[name] = reference[name]
    return result.reset_index()


def classify_observations(raw):
    """Construct hourly observed transitions BEFORE selecting an interval.

    The first raw observation has no preceding observation and is excluded.
    Invalid/gapped chronology is rejected instead of treating an earlier row as
    the previous hour. Synthetic sequences may be shorter than the full dataset.
    """
    frame = raw[["time", "precipitation"]].copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if frame["time"].isna().any() or frame["time"].duplicated().any():
        raise ValueError("Transition timestamps must be valid and unique")
    if not frame["time"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Transitions require a complete ordered hourly sequence")
    if not np.isfinite(frame["precipitation"].to_numpy(dtype=float)).all() or frame["precipitation"].lt(0).any():
        raise ValueError("Precipitation must be finite and nonnegative")
    frame["observed_rain"] = frame["precipitation"].ge(0.1).astype(int)
    frame["previous_rain"] = frame["observed_rain"].shift(1)
    frame["previous_time"] = frame["time"].shift(1)
    frame = frame.iloc[1:].copy()
    frame["previous_rain"] = frame["previous_rain"].astype(int)
    reverse = {pair: name for name, pair in TRANSITIONS.items()}
    frame["transition"] = [reverse[(previous, current)] for previous, current in
                           zip(frame["previous_rain"], frame["observed_rain"])]
    return frame.drop(columns="precipitation").reset_index(drop=True)


def load_observations(root=ROOT):
    rows = []
    for city, slug in CITIES.items():
        raw = pd.read_csv(Path(root) / "data" / "raw" / f"{slug}_2021_2025.csv")
        raw = validate_raw(raw, city)
        frame = classify_observations(raw)
        frame.insert(0, "city", city)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def confusion_counts(frame):
    y, p = frame["actual"].to_numpy(), frame["predicted"].to_numpy()
    return {
        "tn": int(((y == 0) & (p == 0)).sum()),
        "fp": int(((y == 0) & (p == 1)).sum()),
        "fn": int(((y == 1) & (p == 0)).sum()),
        "tp": int(((y == 1) & (p == 1)).sum()),
    }


def summarize_transitions(frame):
    """Return all four groups, including empty groups and undefined rates.

    Detection rate = TP / (TP + FN), only for rainy situations.
    False-alarm rate = FP / (TN + FP), only for dry situations.
    Error share uses total FN or total FP across the parent interval; it is not
    the within-situation error rate. Zero denominators always produce NaN.
    """
    totals = confusion_counts(frame)
    rows = []
    for name, (previous, actual) in TRANSITIONS.items():
        selected = frame.loc[frame["transition"].eq(name)]
        counts = confusion_counts(selected)
        n = len(selected)
        error_type = "FN" if actual == 1 else "FP"
        errors = counts[error_type.lower()]
        error_total = totals[error_type.lower()]
        rows.append({
            "transition": name, "previous_rain": previous, "observed_rain": actual,
            "n_events": n, **counts, "n_errors": errors,
            "detection_rate": counts["tp"] / n if actual == 1 and n else np.nan,
            "false_alarm_rate": counts["fp"] / n if actual == 0 and n else np.nan,
            "error_rate": errors / n if n else np.nan,
            "error_type": error_type, "error_type_total": error_total,
            "error_share_within_type": errors / error_total if error_total else np.nan,
        })
    return rows


def annotate_predictions(predictions, observations, start=COMMON_START, end=END):
    """Join the already constructed transition lookup to selected predictions."""
    selected = predictions.loc[predictions["time"].between(start, end)].copy()
    if selected.empty:
        raise ValueError("No predictions in the requested diagnostic interval")
    if selected.duplicated(GROUPS + ["time"]).any():
        raise ValueError("Duplicate prediction identity/timestamp")
    if not selected[["actual", "predicted"]].isin([0, 1]).all().all():
        raise ValueError("Expected binary actual and predicted labels")
    joined = selected.merge(observations, on=["city", "time"], how="left", validate="many_to_one")
    if joined[["observed_rain", "previous_rain", "previous_time", "transition"]].isna().any().any():
        raise ValueError("Missing current or previous observed hour for a prediction")
    if not joined["actual"].eq(joined["observed_rain"]).all():
        raise ValueError("Saved target differs from the raw-data T1 target")
    if not joined["previous_time"].eq(joined["time"] - pd.Timedelta(hours=1)).all():
        raise ValueError("Previous observations must be exactly one hour earlier")
    return joined


def build_transition_tables(annotated):
    overall, monthly, frequency = [], [], []
    for key, group in annotated.groupby(GROUPS, sort=False):
        identity = dict(zip(GROUPS, key))
        interval = {"evaluation_start": group["time"].min(), "evaluation_end": group["time"].max()}
        overall.extend({**identity, **interval, **row} for row in summarize_transitions(group))
        labels = group["time"].dt.strftime("%Y-%m")
        for month, selected in group.groupby(labels, sort=True):
            interval = {"period": month, "year": int(month[:4]), "month": int(month[5:]),
                        "evaluation_start": selected["time"].min(), "evaluation_end": selected["time"].max()}
            monthly.extend({**identity, **interval, **row} for row in summarize_transitions(selected))
    observations = annotated.drop_duplicates(["city", "time"])
    for city, group in observations.groupby("city", sort=False):
        for name in TRANSITIONS:
            n = int(group["transition"].eq(name).sum())
            frequency.append({"city": city, "transition": name, "n_events": n,
                              "city_hours": len(group), "frequency": n / len(group)})
    return {"transition_frequency": pd.DataFrame(frequency),
            "transition_metrics": pd.DataFrame(overall), "monthly_transition_metrics": pd.DataFrame(monthly)}


def validate_transitions(annotated, tables):
    expected = pd.date_range(COMMON_START, END, freq="h")
    expected_keys = {(city, rep, model) for city in CITIES for rep in VARIANTS
                     for model in model_factories()}
    grouped = dict(tuple(annotated.groupby(GROUPS, sort=False)))
    assert set(grouped) == expected_keys
    reconstruction = []
    overall = tables["transition_metrics"]
    monthly = tables["monthly_transition_metrics"]
    for key, group in grouped.items():
        assert pd.DatetimeIndex(group["time"]).equals(expected)
        rows = overall.loc[np.logical_and.reduce([overall[name].eq(value) for name, value in zip(GROUPS, key)])]
        assert set(rows["transition"]) == set(TRANSITIONS) and len(rows) == 4
        assert rows["n_events"].sum() == len(expected)
        direct = confusion_counts(group)
        record = {**dict(zip(GROUPS, key)), "n_events": len(group)}
        for name, count in direct.items():
            summed = int(rows[name].sum())
            assert summed == count
            record[f"direct_{name}"] = count
            record[f"summed_{name}"] = summed
        record["matches"] = True
        reconstruction.append(record)
    assert annotated["previous_time"].eq(annotated["time"] - pd.Timedelta(hours=1)).all()
    for name, (previous, current) in TRANSITIONS.items():
        mask = annotated["previous_rain"].eq(previous) & annotated["actual"].eq(current)
        assert mask.equals(annotated["transition"].eq(name)), "Transition assignment differs from observations"
    assert annotated.groupby(["city", "time"])["actual"].nunique().eq(1).all()
    assert overall.groupby(["city", "transition"])["n_events"].nunique().eq(1).all()
    assert monthly.groupby(["city", "period", "transition"])["n_events"].nunique().eq(1).all()
    for table in (overall, monthly):
        assert table[["tn", "fp", "fn", "tp"]].sum(axis=1).eq(table["n_events"]).all()
        rainy = table["observed_rain"].eq(1)
        assert table.loc[rainy, "false_alarm_rate"].isna().all()
        assert table.loc[~rainy, "detection_rate"].isna().all()
        empty = table["n_events"].eq(0)
        assert table.loc[empty, ["detection_rate", "false_alarm_rate", "error_rate"]].isna().all().all()
        for metric in ["detection_rate", "false_alarm_rate", "error_rate", "error_share_within_type"]:
            defined = table[metric].dropna()
            assert defined.between(0, 1).all()
        assert table.loc[table["error_type_total"].eq(0), "error_share_within_type"].isna().all()
    summed_months = monthly.groupby(GROUPS + ["transition"])[COUNTS].sum().sort_index()
    parent = overall.set_index(GROUPS + ["transition"])[COUNTS].sort_index()
    pd.testing.assert_frame_equal(summed_months, parent)
    for _, row in overall.iterrows():
        n = row["n_events"]
        expected_detection = row["tp"] / n if row["observed_rain"] == 1 and n else np.nan
        expected_false_alarm = row["fp"] / n if row["observed_rain"] == 0 and n else np.nan
        assert np.isclose(row["detection_rate"], expected_detection, equal_nan=True)
        assert np.isclose(row["false_alarm_rate"], expected_false_alarm, equal_nan=True)
    error_totals = overall.groupby(GROUPS + ["error_type"])
    for _, group in error_totals:
        if group["error_type_total"].iloc[0]:
            assert np.isclose(group["error_share_within_type"].sum(), 1)
    frequency = tables["transition_frequency"]
    assert frequency.groupby("city")["n_events"].sum().eq(len(expected)).all()
    return pd.DataFrame(reconstruction), pd.DataFrame({
        "check": ["30 cached histories with the same 35,064 UTC hours",
                  "Previous observation is exactly one hour earlier, including year boundaries",
                  "Every observation belongs to exactly one transition",
                  "Identical target and transition support across all representations and models",
                  "Global confusion matrices reconstructed from four situations",
                  "Monthly partitions reconstruct every overall transition count",
                  "Undefined rates remain missing; defined rates match their denominators",
                  "FN and FP contributions use their separate total-error denominators"],
        "passed": True,
    })


def configuration(root=ROOT):
    """Stable inputs/settings signature; wall-clock metadata belongs to the run."""
    root = Path(root)
    sources = [root / "src" / name for name in
               ["stream_analysis.py", "features.py", "experiment.py", "prepare_data.py"]]
    data = [root / "data" / "raw" / f"{slug}_2021_2025.csv" for slug in CITIES.values()]
    params = json.loads(json.dumps(
        {name: factory()._get_params() for name, factory in model_factories().items()},
        default=lambda value: f"{value.__module__}.{value.__qualname__}"
        if isinstance(value, type) else repr(value),
    ))
    return {
        "schema_version": 1, "seed": SEED,
        "protocol": "predict -> evaluate -> learn from first eligible hour; scoring starts after 30 raw-calendar days",
        "raw_start": START.isoformat(), "raw_end": END.isoformat(),
        "feature_history_hours": HISTORY_HOURS, "final_scoring_start": FINAL_START.isoformat(),
        "common_exploratory_start": COMMON_START.isoformat(), "features": FEATURES,
        "target": "precipitation >= 0.1 mm; class 1 = rain", "models": params,
        "fallback": "class 0 when predict_one returns None; every occurrence flagged",
        "metric_policy": "fixed labels [0,1]; zero undefined ratios in aggregate metrics; display columns mask undefined rates and single-class Macro-F1",
        "time_zones": ZONES, "versions": package_versions(),
        "effective_tree_depth": MAX_DEPTH,
        "sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sources + data},
    }


def generate_predictions(output=FINAL_OUTPUT, root=ROOT):
    """Explicit fresh run: 20 required A/B combinations plus 10 calendar ones."""
    output = Path(output)
    directory = output / "predictions"
    directory.mkdir(parents=True, exist_ok=True)
    streams, quality = load_streams(root)
    manifest = {"complete": False, "run_id": str(uuid4()),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "configuration": configuration(root), "prediction_files": {}}
    path = output / "configuration.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    quality.to_csv(output / "data_checks.csv", index=False)
    count = 0
    for city, representations in streams.items():
        for rep, frame in representations.items():
            for name, factory in model_factories().items():
                count += 1
                print(f"[{count}/30] {city} | {rep} | {name}", flush=True)
                history = run_stream(frame, FEATURES[rep], factory(), score_from=FINAL_START)
                for column, value in reversed(list(zip(GROUPS, [city, rep, name]))):
                    history.insert(0, column, value)
                target = directory / f"{city}_{rep}_{name}.csv.gz"
                history.to_csv(target, index=False)
                manifest["prediction_files"][target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
                print(f"  Saved {len(history):,} predictions", flush=True)
    manifest["complete"] = True
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_predictions(output=FINAL_OUTPUT, root=ROOT):
    """Read only the current shared store; never train on a cache miss."""
    output = Path(output)
    path = output / "configuration.json"
    if not path.exists():
        raise FileNotFoundError("No current results. Run: python -m src.experiment --run")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("configuration") != configuration(root):
        raise ValueError("Incomplete or stale current run. Explicitly rerun the experiment.")
    expected = {f"{city}_{rep}_{model}.csv.gz" for city in CITIES for rep in FEATURES
                for model in model_factories()}
    if set(manifest["prediction_files"]) != expected:
        raise ValueError("Expected exactly 30 current prediction histories")
    if {p.name for p in (output / "predictions").glob("*.csv.gz")} != expected:
        raise ValueError("Prediction directory contains missing or unexpected histories")
    frames = []
    for city in CITIES:
        for rep in FEATURES:
            for model in model_factories():
                name = f"{city}_{rep}_{model}.csv.gz"
                target = output / "predictions" / name
                if hashlib.sha256(target.read_bytes()).hexdigest() != manifest["prediction_files"][name]:
                    raise ValueError(f"Prediction checksum mismatch: {name}")
                frame = pd.read_csv(target)
                frame["time"] = pd.to_datetime(frame["time"], utc=True)
                if not ((frame.city == city) & (frame.representation == rep) & (frame.model == model)).all():
                    raise ValueError(f"Incorrect prediction identity: {name}")
                frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def score_interval(predictions, boundary, variants=None):
    selected = predictions.loc[predictions.time.ge(boundary)]
    if variants is not None:
        selected = selected.loc[selected.representation.isin(variants)]
    return pd.DataFrame([{**dict(zip(GROUPS, key)), **summarize(group)}
                         for key, group in selected.groupby(GROUPS, sort=False)])


def build_warmup_table(predictions):
    rows = []
    required = predictions.loc[predictions.representation.isin(["A", "B"])]
    for key, group in required.groupby(GROUPS, sort=False):
        for label, boundary in CANDIDATES.items():
            rows.append({**dict(zip(GROUPS, key)), "warmup": label, "boundary": boundary,
                         "learning_hours_before_scoring": int(group.time.lt(boundary).sum()),
                         **summarize(group.loc[group.time.ge(boundary)])})
    return pd.DataFrame(rows)


def verify_current_results(predictions, tables, transition_checks):
    expected = pd.date_range(START + pd.Timedelta(hours=24), END, freq="h")
    groups = dict(tuple(predictions.groupby(GROUPS, sort=False)))
    assert len(groups) == 30
    for key, frame in groups.items():
        assert pd.DatetimeIndex(frame.time).equals(expected), key
        assert frame[["actual", "predicted"]].isin([0, 1]).all().all()
        assert np.isfinite(frame[["prediction_seconds", "update_seconds"]]).all().all()
        assert frame[["prediction_seconds", "update_seconds"]].ge(0).all().all()
        assert frame.scored.eq(frame.time.ge(FINAL_START)).all()
    observations = load_observations()
    aligned = annotate_predictions(predictions, observations, start=START + pd.Timedelta(hours=24))
    assert len(aligned) == len(predictions)
    final = tables["final_metrics"]
    assert len(final) == 20 and set(final.representation) == {"A", "B"}
    assert final.n_events.eq(43104).all()
    assert final.evaluation_start.eq(FINAL_START).all()
    assert tables["calendar_metrics"].n_events.eq(35064).all()
    for name in ["final_metrics", "warmup_metrics", "calendar_metrics", "final_monthly_metrics", "monthly_baseline_metrics"]:
        table = tables[name]
        assert table[["tn", "fp", "fn", "tp"]].sum(axis=1).eq(table.n_events).all(), name
    for name, boundary_field in [("final_metrics", None), ("warmup_metrics", "boundary"), ("calendar_metrics", None)]:
        for _, row in tables[name].iterrows():
            source = groups[tuple(row[k] for k in GROUPS)]
            boundary = row[boundary_field] if boundary_field else (FINAL_START if name == "final_metrics" else COMMON_START)
            rebuilt = summarize(source.loc[source.time.ge(boundary)])
            for metric in SCORES + ["tn", "fp", "fn", "tp"]:
                assert np.isclose(rebuilt[metric], row[metric]), f"Reconstruction: {name}/{metric}"
    for _, row in tables["warmup_metrics"].iterrows():
        assert row.n_events == int((expected >= row.boundary).sum())
    assert tables["final_monthly_metrics"].groupby(GROUPS).n_events.sum().eq(43104).all()
    monthly = tables["monthly_baseline_metrics"]
    assert monthly.loc[monthly.rain_hours.eq(0), "recall_display"].isna().all()
    assert monthly.loc[~monthly.both_classes_present, "macro_f1_display"].isna().all()
    assert transition_checks.passed.all()
    return pd.DataFrame({"check": [
        "30 complete, aligned histories from one current run",
        "Binary labels match raw T1 targets; valid model timings",
        "20 required combinations with 43,104 final scored hours each",
        "Warm-up boundaries and metric reconstruction verified",
        "Monthly coverage and undefined-metric display policy verified",
        "Calendar comparison uses the same 35,064 common-interval hours",
        "All transition diagnostics and confusion reconstructions verified",
    ], "passed": True})


def refresh_results(predictions, final_output=FINAL_OUTPUT, exploratory_output=EXPLORATORY_OUTPUT):
    """Regenerate small derived tables from the single prediction store."""
    final_output, exploratory_output = Path(final_output), Path(exploratory_output)
    final_output.mkdir(parents=True, exist_ok=True)
    exploratory_output.mkdir(parents=True, exist_ok=True)
    required = predictions.loc[predictions.representation.isin(["A", "B"])]
    final_scored = required.loc[required.time.ge(FINAL_START)]
    calendar = score_interval(predictions, COMMON_START, ["B", "B_calendar"])
    observations = load_observations()
    annotated = annotate_predictions(predictions, observations)
    transition_tables = build_transition_tables(annotated)
    reconstruction, transition_checks = validate_transitions(annotated, transition_tables)
    tables = {
        "final_metrics": score_interval(predictions, FINAL_START, ["A", "B"]),
        "final_monthly_metrics": summarize_periods(final_scored, "month"),
        "warmup_metrics": build_warmup_table(predictions),
        "calendar_metrics": calendar,
        "calendar_deltas": paired_deltas(calendar),
        "monthly_baseline_metrics": summarize_periods(predictions.loc[predictions.representation.eq("B")], "month"),
        **transition_tables, "confusion_reconstruction": reconstruction,
        "transition_verification": transition_checks,
    }
    tables["verification"] = verify_current_results(predictions, tables, transition_checks)
    for name, table in tables.items():
        directory = final_output if name in ["final_metrics", "final_monthly_metrics", "verification"] else exploratory_output
        table.to_csv(directory / f"{name}.csv", index=False, na_rep="NaN")
    (exploratory_output / "source.json").write_text(json.dumps({
        "prediction_store": "../final/predictions", "warmup": "different scoring starts; 30 days selected",
        "calendar_and_transitions": "2022-01-01 through 2025-12-31 UTC; diagnostic interval only",
    }, indent=2), encoding="utf-8")
    return tables

