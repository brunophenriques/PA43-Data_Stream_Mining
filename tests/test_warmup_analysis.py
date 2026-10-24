"""Methodological checks for temporal features and warm-up scoring."""

import unittest

import numpy as np
import pandas as pd
from river import metrics

from src.features import add_representation_b, add_target
from src.stream_analysis import (
    CANDIDATES, FEATURES_B, load_streams, model_factories, run_stream, summarize, validate_raw,
)


class WarmupChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.streams, cls.quality = load_streams()

    def test_shared_timestamps_and_boundaries(self):
        for reps in self.streams.values():
            pd.testing.assert_series_equal(reps["A"]["time"], reps["B"]["time"])
            self.assertEqual(len(reps["A"]), 43800)
            self.assertEqual(
                [int(reps["A"]["time"].ge(t).sum()) for t in CANDIDATES.values()],
                [43800, 43104, 41664, 39504, 35064],
            )

    def test_current_and_future_values_cannot_change_current_features(self):
        raw = pd.DataFrame({
            "temperature_2m": np.arange(80, dtype=float),
            "relative_humidity_2m": np.arange(80, dtype=float),
            "precipitation": np.arange(80, dtype=float) / 10,
            "wind_speed_10m": np.arange(80, dtype=float),
        })
        original = add_representation_b(add_target(raw))
        changed = raw.copy()
        changed.loc[40:, :] += 100
        changed = add_representation_b(add_target(changed))
        pd.testing.assert_frame_equal(original.loc[:40, FEATURES_B], changed.loc[:40, FEATURES_B])
        self.assertAlmostEqual(original.loc[40, "humidity_mean_6h"], raw.loc[34:39, "relative_humidity_2m"].mean())
        self.assertEqual(original.loc[40, "temperature_delta"], 1)

    def test_predict_before_learn_and_fallback(self):
        class RememberLast:
            def __init__(self):
                self.last = None
                self.calls = []

            def predict_one(self, x):
                self.calls.append("predict")
                return self.last

            def learn_one(self, x, y):
                self.calls.append("learn")
                self.last = y

        model = RememberLast()
        frame = self.streams["Faro"]["B"].head(4).copy()
        frame["rain_event"] = [1, 0, 1, 0]
        result = run_stream(frame, FEATURES_B, model)
        self.assertEqual(model.calls, ["predict", "learn"] * 4)
        self.assertEqual(result["predicted"].tolist(), [0, 1, 0, 1])
        self.assertEqual(result["fallback"].tolist(), [True, False, False, False])
        self.assertEqual(result["outcome"].tolist(), ["fn", "fp", "fn", "fp"])

    def test_scoring_boundary_does_not_change_predictions(self):
        # Include dry and rainy observations in a chronological slice.
        frame = self.streams["Angra"]["B"].head(240)
        for name, factory in model_factories().items():
            with self.subTest(model=name):
                early = run_stream(frame, FEATURES_B, factory())
                late = run_stream(frame, FEATURES_B, factory(), score_from=frame["time"].iloc[100])
                pd.testing.assert_series_equal(early["predicted"], late["predicted"])
                pd.testing.assert_series_equal(early["fallback"], late["fallback"])
                self.assertEqual(late["scored"].sum(), 140)
                self.assertEqual(summarize(early.iloc[100:])["macro_f1"],
                                 summarize(late.loc[late["scored"]])["macro_f1"])

    def test_metric_reconstruction_against_river(self):
        frame = pd.DataFrame({
            "time": pd.date_range("2021-01-02", periods=8, freq="h", tz="UTC"),
            "actual": [0, 0, 0, 1, 1, 1, 0, 1],
            "predicted": [0, 1, 0, 1, 0, 1, 0, 1],
            "fallback": False, "prediction_seconds": 0.001, "update_seconds": 0.002,
        })
        result = summarize(frame)
        references = {"accuracy": metrics.Accuracy(), "macro_f1": metrics.MacroF1(),
                      "precision": metrics.Precision(), "recall": metrics.Recall()}
        for y, p in zip(frame["actual"], frame["predicted"]):
            for metric in references.values():
                metric.update(y, p)
        for name, metric in references.items():
            self.assertAlmostEqual(result[name], metric.get())
        self.assertEqual(sum(result[name] for name in ["tn", "fp", "fn", "tp"]), 8)

    def test_bad_raw_data_is_rejected(self):
        from src.stream_analysis import ROOT
        frame = pd.read_csv(ROOT / "data/raw/faro_2021_2025.csv")
        with self.assertRaises(ValueError):
            validate_raw(frame.iloc[::-1], "Faro")
        broken = frame.copy()
        broken.loc[10, "precipitation"] = -1
        with self.assertRaises(ValueError):
            validate_raw(broken, "Faro")
        broken = frame.copy()
        broken.loc[10, "relative_humidity_2m"] = np.nan
        with self.assertRaises(ValueError):
            validate_raw(broken, "Faro")


if __name__ == "__main__":
    unittest.main()
