"""Check transition alignment, conditional denominators and empty situations."""

import unittest

import numpy as np
import pandas as pd

from src.stream_analysis import (
    TRANSITIONS, annotate_predictions, classify_observations, confusion_counts, summarize_transitions,
)


class RainTransitionChecks(unittest.TestCase):
    def setUp(self):
        self.raw = pd.DataFrame({
            "time": pd.date_range("2021-12-31 22:00", periods=7, freq="h", tz="UTC"),
            "precipitation": [0, 0.1, 0.2, 0, 0, 0.1, 0],
        })
        self.observations = classify_observations(self.raw).assign(city="Faro")
        self.predictions = pd.DataFrame({
            "city": "Faro", "representation": "B", "model": "Example",
            "time": self.observations["time"], "actual": self.observations["observed_rain"],
            "predicted": [1, 0, 1, 0, 1, 0], "fallback": False,
        })
        self.start = self.raw["time"].iloc[0]
        self.end = self.raw["time"].iloc[-1]

    def annotated(self):
        return annotate_predictions(self.predictions, self.observations, self.start, self.end)

    def test_four_situations_and_exact_threshold(self):
        self.assertEqual(self.observations["transition"].tolist(), [
            "Rain onset", "Continued rain", "Rain cessation", "Continued dry weather",
            "Rain onset", "Rain cessation",
        ])
        self.assertEqual(self.observations["observed_rain"].tolist(), [1, 1, 0, 0, 1, 0])
        self.assertEqual(len(self.observations), len(self.raw) - 1)
        self.assertEqual(set(self.observations["transition"]), set(TRANSITIONS))

    def test_previous_year_hour_is_preserved_before_filter(self):
        selected = annotate_predictions(self.predictions, self.observations,
                                        pd.Timestamp("2022-01-01", tz="UTC"), self.end)
        row = selected.iloc[0]
        self.assertEqual(row["previous_time"], pd.Timestamp("2021-12-31 23:00", tz="UTC"))
        self.assertEqual(row["previous_rain"], 1)
        self.assertEqual(row["transition"], "Continued rain")

    def test_counts_rates_and_error_denominators(self):
        frame = self.annotated()
        table = pd.DataFrame(summarize_transitions(frame)).set_index("transition")
        self.assertEqual(confusion_counts(frame), {"tn": 2, "fp": 1, "fn": 1, "tp": 2})
        self.assertEqual(table["n_events"].sum(), 6)
        self.assertEqual(table.loc["Rain onset", "detection_rate"], 1)
        self.assertEqual(table.loc["Continued rain", "detection_rate"], 0)
        self.assertEqual(table.loc["Rain cessation", "false_alarm_rate"], 0.5)
        self.assertEqual(table.loc["Continued dry weather", "false_alarm_rate"], 0)
        self.assertEqual(table.loc["Continued rain", "error_share_within_type"], 1)
        self.assertEqual(table.loc["Rain cessation", "error_share_within_type"], 1)
        self.assertTrue(np.isnan(table.loc["Rain onset", "false_alarm_rate"]))
        self.assertTrue(np.isnan(table.loc["Rain cessation", "detection_rate"]))
        for name, count in confusion_counts(frame).items():
            self.assertEqual(table[name].sum(), count)

    def test_empty_situations_and_no_rain_are_undefined(self):
        raw = self.raw.assign(precipitation=0)
        observations = classify_observations(raw).assign(city="Faro")
        predictions = self.predictions.assign(actual=0, predicted=0)
        frame = annotate_predictions(predictions, observations, self.start, self.end)
        table = pd.DataFrame(summarize_transitions(frame)).set_index("transition")
        self.assertTrue(table["detection_rate"].isna().all())
        self.assertTrue(table["error_share_within_type"].isna().all())
        self.assertTrue(table.loc["Rain onset", ["detection_rate", "false_alarm_rate", "error_rate"]].isna().all())
        self.assertEqual(table.loc["Continued dry weather", "false_alarm_rate"], 0)
        empty = pd.DataFrame(summarize_transitions(frame.iloc[:0]))
        self.assertEqual(empty["n_events"].sum(), 0)
        self.assertTrue(empty[["detection_rate", "false_alarm_rate", "error_rate", "error_share_within_type"]].isna().all().all())

    def test_invalid_chronology_and_precipitation_are_rejected(self):
        for invalid in [self.raw.iloc[::-1], self.raw.drop(index=2),
                        pd.concat([self.raw, self.raw.iloc[-1:]], ignore_index=True),
                        self.raw.assign(precipitation=np.nan), self.raw.assign(precipitation=-1)]:
            with self.subTest(rows=len(invalid)):
                with self.assertRaises(ValueError):
                    classify_observations(invalid)

    def test_missing_lookup_wrong_target_and_duplicate_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            annotate_predictions(self.predictions, self.observations.iloc[1:], self.start, self.end)
        with self.assertRaises(ValueError):
            annotate_predictions(self.predictions.assign(actual=0), self.observations, self.start, self.end)
        duplicate = pd.concat([self.predictions, self.predictions.iloc[:1]], ignore_index=True)
        with self.assertRaises(ValueError):
            annotate_predictions(duplicate, self.observations, self.start, self.end)


if __name__ == "__main__":
    unittest.main()
