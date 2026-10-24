"""Check cyclic encodings and the controlled feature comparison."""

import unittest

import numpy as np
import pandas as pd

from src.stream_analysis import (
    CALENDAR_FEATURES, add_calendar_features, load_streams, paired_deltas, summarize_periods,
)


def encoded(times, city="Faro"):
    return add_calendar_features(pd.DataFrame({"time": pd.to_datetime(times, utc=True)}), city)


class SeasonalFeatureChecks(unittest.TestCase):
    def test_hour_cycle_and_city_time_zones(self):
        frame = encoded(["2024-01-01 12:00", "2024-01-02 12:00", "2024-07-01 12:00"])
        np.testing.assert_allclose(frame.loc[0, CALENDAR_FEATURES[:2]].to_numpy(dtype=float),
                                   frame.loc[1, CALENDAR_FEATURES[:2]].to_numpy(dtype=float))
        self.assertAlmostEqual(frame.loc[0, "local_hour_cos"], np.cos(2 * np.pi * 12 / 24))
        self.assertAlmostEqual(frame.loc[2, "local_hour_sin"], np.sin(2 * np.pi * 13 / 24))
        angra = encoded(["2024-01-01 12:00", "2024-07-01 12:00"], "Angra")
        self.assertAlmostEqual(angra.loc[0, "local_hour_sin"], np.sin(2 * np.pi * 11 / 24))
        self.assertAlmostEqual(angra.loc[1, "local_hour_cos"], -1)

    def test_daylight_saving_transitions(self):
        spring = encoded(["2024-03-31 00:00", "2024-03-31 01:00"])
        self.assertAlmostEqual(spring.loc[0, "local_hour_sin"], 0)
        self.assertAlmostEqual(spring.loc[1, "local_hour_sin"], np.sin(2 * np.pi * 2 / 24))
        autumn = encoded(["2024-10-27 00:00", "2024-10-27 01:00"])
        np.testing.assert_allclose(autumn[CALENDAR_FEATURES[:2]].iloc[0],
                                   autumn[CALENDAR_FEATURES[:2]].iloc[1])

    def test_leap_year_and_annual_wrap(self):
        leap = encoded(["2024-02-28 12:00", "2024-02-29 12:00", "2024-03-01 12:00"])
        expected = 2 * np.pi * np.array([58, 59, 60]) / 366
        np.testing.assert_allclose(leap["annual_phase_sin"], np.sin(expected))
        regular = encoded(["2023-03-01 12:00"])
        self.assertAlmostEqual(regular.loc[0, "annual_phase_sin"], np.sin(2 * np.pi * 59 / 365))
        for year in [2023, 2024]:
            frame = encoded([f"{year}-12-31 12:00", f"{year + 1}-01-01 12:00"])
            distance = np.linalg.norm(np.diff(frame[CALENDAR_FEATURES[2:]].to_numpy(), axis=0))
            self.assertLess(distance, 0.02)
            self.assertAlmostEqual(frame.loc[1, "annual_phase_cos"], 1)

    def test_calendar_does_not_read_weather_or_label(self):
        frame = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC"),
                              "rain_event": 0, "humidity_lag1": 10.0})
        changed = frame.assign(rain_event=1, humidity_lag1=9999.0)
        original_features = add_calendar_features(frame, "Faro")[CALENDAR_FEATURES]
        pd.testing.assert_frame_equal(original_features, add_calendar_features(changed, "Faro")[CALENDAR_FEATURES])
        np.testing.assert_allclose(np.square(original_features.iloc[:, :2]).sum(axis=1), 1)
        np.testing.assert_allclose(np.square(original_features.iloc[:, 2:]).sum(axis=1), 1)
        with self.assertRaises(ValueError):
            encoded([pd.NaT])

    def test_original_b_is_unchanged(self):
        from src.stream_analysis import load_streams as load_original
        original, _ = load_original()
        augmented, _ = load_streams()
        for city, reps in augmented.items():
            frame = reps["B_calendar"]
            columns = original[city]["B"].columns
            pd.testing.assert_frame_equal(frame[columns], original[city]["B"])
            self.assertEqual(len(frame), 43800)

    def test_calendar_month_partition_and_paired_support(self):
        times = pd.date_range("2023-01-31 23:00", periods=3, freq="h", tz="UTC")
        records = pd.DataFrame({"city": "Faro", "representation": "B", "model": "Example",
                                "time": times, "actual": [0, 1, 1], "predicted": [0, 0, 1],
                                "fallback": False, "prediction_seconds": 0.1, "update_seconds": 0.2})
        table = summarize_periods(records, "month")
        self.assertEqual(table["n_events"].tolist(), [1, 2])
        self.assertEqual(table["rain_hours"].tolist(), [0, 2])
        self.assertFalse(table["complete_period"].any())
        a = summarize_periods(records, "year").drop(columns=["period", "year", "complete_period"])
        b = a.assign(representation="B_calendar", accuracy=1.0)
        delta = paired_deltas(pd.concat([a, b], ignore_index=True))
        self.assertAlmostEqual(delta.loc[0, "delta_accuracy"], 1 / 3)
        with self.assertRaises(AssertionError):
            paired_deltas(pd.concat([a, b.assign(n_events=99)], ignore_index=True))


if __name__ == "__main__":
    unittest.main()
