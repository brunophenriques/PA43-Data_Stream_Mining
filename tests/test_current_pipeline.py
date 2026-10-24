"""Current configuration, process isolation and safe cached execution."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4

from src.stream_analysis import MAX_DEPTH, ROOT, SEED, configuration, load_predictions, model_factories


class CurrentPipelineChecks(unittest.TestCase):
    def test_seed_and_fresh_models(self):
        factories = model_factories()
        self.assertEqual(SEED, 42)
        for name, factory in factories.items():
            self.assertIsNot(factory(), factory(), name)
            params = factory()._get_params()
            if name in ("OnlineBagging", "AdaptiveRandomForest"):
                self.assertEqual(params["seed"], 42)
            else:
                self.assertNotIn("seed", params)
            if name in ("HoeffdingTree", "AdaptiveRandomForest"):
                self.assertEqual(params["max_depth"], MAX_DEPTH)
        manifest = configuration()
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(len(manifest["sha256"]), 6)

    def test_separate_processes_reproduce_all_30_combinations(self):
        probes = []
        for noise_seed in (1, 99):
            env = dict(os.environ, PYTHONHASHSEED=str(noise_seed))
            completed = subprocess.run(
                [sys.executable, "-m", "tests.reproducibility_worker", str(noise_seed)],
                cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=180,
            )
            probes.append(json.loads(completed.stdout))
        self.assertEqual(len(probes[0]), 30)
        self.assertEqual(probes[0], probes[1])

    def test_missing_cache_never_trains(self):
        directory = ROOT / ".venv" / f"absent-cache-{uuid4()}"
        with self.assertRaises(FileNotFoundError):
            load_predictions(directory)
        self.assertFalse(directory.exists())

    def test_entrypoints_are_import_safe(self):
        source = "import src.experiment, src.prepare_data, src.download_data; print('safe')"
        result = subprocess.run([sys.executable, "-c", source], cwd=ROOT,
                                check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.stdout.strip(), "safe")


if __name__ == "__main__":
    unittest.main()
