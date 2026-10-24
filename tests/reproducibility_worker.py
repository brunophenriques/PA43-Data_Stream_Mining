"""Independent-process probe; timings are deliberately excluded."""
import hashlib
import json
import random
import sys

import numpy as np

from src.stream_analysis import FEATURES, GROUPS, SCORES, load_streams, model_factories, run_stream, summarize


def main():
    noise_seed = int(sys.argv[1])
    random.seed(noise_seed)
    np.random.seed(noise_seed)
    sys.setrecursionlimit(2000 if noise_seed == 1 else 3000)
    streams, _ = load_streams()
    results = {}
    for city, representations in streams.items():
        for representation, frame in representations.items():
            for name, factory in model_factories().items():
                history = run_stream(frame.head(720), FEATURES[representation], factory())
                score = summarize(history)
                logical = history[["actual", "predicted", "fallback", "outcome"]].to_csv(index=False)
                results["/".join([city, representation, name])] = {
                    "sha256": hashlib.sha256(logical.encode()).hexdigest(),
                    "metrics": {key: score[key] for key in SCORES + ["tn", "fp", "fn", "tp", "fallback_count"]},
                }
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
