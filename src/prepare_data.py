"""Export validated, aligned A/B model inputs without training or downloading."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stream_analysis import ROOT, CITIES, load_streams


def main():
    streams, checks = load_streams()
    output = ROOT / "data" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    for city, slug in CITIES.items():
        for rep in ("A", "B"):
            streams[city][rep].to_csv(output / f"{slug}_representation_{rep.lower()}.csv", index=False)
    print(checks.to_string(index=False))
    print("Saved four aligned A/B input files. No model was trained.")


if __name__ == "__main__":
    main()
