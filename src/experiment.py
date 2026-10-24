"""Explicit experiment entry point; loading this module never trains models."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stream_analysis import generate_predictions, load_predictions, refresh_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Train 20 required A/B combinations and reproduce 10 calendar combinations")
    args = parser.parse_args()
    if args.run:
        generate_predictions()
    predictions = load_predictions()
    tables = refresh_results(predictions)
    print(tables["verification"].to_string(index=False))


if __name__ == "__main__":
    main()
