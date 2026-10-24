# PA43 — Data Stream Mining

Practical Assignment I compares **Faro (City A)** and **Angra do Heroísmo
(City B)** using hourly weather from **2021–2025**. The business context is an
outdoor excursion operator assessing rain-free operating periods. The assigned
target, **T1 — Rain Event**, is precipitation **≥ 0.1 mm** in the predicted hour.
The operational comparison concerns rain only; dry weather alone does not
establish complete excursion suitability.

Student names and group number: to be completed by the pair before submission.

## Project structure

| Path | Purpose |
|---|---|
| `data/raw/` | Original hourly weather snapshots |
| `data/processed/` | Optional aligned A/B feature exports |
| `notebooks/eda.ipynb` | Data quality, weather patterns and historical rain-free windows |
| `notebooks/exploratory_analysis.ipynb` | Warm-up, calendar features and rain-transition diagnostics |
| `notebooks/final_results.ipynb` | Required 20 A/B combinations, metrics, confusion matrices and costs |
| `src/features.py` | T1 and past-only A/B weather features |
| `src/stream_analysis.py` | Shared stream protocol, model settings, prediction store and summaries |
| `src/experiment.py` | Explicit experiment and table-refresh entry point |
| `src/prepare_data.py` | Optional A/B input exports, without training |
| `results/eda/` | Descriptive tables and figures |
| `results/final/` | Shared prediction histories, configuration, final tables and figures |
| `results/exploratory/` | Exploratory tables and figures using the shared histories |

## Run the project

The project uses **Python 3.13.7** and **River 0.26.1**. Package versions are
pinned in `requirements.txt` and recorded in the saved configuration.

Install dependencies in the project Python environment:

```sh
python -m pip install -r requirements.txt
```

Open the three notebooks in that environment and run all cells. They support
the repository root or `notebooks/` as the working directory. Outputs remain
under `results/`, independently of the working directory.

The notebooks load saved predictions by default. In `final_results.ipynb`,
`RUN_EXPERIMENT = False` preserves that behaviour; setting it to `True`
explicitly regenerates all 30 histories. The exploratory notebook never trains
models. Missing, incomplete or stale saved results raise an error instead of
starting training automatically.

From the repository root, refresh derived tables from saved histories:

```sh
python -m src.experiment
```

To explicitly regenerate the complete experiment:

```sh
python -m src.experiment --run
```

This creates **30 fresh combinations**: 20 required (two cities × A/B × five
classifiers), plus ten B + calendar combinations. Both analysis notebooks use
this single prediction store. Run the notebooks afterwards to regenerate the
figures and displayed outputs.

Optional feature exports use `python -m src.prepare_data`. Experiments construct
features directly from the raw snapshots, so processed exports are not required.
From `notebooks/`, use `python ../src/experiment.py` (optionally with `--run`)
or `python ../src/prepare_data.py`. Importing these modules does not train,
download or write files. Data retrieval is a separate optional operation.

## Experimental protocol

- **A:** previous-hour temperature, relative humidity, precipitation and wind.
- **B:** A plus temperature delta, six-hour humidity mean, six-hour precipitation
  standard deviation and the previous-24-hour rain-event count.
- **B + calendar:** B plus sine/cosine pairs for local hour and annual phase,
  using `Europe/Lisbon` and `Atlantic/Azores`, including leap years.
- Models: GaussianNB, online LogisticRegression, HoeffdingTree, Online Bagging
  and Adaptive Random Forest.
- All representations start on **2021-01-02 00:00 UTC**, after the same 24-hour
  feature-history interval. Weather features use only preceding observations;
  the prediction timestamp's calendar is known in advance.
- Each hour follows **predict → evaluate → learn**, in chronological order.
  An undefined prediction falls back to no rain (0), with the occurrence flagged.
  Model learning continues across all scoring boundaries.
- Final A/B scoring begins on **2021-01-31 00:00 UTC**, after 30 raw calendar
  days, and ends on **2025-12-31 23:00 UTC**: **43,104 hours per combination**.
  There are 696 eligible learning observations before final scoring begins.
- Warm-up sensitivity changes the scoring start to 0, 30, 90 or 180 days, or
  the end of calendar year 2021. These candidates score different historical
  intervals from the same continuous prediction histories.
- Calendar and rain-transition diagnostics use **2022–2025**, with **35,064
  hours per combination**. Transition categories use actual current and previous
  rain observations only for analysis after prediction.

Outputs include Accuracy, fixed-label binary Macro-F1, rain Precision and
Recall, class counts, confusion matrices and mean prediction/update times.
Aggregate undefined ratios use zero. Monthly display columns mask undefined
rain recall/precision and single-class Macro-F1; transition rates with no
denominator remain NaN. Precision and Macro-F1 are not reported within
constant-target transition groups. Timings measure model calls, excluding feature
construction, aggregation, disk writes and plotting, and vary with system load.

## Reproducibility

`SEED = 42` is centralized. Online Bagging and Adaptive Random Forest receive
River's `seed=42`. GaussianNB, HoeffdingTree and the zero-initialized logistic
regression are deterministic in these configurations. Every combination starts
with a fresh model; chronology and feature order are fixed, without shuffling.
Tree depth is explicitly fixed at 980.

`results/final/configuration.json` records effective model parameters, package
versions, the historical period and source/data/prediction fingerprints. Changes
to those inputs require an explicit new experiment. Moving or editing notebook
presentation does not change the model configuration or saved predictions.
