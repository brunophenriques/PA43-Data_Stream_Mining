
import pandas as pd
import time
WARMUP_SIZE = 24 * 30  # 24 hours per day, 30 days per month, so the first 720 hours of data are used for warmup and not for evaluation

FEATURES_A = [
    "temperature_lag1",
    "humidity_lag1",
    "precipitation_lag1",
    "wind_lag1",
]

FEATURES_B = FEATURES_A + [
    "temperature_delta",
    "humidity_mean_6h",
    "precipitation_std_6h",
    "rain_events_24h",
]
'''
faro_a = pd.read_csv("data/processed/faro_representation_a.csv")

faro_a = faro_a.dropna(
    subset=FEATURES_A
).reset_index(drop=True)

print(faro_a[FEATURES_A + ["rain_event"]].head())
print(faro_a.shape)

'''
from river import naive_bayes
from river import metrics


'''
model = naive_bayes.GaussianNB()
accuracy = metrics.Accuracy()
macro_f1 = metrics.MacroF1()
precision = metrics.Precision()
recall = metrics.Recall()


for i, row in faro_a.iterrows():
    x = row[FEATURES_A].to_dict()
    y = int(row["rain_event"])

    if i < WARMUP_SIZE:
        model.learn_one(x, y)
        continue

    y_pred = model.predict_one(x)

    accuracy.update(y, y_pred)
    macro_f1.update(y, y_pred)
    precision.update(y, y_pred)
    recall.update(y, y_pred)

    model.learn_one(x, y)


print("Accuracy:", accuracy)
print("Macro-F1:", macro_f1)
print("Precision:", precision)
print("Recall:", recall)


'''


#now we do the functions 2x2x5, 2 cities, 2 representations, 5 models


def run_stream(df, features, model):
    df = df.dropna(subset=features).reset_index(drop=True)

    accuracy = metrics.Accuracy()
    macro_f1 = metrics.MacroF1()
    precision = metrics.Precision()
    recall = metrics.Recall()
    confusion_matrix = metrics.ConfusionMatrix()

    total_prediction_time = 0
    total_update_time = 0
    evaluated_events = 0

    for i, row in df.iterrows():

        x = row[features].to_dict()
        y = int(row["rain_event"])

        # Warm-up
        if i < WARMUP_SIZE:
            model.learn_one(x, y)
            continue

        # PREDICT
        start = time.perf_counter()
        y_pred = model.predict_one(x)
        total_prediction_time += time.perf_counter() - start

        # EVALUATE
        accuracy.update(y, y_pred)
        macro_f1.update(y, y_pred)
        precision.update(y, y_pred)
        recall.update(y, y_pred)
        confusion_matrix.update(y, y_pred)

        # LEARN
        start = time.perf_counter()
        model.learn_one(x, y)
        total_update_time += time.perf_counter() - start

        evaluated_events += 1

    # Get confusion matrix values AFTER the whole stream
    tn = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tp = confusion_matrix[1][1]

    return {
        "accuracy": accuracy.get(),
        "macro_f1": macro_f1.get(),
        "precision": precision.get(),
        "recall": recall.get(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "prediction_time_per_event": total_prediction_time / evaluated_events,
        "update_time_per_event": total_update_time / evaluated_events
    }

faro_a = pd.read_csv("data/processed/faro_representation_a.csv")
faro_b = pd.read_csv("data/processed/faro_representation_b.csv")

angra_a = pd.read_csv("data/processed/angra_representation_a.csv")
angra_b = pd.read_csv("data/processed/angra_representation_b.csv")

datasets = {
    "Faro": {
        "A": faro_a,
        "B": faro_b
    },

    "Angra": {
        "A": angra_a,
        "B": angra_b
    }
}

from river import linear_model
from river import tree
from river import ensemble
from river import forest

models = {
    "GaussianNB": lambda: naive_bayes.GaussianNB(),

    "LogisticRegression": lambda: linear_model.LogisticRegression(),

    "HoeffdingTree": lambda: tree.HoeffdingTreeClassifier(),

    "OnlineBagging": lambda: ensemble.BaggingClassifier(
        model=tree.HoeffdingTreeClassifier(),
        n_models=10,
        seed=42
    ),

    "AdaptiveRandomForest": lambda: forest.ARFClassifier(
        n_models=10,
        seed=42
    )
}


results = []

for city_name, representations in datasets.items():

    for representation_name, df in representations.items():

        if representation_name == "A":
            features = FEATURES_A
        else:
            features = FEATURES_B

        for model_name, model_factory in models.items():

            print(
                f"Running {city_name} | "
                f"Representation {representation_name} | "
                f"{model_name}"
            )

            model = model_factory()

            result = run_stream(
                df=df,
                features=features,
                model=model
            )

            result["city"] = city_name
            result["representation"] = representation_name
            result["model"] = model_name

            results.append(result)


results_df = pd.DataFrame(results)

results_df = results_df[
    [
        "city",
        "representation",
        "model",
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
        "prediction_time_per_event",
        "update_time_per_event"
    ]
]

results_df.to_csv(
    "results/metrics.csv",
    index=False
)

print("\nAll experiments finished.")
print(results_df)