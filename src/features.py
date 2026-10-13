import pandas as pd


def add_target(df):
    df = df.copy()

    df["rain_event"] = (
        df["precipitation"] >= 0.1
    ).astype(int)

    return df

def add_representation_a(df):
    df = df.copy()

    df["temperature_lag1"] = df["temperature_2m"].shift(1)
    df["humidity_lag1"] = df["relative_humidity_2m"].shift(1)
    df["precipitation_lag1"] = df["precipitation"].shift(1)
    df["wind_lag1"] = df["wind_speed_10m"].shift(1)

    return df


def add_representation_b(df):
    df = add_representation_a(df)

    # Change between the previous two temperature observations
    df["temperature_delta"] = (
        df["temperature_2m"].shift(1)
        - df["temperature_2m"].shift(2)
    )

    # Mean humidity over the previous 6 hours
    df["humidity_mean_6h"] = (
        df["relative_humidity_2m"]
        .shift(1)
        .rolling(6)
        .mean()
    )

    # Precipitation variability over the previous 6 hours
    df["precipitation_std_6h"] = (
        df["precipitation"]
        .shift(1)
        .rolling(6)
        .std()
    )

    # Number of rain events during the previous 24 hours
    df["rain_events_24h"] = (
        df["rain_event"]
        .shift(1)
        .rolling(24)
        .sum()
    )

    return df


def save_dataset(df, path):
    df.to_csv(path, index=False)