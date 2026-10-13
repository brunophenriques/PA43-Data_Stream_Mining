import requests
import pandas as pd
import time

URL = "https://archive-api.open-meteo.com/v1/archive"

cities = {
    "faro": (37.02, -7.93),
    "angra": (38.65, -27.22)
}

for city, (lat, lon) in cities.items():
    print(f"\n=== Starting {city.upper()} ===")

    city_start = time.perf_counter()
    dfs = []

    for year in range(2021, 2026):
        print(f"\n[{city}] Downloading {year}...")

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
            "timezone": "GMT"
        }

        year_start = time.perf_counter()

        for attempt in range(1, 6):
            try:
                print(f"  Attempt {attempt}/5...")

                response = requests.get(
                    URL,
                    params=params,
                    timeout=120
                )

                if response.status_code == 200:
                    break

                print(f"  HTTP {response.status_code} — retrying...")

            except requests.RequestException as e:
                print(f"  Request error: {e}")

            time.sleep(5)

        else:
            raise RuntimeError(f"Failed to download {city} {year}")

        df = pd.DataFrame(response.json()["hourly"])
        dfs.append(df)

        elapsed = time.perf_counter() - year_start

        print(f"  Done {year}")
        print(f"  Rows: {len(df):,}")
        print(f"  Time: {elapsed:.2f}s")

    city_df = pd.concat(dfs, ignore_index=True)

    output_file = f"{city}_2021_2025.csv"
    city_df.to_csv(output_file, index=False)

    city_elapsed = time.perf_counter() - city_start

    print(f"\n=== Finished {city.upper()} ===")
    print(f"Total rows: {len(city_df):,}")
    print(f"Saved to: {output_file}")
    print(f"Total time: {city_elapsed:.2f}s")