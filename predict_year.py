# predict_year.py
# Enter a year and get MVP/OPOY/DPOY/OROY/DROY predictions using logistic regression
# CSVs need to be in the same folder as this script

import warnings
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))


def load_data():
    datasets = {}
    files = {
        "MVP":  "mvp_df_final_clean.csv",
        "OPOY": "opoy_df_final_clean.csv",
        "DPOY": "dpoy_df_final_clean.csv",
        "OROY": "oroy_df_final_clean.csv",
        "DROY": "droy_df_final_clean.csv",
    }
    for award, fname in files.items():
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f"couldn't find {fname}, make sure it's in the same folder")
            sys.exit(1)
        datasets[award] = pd.read_csv(path)
    return datasets


def add_winner_col(df):
    # label the player with the most votes each year as the winner
    df = df.copy()
    df["_votes"] = df["Votes"].fillna(-1)
    max_votes = df.groupby("Year")["_votes"].transform("max")
    df["Winner"] = (df["_votes"] == max_votes).astype(int)
    df.drop(columns=["_votes"], inplace=True)
    return df


def get_stat_cols(df):
    # drop non-stat columns before modeling
    skip = {"Year", "Rank", "Votes", "Winner"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]


def predict_year(df, year, award):
    df = add_winner_col(df)

    years_available = sorted(df["Year"].unique())
    if year not in years_available:
        print(f"{award}: year {year} not in data (range is {years_available[0]}-{years_available[-1]})")
        return None

    stat_cols = get_stat_cols(df)

    # train on everything except the target year
    train = df[df["Year"] != year].copy()
    test  = df[df["Year"] == year].copy()

    X_train = train[stat_cols].fillna(0).values
    y_train = train["Winner"].values
    X_test  = test[stat_cols].fillna(0).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=3540)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    results = test[["Player", "Team", "Position", "Votes"]].copy() if "Team" in test.columns and "Position" in test.columns else test[["Player", "Votes"]].copy()
    results["Win_Probability_%"] = (probs * 100).round(1)
    results["Predicted_Winner"] = (probs == probs.max()).astype(int)
    results = results.sort_values("Win_Probability_%", ascending=False).reset_index(drop=True)
    results.index += 1

    return results


def main():
    datasets = load_data()

    all_years = set()
    for df in datasets.values():
        all_years.update(df["Year"].unique())
    all_years = sorted(all_years)

    print(f"\nNFL Award Predictions (MVP, OPOY, DPOY, OROY, DROY) — data: {all_years[0]}-{all_years[-1]}\n")

    while True:
        raw = input(f"Enter a year ({all_years[0]}-{all_years[-1]}): ").strip()
        if not raw.isdigit():
            print("please enter a valid year")
            continue
        year = int(raw)
        if year not in all_years:
            print(f"{year} isn't in the dataset, try between {all_years[0]} and {all_years[-1]}")
            continue
        break

    print(f"\nRunning predictions for {year}...\n")

    for award, df in datasets.items():
        print(f"--- {award} {year} ---")

        results = predict_year(df, year, award)
        if results is None:
            continue

        # only show top 3
        top3 = results.head(3)
        cols = [c for c in ["Player", "Team", "Position", "Votes", "Win_Probability_%", "Predicted_Winner"] if c in top3.columns]
        print(top3[cols].to_string())

        winner = results[results["Predicted_Winner"] == 1]
        if not winner.empty:
            name = winner.iloc[0]["Player"]
            prob = winner.iloc[0]["Win_Probability_%"]
            print(f"\npredicted {award} winner: {name} ({prob}%)\n")


if __name__ == "__main__":
    main()