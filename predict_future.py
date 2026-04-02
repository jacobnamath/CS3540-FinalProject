# predict_future.py
# predicts NFL award winners for a current/future season
# trains on historical CSVs, predicts against candidates_{year}.csv from fetch_stats.R
# Before you use this you must use fetch_stats.R, should gather information on years after 2024
# You will have to specify year, may not be available yet

#
# usage:
#   python predict_future.py            # predicts 2025
#   python predict_future.py --year 2024

import argparse
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

script_dir = os.path.dirname(os.path.abspath(__file__))

DEFENSE_POSITIONS = {"DE", "DT", "LB", "CB", "S", "DB", "OLB", "ILB",
                     "MLB", "NT", "DL", "FS", "SS", "EDGE"}

def is_defense(pos):
    if not isinstance(pos, str):
        return False
    p = pos.upper()
    return any(p == d or p.startswith(d) or p.endswith(d) for d in DEFENSE_POSITIONS)

# rename historical columns to match candidates CSV naming from fetch_stats.R
COLUMN_ALIASES = {
    # MVP / general
    "Games":            "G",
    "Games_Started":    "GS",
    "Pass_Yards":       "Pass_Yds",
    "Pass_Attempts":    "Pass_Att",
    "Pass_Completions": "Pass_Cmp",
    "Pass_Touchdowns":  "Pass_TD",
    "Pass_Interceptions": "Pass_Int",
    "Rush_Attempts":    "Rush_Att",
    "Rush_Yards":       "Rush_Yds",
    "Rush_Touchdowns":  "Rush_TD",
    "Rec_Receptions":   "Rec",
    "Rec_Yards":        "Rec_Yds",
    "Rec_Touchdowns":   "Rec_TD",
    # DPOY
    "Solo":             "Def_Solo",
    "Ast":              "Def_Ast",
    "Comb":             "Def_Comb",
    "TFL":              "Def_TFL",
    "QBHits":           "Def_QBHits",
    "Sk":               "Sk",
    "FF":               "FF",
    "Int":              "Int",
    "PD":               "PD",
    "Def_Int":          "Int",
    "Def_Sk":           "Sk",
}

def normalize_cols(df):
    return df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

AWARD_FILTERS = {
    "MVP": lambda df: df[
        (df["Position"] == "QB") &
        (df["Pass_Att"] >= 200) &
        (df["G"] >= 10)
    ],
    "OPOY": lambda df: df[
        df["Position"].isin(["QB", "RB", "WR", "TE"]) &
        (df["G"] >= 10) &
        (
            (df["Pass_Yds"] >= 2000) |
            (df["Rush_Yds"] >= 600) |
            (df["Rec_Yds"] >= 600)
        )
    ],
    "DPOY": lambda df: df[
        df["Position"].apply(is_defense) &
        (df["G"] >= 10) &
        (
            (df["Sk"] >= 5) |
            (df["Def_Comb"] >= 40) |
            (df["Int"] >= 3) |
            (df["Def_TFL"] >= 8)
        )
    ],
    "OROY": lambda df: df[
        (df["Is_Rookie"] == 1) &
        df["Position"].isin(["QB", "RB", "WR", "TE"]) &
        (df["G"] >= 6) &
        (
            (df["Pass_Yds"] >= 500) |
            (df["Rush_Yds"] >= 300) |
            (df["Rec_Yds"] >= 300)
        )
    ],
    "DROY": lambda df: df[
        (df["Is_Rookie"] == 1) &
        df["Position"].apply(is_defense) &
        (df["G"] >= 6) &
        (
            (df["Sk"] >= 2) |
            (df["Def_Comb"] >= 20) |
            (df["Int"] >= 1) |
            (df["Def_TFL"] >= 3)
        )
    ],
}


def load_historical():
    files = {
        "MVP":  "mvp_df_final_clean.csv",
        "OPOY": "opoy_df_final_clean.csv",
        "DPOY": "dpoy_df_final_clean.csv",
        "OROY": "oroy_df_final_clean.csv",
        "DROY": "droy_df_final_clean.csv",
    }
    datasets = {}
    for award, fname in files.items():
        path = os.path.join(script_dir, fname)
        if not os.path.exists(path):
            print(f"couldn't find {fname}, make sure it's in the same folder")
            sys.exit(1)
        df = pd.read_csv(path)
        datasets[award] = normalize_cols(df)
    return datasets


def load_candidates(year):
    path = os.path.join(script_dir, f"candidates_{year}.csv")
    if not os.path.exists(path):
        print(f"couldn't find candidates_{year}.csv")
        print(f"run: Rscript fetch_stats.R {year}")
        sys.exit(1)
    return pd.read_csv(path)


def add_winner_col(df):
    df = df.copy()
    df["_votes"] = df["Votes"].fillna(-1)
    max_votes = df.groupby("Year")["_votes"].transform("max")
    df["Winner"] = (df["_votes"] == max_votes).astype(int)
    df.drop(columns=["_votes"], inplace=True)
    return df


def get_shared_cols(hist_df, candidates_df):
    skip = {"Year", "Rank", "Votes", "Winner", "Is_Rookie", "GS"}
    hist_cols = set(hist_df.select_dtypes(include=[np.number]).columns) - skip
    cand_cols = set(candidates_df.select_dtypes(include=[np.number]).columns) - skip
    return sorted(hist_cols & cand_cols)


def predict_award(hist_df, candidates_df, award):
    filter_fn = AWARD_FILTERS.get(award)
    filtered = filter_fn(candidates_df) if filter_fn else candidates_df

    if len(filtered) == 0:
        print(f"  no candidates passed the filter for {award}")
        return None

    hist_df = add_winner_col(hist_df)
    shared_cols = get_shared_cols(hist_df, filtered)

    if len(shared_cols) == 0:
        print(f"  no shared columns for {award}")
        return None

    X_train = hist_df[shared_cols].fillna(0).values
    y_train = hist_df["Winner"].values
    X_test  = filtered[shared_cols].fillna(0).values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=3540)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    results = filtered[["Player", "Team", "Position"]].copy().reset_index(drop=True)
    results["Win_Probability_%"] = (probs * 100).round(1)
    results = results.sort_values("Win_Probability_%", ascending=False).reset_index(drop=True)
    results.index += 1

    return results


def prob_bar(prob, width=30):
    filled = int(round(prob / 100 * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"]  {prob:.1f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()
    year = args.year

    print(f"\nNFL Award Predictions — {year} season\n")

    historical = load_historical()
    candidates = load_candidates(year)
    print(f"loaded {len(candidates)} candidates from candidates_{year}.csv\n")

    for award, hist_df in historical.items():
        print(f"--- {award} {year} ---")
        results = predict_award(hist_df, candidates, award)
        if results is None:
            continue
        top3 = results.head(3)
        for _, row in top3.iterrows():
            print(f"  {row.name}. {row['Player']} ({row['Position']}, {row['Team']})")
            print(f"     {prob_bar(row['Win_Probability_%'])}")
        print()

    print("done.")


if __name__ == "__main__":
    main()