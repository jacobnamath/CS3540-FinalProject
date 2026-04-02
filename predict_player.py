# predict_player.py
# Enter a year and player name, get their predicted win probability for all 5 NFL awards
# uses logistic regression trained on all other years (leave-one-year-out)
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
    # who got the most votes each year = winner
    df = df.copy()
    df["_votes"] = df["Votes"].fillna(-1)
    max_votes = df.groupby("Year")["_votes"].transform("max")
    df["Winner"] = (df["_votes"] == max_votes).astype(int)
    df.drop(columns=["_votes"], inplace=True)
    return df


def get_stat_cols(df):
    skip = {"Year", "Rank", "Votes", "Winner"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]


def match_player(name_input, candidates):
    # exact match only (case-insensitive)
    needle = name_input.strip().lower()
    return [n for n in candidates if n.lower() == needle]


def run_model(df, year, player_name, award):
    df = add_winner_col(df)
    stat_cols = get_stat_cols(df)

    train = df[df["Year"] != year].copy()
    test  = df[df["Year"] == year].copy()

    player_row = test[test["Player"].str.lower() == player_name.lower()]
    if player_row.empty:
        return None

    X_train = train[stat_cols].fillna(0).values
    y_train = train["Winner"].values
    X_test  = test[stat_cols].fillna(0).values

    if y_train.sum() == 0:
        return None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=3540)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    keep_cols = [c for c in ["Player", "Team", "Position", "Votes"] if c in test.columns]
    results = test[keep_cols].copy().reset_index(drop=True)
    results["Win_Probability_%"] = (probs * 100).round(1)
    results = results.sort_values("Win_Probability_%", ascending=False).reset_index(drop=True)
    results.index += 1

    actual_winner = test.loc[test["Winner"] == 1, "Player"].values
    actual_winner = actual_winner[0] if len(actual_winner) else "Unknown"

    player_prob = results.loc[results["Player"].str.lower() == player_name.lower(), "Win_Probability_%"].values[0]
    player_rank = results.index[results["Player"].str.lower() == player_name.lower()].tolist()[0]

    player_stats = player_row.iloc[0].to_dict()

    return {
        "award": award,
        "player": player_row.iloc[0]["Player"],
        "team": player_row.iloc[0].get("Team", "N/A"),
        "position": player_row.iloc[0].get("Position", "N/A"),
        "votes": player_row.iloc[0]["Votes"],
        "win_prob": player_prob,
        "rank": player_rank,
        "field_size": len(results),
        "actual_winner": actual_winner,
        "candidates": results,
        "stats": {k: v for k, v in player_stats.items()
                  if k not in {"Year", "Rank", "Player", "Team", "Position", "Votes", "Winner"}},
    }


def prob_bar(prob, width=30):
    filled = int(round(prob / 100 * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"]  {prob:.1f}%"


# columns that are noise / derived / not useful to display
SKIP_STAT_KEYWORDS = [
    "unnamed", "pfr_", "scrim_", "y_per", "any_a", "ny_a", "y_a", "y_g",
    "a_g", "r_g", "y_r", "y_tgt", "y_tch", "succ", "lng", "ctch", "qbrec",
    "rank", "age", "tm", "pos", "g_", "_g", "rate", "qbr", "^g$", "^gs$"
]

SKIP_EXACT = {"g", "gs", "games", "games_started"}

def pick_key_stats(stats, n=6):
    # filter to raw counting stats, skip derived/ratio columns
    filtered = {}
    for col, val in stats.items():
        if not pd.notna(val) or val == 0:
            continue
        col_lower = col.lower()
        if col_lower in SKIP_EXACT or any(kw in col_lower for kw in SKIP_STAT_KEYWORDS):
            continue
        filtered[col] = val
    # return top n by value (biggest numbers are usually the most meaningful)
    sorted_stats = sorted(filtered.items(), key=lambda x: abs(float(x[1])) if str(x[1]).replace('.','').lstrip('-').isdigit() else 0, reverse=True)
    return sorted_stats[:n]


def show_result(res, year):
    info = res['player']
    if res['position'] != 'N/A' or res['team'] != 'N/A':
        info += f" ({res['position']}, {res['team']})"
    print(f"\n{res['award']} {year} — {info}")
    print(f"Win Probability  {prob_bar(res['win_prob'])}")
    print(f"Rank: #{res['rank']} of {res['field_size']}  |  Actual winner: {res['actual_winner']}")
    key_stats = pick_key_stats(res['stats'])
    if key_stats:
        # clean up column names for display: "Pass_Yards" -> "Pass Yards"
        stats_str = "  |  ".join(f"{col.replace('_', ' ').title()}: {val}" for col, val in key_stats)
        print(f"Stats: {stats_str}")


def main():
    datasets = load_data()

    all_years = set()
    for df in datasets.values():
        all_years.update(df["Year"].unique())
    all_years = sorted(all_years)

    print(f"\nNFL Player Award Predictor (MVP, OPOY, DPOY, OROY, DROY) — data: {all_years[0]}-{all_years[-1]}\n")

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

    # get all players who received votes that year across all awards
    year_players = set()
    for df in datasets.values():
        year_players.update(df[df["Year"] == year]["Player"].dropna().unique())

    while True:
        raw_name = input("\nEnter player full name (Must have received votes for at least one award): ").strip()
        if not raw_name:
            print("please enter a name")
            continue

        matches = match_player(raw_name, list(year_players))

        if not matches:
            print(f"no player named '{raw_name}' found in {year}, check the spelling and try again, (Make sure they received votes for at least one award)")
            continue

        player_name = matches[0]
        break

    print(f"\nrunning predictions for {player_name} ({year})...")

    found_any = False
    for award, df in datasets.items():
        result = run_model(df, year, player_name, award)
        if result is None:
            continue
        found_any = True
        show_result(result, year)

    if not found_any:
        print(f"\n{player_name} wasn't found in any award for {year}")

    print("\ndone.")


if __name__ == "__main__":
    main()