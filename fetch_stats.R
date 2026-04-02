# fetch_stats.R
# pulls NFL season stats using nflfastR, filters to relevant players only,
# and tags rookies for OROY/DROY predictions
#
# usage:
#   Rscript fetch_stats.R          # fetches 2025
#   Rscript fetch_stats.R 2024     # fetches a specific year

library(nflfastR)
library(dplyr)
library(tidyr)

args <- commandArgs(trailingOnly = TRUE)
year <- if (length(args) > 0) as.integer(args[1]) else 2025

cat(sprintf("fetching %d season stats...\n", year))

# ── offensive stats ───────────────────────────────────────────────────────────
stats <- load_player_stats(seasons = year) %>%
  filter(season_type == "REG")

if (nrow(stats) == 0) {
  cat(sprintf("no data found for %d, it may not be available yet\n", year))
  quit(status = 1)
}

cat(sprintf("  %d offensive weekly rows\n", nrow(stats)))

offense <- stats %>%
  group_by(player_display_name, team, position) %>%
  summarise(
    G        = n(),
    Pass_Cmp = sum(completions, na.rm = TRUE),
    Pass_Att = sum(attempts, na.rm = TRUE),
    Pass_Yds = sum(passing_yards, na.rm = TRUE),
    Pass_TD  = sum(passing_tds, na.rm = TRUE),
    Pass_Int = sum(passing_interceptions, na.rm = TRUE),
    Rush_Att = sum(carries, na.rm = TRUE),
    Rush_Yds = sum(rushing_yards, na.rm = TRUE),
    Rush_TD  = sum(rushing_tds, na.rm = TRUE),
    Rec      = sum(receptions, na.rm = TRUE),
    Rec_Tgt  = sum(targets, na.rm = TRUE),
    Rec_Yds  = sum(receiving_yards, na.rm = TRUE),
    Rec_TD   = sum(receiving_tds, na.rm = TRUE),
    Def_Solo   = sum(def_tackles_solo, na.rm = TRUE),
    Def_Ast    = sum(def_tackle_assists, na.rm = TRUE),
    Def_Comb   = sum(def_tackles_solo + def_tackle_assists, na.rm = TRUE),
    Def_TFL    = sum(def_tackles_for_loss, na.rm = TRUE),
    Def_QBHits = sum(def_qb_hits, na.rm = TRUE),
    Sk         = sum(def_sacks, na.rm = TRUE),
    Int        = sum(def_interceptions, na.rm = TRUE),
    PD         = sum(def_pass_defended, na.rm = TRUE),
    FF         = sum(def_fumbles_forced, na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  rename(
    Player   = player_display_name,
    Team     = team,
    Position = position
  ) %>%
  mutate(Year = year)

# ── filter to relevant players only ──────────────────────────────────────────
# keep players who actually contributed meaningfully — avoids practice squad
# guys, long snappers, punters, etc from cluttering the candidate pool
relevant <- offense %>%
  filter(
    G >= 6,  # played at least 6 games
    (
      Pass_Att  >= 50  |   # QBs with real playing time
      Rush_Att  >= 30  |   # RBs/scrimmage backs
      Rec_Tgt   >= 20  |   # receivers with real target share
      Def_Comb  >= 20  |   # defenders with meaningful tackles
      Sk        >= 2   |   # pass rushers
      Int       >= 1   |   # DBs with picks
      PD        >= 3       # corners with pass breakups
    ),
    # skip positions that never win these awards
    !Position %in% c("K", "P", "LS", "OT", "OG", "OL", "C", "T", "G",
                     "LT", "RT", "LG", "RG", "FB")
  )

cat(sprintf("  filtered to %d relevant players\n", nrow(relevant)))

# ── tag rookies ───────────────────────────────────────────────────────────────
cat("  loading roster data to tag rookies...\n")
rosters <- load_rosters(seasons = year) %>%
  filter(!is.na(full_name)) %>%
  select(full_name, years_exp) %>%
  rename(Player = full_name) %>%
  # some players appear multiple times (traded etc), keep min exp
  group_by(Player) %>%
  summarise(years_exp = min(years_exp, na.rm = TRUE), .groups = "drop") %>%
  mutate(Is_Rookie = as.integer(years_exp == 0)) %>%
  select(Player, Is_Rookie)

relevant <- relevant %>%
  left_join(rosters, by = "Player") %>%
  mutate(Is_Rookie = replace_na(Is_Rookie, 0))

rookies_found <- sum(relevant$Is_Rookie)
cat(sprintf("  tagged %d rookies\n", rookies_found))

# ── save ──────────────────────────────────────────────────────────────────────
relevant <- relevant %>% arrange(Player)
out_path <- sprintf("candidates_%d.csv", year)
write.csv(relevant, out_path, row.names = FALSE)

cat(sprintf("\nsaved %d players to %s\n", nrow(relevant), out_path))
cat(sprintf("columns: %s\n", paste(colnames(relevant), collapse = ", ")))
cat("\nsample — top QBs by pass yards:\n")
relevant %>%
  filter(Position == "QB") %>%
  arrange(desc(Pass_Yds)) %>%
  select(Player, Team, G, Pass_Yds, Pass_TD, Pass_Int, Rush_Yds, Is_Rookie) %>%
  head(5) %>%
  print()
cat("\nsample — top defenders by sacks:\n")
relevant %>%
  filter(Sk > 0) %>%
  arrange(desc(Sk)) %>%
  select(Player, Team, Position, G, Sk, Def_Comb, Def_TFL, Is_Rookie) %>%
  head(5) %>%
  print()