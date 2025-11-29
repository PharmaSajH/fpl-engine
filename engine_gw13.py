import requests
import pandas as pd
import numpy as np

# -------- CONFIGURATION --------
NUM_GWS = 3          # How many GWs to sum (e.g. 3 for GW13-15)
TEAM_ID = 10856343   # Your FPL team id
FREE_TRANSFERS = 1   # Change to 2 if you roll a FT!
HIT_COST = 4         # Points lost per extra transfer
MAX_ADDITIONAL_TRANSFERS = 2   # At most –8 points total hits (2 × 4 points)


OUTPUT_ALL_CSV = None
OUTPUT_MYTEAM_CSV = None
OUTPUT_TRANSFERS_CSV = None
OUTPUT_DOUBLE_CSV = None
OUTPUT_MILP_PLAN_CSV = None

# How much to penalise volatile players in robust EP
ROBUST_ALPHA = 0.2  # try 0.3–0.7; higher = more risk averse

# -------------------------------

def fetch_bootstrap():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def fetch_fixtures():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()

def fetch_manager_picks_and_bank(team_id, gw):
    url = f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Could not fetch picks for GW{gw} (status {r.status_code}).")
        return None, 0, 0
    data = r.json()
    picks = pd.DataFrame(data.get("picks", []))
    entry_hist = data.get("entry_history", {}) or {}
    bank_tenths = entry_hist.get("bank", 0)
    value_tenths = entry_hist.get("value", 0)
    return picks, bank_tenths, value_tenths

def build_players_df(data):
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])[["id", "name", "short_name"]]
    positions = pd.DataFrame(data["element_types"])[["id", "singular_name_short"]]

    players = players.merge(
        teams, left_on="team", right_on="id", how="left", suffixes=("", "_team")
    )
    players = players.rename(columns={"name": "team_name", "short_name": "team_short"})

    players = players.merge(
        positions, left_on="element_type", right_on="id", how="left", suffixes=("", "_pos")
    )
    players = players.rename(columns={"singular_name_short": "position"})

    keep_cols = [
        "id", "web_name", "team", "team_name", "team_short", "position", "now_cost",
        "total_points", "minutes", "points_per_game", "form", "goals_scored",
        "assists", "clean_sheets", "chance_of_playing_next_round",
        "selected_by_percent", "status",
    ]
    players = players[keep_cols]

    players["price"] = players["now_cost"] / 10.0
    players["points_per_game"] = pd.to_numeric(players["points_per_game"], errors="coerce").fillna(0.0)
    players["form"] = pd.to_numeric(players["form"], errors="coerce").fillna(0.0)
    players["selected_by_percent"] = pd.to_numeric(players["selected_by_percent"], errors="coerce").fillna(0.0)

    minutes = players["minutes"].replace(0, np.nan)
    players["g_per90"] = players["goals_scored"] / minutes * 90
    players["a_per90"] = players["assists"] / minutes * 90

    players["g_per90"] = players["g_per90"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    players["a_per90"] = players["a_per90"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Cap per-90 rates to realistic ranges
    players["g_per90"] = players["g_per90"].clip(0, 1.2)
    players["a_per90"] = players["a_per90"].clip(0, 1.2)

    players["ga_per90"] = players["g_per90"] + players["a_per90"]

    # --- VOLATILITY SCORE ---
    # Simple proxy: attacking involvement + instability in recent form
    vol = (players["ga_per90"].fillna(0.0)
           + (players["form"] - players["points_per_game"]).abs())
    # Clip to avoid weird outliers, roughly 0–5
    players["volatility_score"] = vol.clip(0.0, 5.0)
    # ------------------------

    return players

def compute_team_goal_model(fixtures, window_gws=8, min_matches=4):
    """
    Build simple Poisson-style attack/defence strengths from past results.

    - Uses only finished fixtures with known scores.
    - Looks back over the last `window_gws` gameweeks.
    - For each team, computes goals for/against per match.
    - Normalises by league average to get attack/defence multipliers.
    """
    fx = pd.DataFrame(fixtures).copy()

    # Keep finished fixtures with scores
    mask = fx["finished"] & fx["team_h_score"].notna() & fx["team_a_score"].notna()
    hist = fx[mask].copy()
    if hist.empty:
        teams = pd.unique(pd.concat([fx["team_h"], fx["team_a"]], ignore_index=True))
        return pd.DataFrame({"team_id": teams, "att_strength": 1.0, "def_strength": 1.0})

    # Restrict to last N gameweeks
    if "event" in hist.columns:
        max_gw = hist["event"].max()
        hist = hist[hist["event"] >= max_gw - window_gws + 1]

    rows = []
    for _, r in hist.iterrows():
        rows.append({
            "team_id": r["team_h"],
            "goals_for": r["team_h_score"],
            "goals_against": r["team_a_score"],
        })
        rows.append({
            "team_id": r["team_a"],
            "goals_for": r["team_a_score"],
            "goals_against": r["team_h_score"],
        })
    tall = pd.DataFrame(rows)

    grouped = tall.groupby("team_id").agg(
        matches=("goals_for", "count"),
        gf=("goals_for", "sum"),
        ga=("goals_against", "sum"),
    ).reset_index()

    # Require a minimum number of matches for stability
    grouped = grouped[grouped["matches"] >= min_matches]
    if grouped.empty:
        teams = pd.unique(pd.concat([fx["team_h"], fx["team_a"]], ignore_index=True))
        return pd.DataFrame({"team_id": teams, "att_strength": 1.0, "def_strength": 1.0})

    grouped["gf_per_match"] = grouped["gf"] / grouped["matches"]
    grouped["ga_per_match"] = grouped["ga"] / grouped["matches"]

    mean_gf = grouped["gf_per_match"].mean()
    mean_ga = grouped["ga_per_match"].mean()

    # Attack: higher is better; defence: lower is better
    grouped["att_strength"] = grouped["gf_per_match"] / mean_gf
    grouped["def_strength"] = grouped["ga_per_match"] / mean_ga

    # Clip to a sensible range to avoid crazy extremes
    grouped["att_strength"] = grouped["att_strength"].clip(0.5, 1.5)
    grouped["def_strength"] = grouped["def_strength"].clip(0.5, 1.5)

    return grouped[["team_id", "att_strength", "def_strength"]]

def build_fixture_difficulty_df(fixtures, gw, team_model, base_lambda=1.4):
    """
    For a given GW, build fixture-level Poisson features for each team:

    - lambda_for, lambda_against : expected goals for/against
    - scoring_factor             : relative attacking intensity (λ_for / base_lambda)
    - cs_prob                    : clean sheet probability ≈ P(0 goals against)
    """
    fx = pd.DataFrame(fixtures)
    gw_fx = fx[fx["event"] == gw].copy()
    if gw_fx.empty:
        return pd.DataFrame()

    # Merge team strengths for home and away
    tm_h = team_model.rename(columns={"team_id": "team_h"})
    gw_fx = gw_fx.merge(tm_h, on="team_h", how="left")
    tm_a = team_model.rename(columns={"team_id": "team_a", "att_strength": "att_strength_a", "def_strength": "def_strength_a"})
    gw_fx = gw_fx.merge(tm_a, on="team_a", how="left")

    # Fill missing strengths with neutral 1.0
    for col in ["att_strength", "def_strength", "att_strength_a", "def_strength_a"]:
        if col not in gw_fx.columns:
            gw_fx[col] = 1.0
    gw_fx[["att_strength", "def_strength", "att_strength_a", "def_strength_a"]] = \
        gw_fx[["att_strength", "def_strength", "att_strength_a", "def_strength_a"]].fillna(1.0)

    rows = []
    for _, r in gw_fx.iterrows():
        att_h = r["att_strength"]
        def_h = r["def_strength"]
        att_a = r["att_strength_a"]
        def_a = r["def_strength_a"]

        # Standard Poisson-style link: λ = base * attack_strength / opponent_def_strength
        lam_home = base_lambda * att_h / max(def_a, 1e-3)
        lam_away = base_lambda * att_a / max(def_h, 1e-3)

        # Home side record
        rows.append({
            "team_id": r["team_h"],
            "opponent_id": r["team_a"],
            "home_away": "H",
            "lambda_for": lam_home,
            "lambda_against": lam_away,
            "scoring_factor": lam_home / base_lambda,
            "cs_prob": float(np.exp(-lam_away)),  # P(0 goals against) under Poisson
        })
        # Away side record
        rows.append({
            "team_id": r["team_a"],
            "opponent_id": r["team_h"],
            "home_away": "A",
            "lambda_for": lam_away,
            "lambda_against": lam_home,
            "scoring_factor": lam_away / base_lambda,
            "cs_prob": float(np.exp(-lam_home)),
        })

    df = pd.DataFrame(rows)
    # One row per team in that GW
    df = df.groupby("team_id", as_index=False).first()
    return df

def difficulty_to_multiplier(difficulty):
    if pd.isna(difficulty): return 1.0
    d = float(difficulty)
    if d <= 1: return 1.25
    elif d == 2: return 1.12
    elif d == 3: return 1.00
    elif d == 4: return 0.90
    else: return 0.80

def minutes_factor(row):
    status = row["status"]
    if status in ("i", "s"): return 0.0
    cop = row.get("chance_of_playing_next_round", None)
    if pd.notna(cop): return float(cop) / 100.0
    minutes = row.get("minutes", 0)
    if minutes <= 0: return 0.0
    if minutes >= 900: return 0.9
    elif minutes >= 450: return 0.7
    elif minutes >= 180: return 0.5
    else: return 0.3

def base_ep_from_poisson(row):
    """
    Expected FPL points for a full match before minutes multiplier.
    Uses:
    - g_per90, a_per90 : capped attacking contribution
    - scoring_factor   : capped fixture attacking intensity
    - cs_prob          : clean sheet probability (for DEF/GKP)
    """
    pos = row["position"]

    # Safely get and cap per-90 rates again (defensive programming)
    g90 = row.get("g_per90", 0.0) or 0.0
    a90 = row.get("a_per90", 0.0) or 0.0
    g90 = max(0.0, min(g90, 1.2))
    a90 = max(0.0, min(a90, 1.2))

    sf = row.get("scoring_factor", 1.0) or 1.0
    # Cap fixture intensity so no one gets 3x attacking boost
    sf = max(0.6, min(sf, 1.4))

    cs_p = row.get("cs_prob", 0.0) or 0.0
    cs_p = max(0.0, min(cs_p, 1.0))

    # Goal points by position
    if pos == "FWD":
        goal_pts = 4
    elif pos == "MID":
        goal_pts = 5
    elif pos == "DEF":
        goal_pts = 6
    elif pos in ("GKP", "GK"):
        goal_pts = 6
    else:
        goal_pts = 5

    base_app = 2.0  # 60+ mins

    attack_ep = sf * (g90 * goal_pts + a90 * 3.0)

    cs_ep = 0.0
    if pos in ("DEF", "GKP", "GK"):
        cs_ep = 4.0 * cs_p

    ep = base_app + attack_ep + cs_ep

    # 🔒 Final safety: no player gets more than ~10 points per full match
    ep = max(0.0, min(ep, 10.0))

    return ep

def predict_gw_points(players, fixture_df):
    """
    Merge player data with fixture Poisson model and compute per-GW EP.
    Also compute a robust EP that penalises volatile players.
    """
    preds = players.merge(
        fixture_df, left_on="team", right_on="team_id", how="left",
    )

    preds["scoring_factor"] = preds["scoring_factor"].fillna(1.0).clip(0.6, 1.4)
    preds["cs_prob"] = preds["cs_prob"].fillna(0.0).clip(0.0, 1.0)

    # Ensure volatility score exists
    preds["volatility_score"] = preds.get("volatility_score", 0.0).fillna(0.0)

    # Minutes factor = probability and share of playing time
    preds["minutes_multiplier"] = preds.apply(minutes_factor, axis=1)

    # Base EP for a full match from Poisson-style model
    preds["base_ep"] = preds.apply(base_ep_from_poisson, axis=1)

    # Final predicted points (raw)
    preds["predicted_points"] = preds["base_ep"] * preds["minutes_multiplier"]

    # Robust EP: penalise volatility
    alpha = ROBUST_ALPHA
    preds["robust_predicted_points"] = preds["predicted_points"] - alpha * preds["volatility_score"]
    preds["robust_predicted_points"] = preds["robust_predicted_points"].clip(lower=0.0)

    keep = [
        "id",
        "predicted_points",
        "robust_predicted_points",
        "lambda_for",
        "lambda_against",
        "cs_prob",
        "home_away",
    ]
    return preds[keep]


def multi_gw_predictions(players, fixtures, start_gw, num_gws):
    """
    For each GW in [start_gw, start_gw + num_gws - 1]:
    - Build a Poisson-style fixture model from team strengths.
    - Predict GW points for every player (raw and robust).
    - Sum over the horizon to get multi_gw_points (raw) and multi_gw_points_robust.
    """
    team_model = compute_team_goal_model(fixtures)

    results = players.copy()
    total_raw = np.zeros(len(players))
    total_robust = np.zeros(len(players))

    for gw in range(start_gw, start_gw + num_gws):
        fix_df = build_fixture_difficulty_df(fixtures, gw, team_model)
        colname = f"gw{gw}_points"
        robust_colname = f"{colname}_robust"

        if fix_df.empty:
            results[colname] = 0.0
            results[robust_colname] = 0.0
            continue

        gw_preds = predict_gw_points(players, fix_df)
        results = results.merge(
            gw_preds[["id", "predicted_points", "robust_predicted_points"]],
            on="id",
            how="left"
        )

        results[colname] = results["predicted_points"].fillna(0.0)
        results[robust_colname] = results["robust_predicted_points"].fillna(0.0)

        total_raw += results[colname]
        total_robust += results[robust_colname]

        # drop temporary columns before next GW
        results = results.drop(columns=["predicted_points", "robust_predicted_points"])

    results["multi_gw_points"] = total_raw
    results["multi_gw_points_robust"] = total_robust
    return results


def attach_manager_view(preds, picks_df):
    if picks_df is None or picks_df.empty:
        print("No picks data available; skipping personalised view.")
        preds["owned"] = False
        preds["is_captain"] = False
        preds["is_vice_captain"] = False
        preds["squad_position"] = np.nan
        preds["sell_price"] = np.nan
        return preds, preds[preds["owned"]]
    picks_df = picks_df.copy()
    picks_df.rename(columns={"element": "id"}, inplace=True)
    if "selling_price" in picks_df.columns:
        picks_df["sell_price"] = picks_df["selling_price"] / 10.0
    else:
        picks_df["sell_price"] = np.nan
    merged = preds.merge(
        picks_df[["id", "position", "multiplier", "is_captain", "is_vice_captain", "sell_price"]],
        on="id",
        how="left",
        suffixes=("", "_pick"),
    )
    merged["owned"] = merged["position_pick"].notna()
    merged["squad_position"] = merged["position_pick"]
    merged["is_captain"] = merged["is_captain"].fillna(False)
    merged["is_vice_captain"] = merged["is_vice_captain"].fillna(False)
    myteam = merged[merged["owned"]].copy()
    return merged, myteam

def suggest_best_single_transfers_multi_gw(preds_all, myteam, bank_tenths):
    """1-transfer optimiser (multi GW), enforcing max 3 per club.
       Uses robust EP if available.
    """
    if myteam.empty:
        return None

    # Choose metric: robust if available, else raw
    metric = "multi_gw_points_robust" if "multi_gw_points_robust" in preds_all.columns else "multi_gw_points"

    bank = (bank_tenths or 0) / 10.0
    candidates_all = preds_all.sort_values(metric, ascending=False)
    club_counts = myteam["team_short"].value_counts().to_dict()
    suggestions = []

    for _, row in myteam.iterrows():
        current_id = row["id"]
        pos = row["position"]
        current_pred = float(row.get(metric, 0.0))
        sell_price = row.get("sell_price", np.nan)
        sell_team = row["team_short"]

        if pd.isna(sell_price):
            sell_price = row["price"]

        max_price = sell_price + bank

        club_counts_post_sell = club_counts.copy()
        club_counts_post_sell[sell_team] = club_counts_post_sell.get(sell_team, 1) - 1

        pool = candidates_all[
            (candidates_all["position"] == pos)
            & (candidates_all["id"] != current_id)
            & (candidates_all["price"] <= max_price + 1e-6)
            & (candidates_all["status"].isin(["a", "d"]))
        ].copy()

        def allowed(row2):
            team = row2["team_short"]
            after = club_counts_post_sell.get(team, 0) + 1
            return after <= 3

        pool = pool[pool.apply(allowed, axis=1)]
        if pool.empty:
            continue

        best = pool.iloc[0]
        best_pred = float(best.get(metric, 0.0))
        gain = best_pred - current_pred
        if gain <= 0:
            continue

        suggestions.append(
            {
                "position": pos,
                "sell_name": row["web_name"],
                "sell_team": sell_team,
                "sell_price": round(float(sell_price), 1),
                "sell_pred": round(current_pred, 3),
                "buy_name": best["web_name"],
                "buy_team": best["team_short"],
                "buy_price": round(float(best["price"]), 1),
                "buy_pred": round(best_pred, 3),
                "gain": round(float(gain), 3),
            }
        )

    if not suggestions:
        return None

    suggestions_df = pd.DataFrame(suggestions).sort_values("gain", ascending=False)
    return suggestions_df

def suggest_best_double_transfers_multi_gw(
    preds_all,
    myteam,
    bank_tenths,
    hit_cost=4,
    free_transfers=1,
    max_candidates_per_position=25,
):
    """
    Hybrid 2-transfer optimiser (multi-GW):

    - Uses robust metric if available: multi_gw_points_robust, else multi_gw_points.
    - Brute-force over pairs of SELL players from your squad.
    - For each pair, searches among top N candidates per position.
    - Enforces:
        * budget (sell prices + bank)
        * 3-per-club limit
        * same positions as the sold players
        * only 'a' or 'd' status (available/doubtful)
    - Applies hit cost based on free_transfers and hit_cost.

    Returns:
        DataFrame sorted by net_gain (best first), or None if no profitable combos.
    """
    if myteam.empty or len(myteam) < 2:
        return None

    # Use robust metric if present, otherwise raw
    metric = "multi_gw_points_robust" if "multi_gw_points_robust" in preds_all.columns else "multi_gw_points"

    bank = (bank_tenths or 0) / 10.0

    # Sort global candidates by chosen metric
    candidates_all = preds_all.sort_values(metric, ascending=False).copy()

    # Build per-position candidate pools, capped in size for speed
    pos_top = {}
    for pos in ["GKP", "GK", "DEF", "MID", "FWD"]:
        pos_df = candidates_all[candidates_all["position"] == pos]
        if not pos_df.empty:
            pos_top[pos] = pos_df.head(max_candidates_per_position).copy()

    # Current club counts in your squad
    club_counts = myteam["team_short"].value_counts().to_dict()
    suggestions = []

    team_df = myteam.reset_index(drop=True)
    n = len(team_df)

    for i in range(n):
        row1 = team_df.iloc[i]
        for j in range(i + 1, n):
            row2 = team_df.iloc[j]

            pos1, pos2 = row1["position"], row2["position"]

            # Skip if we don't have candidates for one of the positions
            if pos1 not in pos_top or pos2 not in pos_top:
                continue

            # Current projected points for the two players being sold
            current_pred1 = float(row1.get(metric, 0.0))
            current_pred2 = float(row2.get(metric, 0.0))
            current_points = current_pred1 + current_pred2

            # Selling prices (fallback to now price if missing)
            sell_price1 = row1.get("sell_price", np.nan)
            sell_price2 = row2.get("sell_price", np.nan)
            if pd.isna(sell_price1):
                sell_price1 = row1["price"]
            if pd.isna(sell_price2):
                sell_price2 = row2["price"]

            total_budget = sell_price1 + sell_price2 + bank

            # Club counts after selling these two
            cc_after_sell = club_counts.copy()
            cc_after_sell[row1["team_short"]] = cc_after_sell.get(row1["team_short"], 1) - 1
            cc_after_sell[row2["team_short"]] = cc_after_sell.get(row2["team_short"], 1) - 1

            # Build candidate pools for each position
            pool1 = pos_top[pos1].copy()
            pool2 = pos_top[pos2].copy()

            # Exclude the players we're selling
            pool1 = pool1[pool1["id"] != row1["id"]]
            pool2 = pool2[pool2["id"] != row2["id"]]

            # Only consider available or doubtful players
            pool1 = pool1[pool1["status"].isin(["a", "d"])]
            pool2 = pool2[pool2["status"].isin(["a", "d"])]

            if pool1.empty or pool2.empty:
                continue

            # Hybrid search: iterate over all pairs in the trimmed pools
            for _, buy1 in pool1.iterrows():
                for _, buy2 in pool2.iterrows():
                    # Can't buy the same player twice
                    if buy1["id"] == buy2["id"]:
                        continue

                    price1 = float(buy1["price"])
                    price2 = float(buy2["price"])

                    # Budget constraint
                    if price1 + price2 > total_budget + 1e-6:
                        continue

                    # 3-per-club constraint after transfers
                    tmp_counts = cc_after_sell.copy()
                    t1 = buy1["team_short"]
                    t2 = buy2["team_short"]
                    tmp_counts[t1] = tmp_counts.get(t1, 0) + 1
                    tmp_counts[t2] = tmp_counts.get(t2, 0) + 1

                    if max(tmp_counts.values()) > 3:
                        continue

                    # Projected points for the two incoming players
                    new_points = float(buy1.get(metric, 0.0)) + float(buy2.get(metric, 0.0))
                    gain = new_points - current_points
                    if gain <= 0:
                        continue

                    # Hit cost: extra transfers beyond free_transfers
                    n_transfers = 2
                    extra_transfers = max(0, n_transfers - free_transfers)
                    hit_points = extra_transfers * hit_cost
                    net_gain = gain - hit_points

                    suggestions.append({
                        "sell1": row1["web_name"],
                        "sell1_team": row1["team_short"],
                        "sell1_price": round(float(sell_price1), 1),
                        "sell1_pred": round(current_pred1, 3),

                        "buy1": buy1["web_name"],
                        "buy1_team": buy1["team_short"],
                        "buy1_price": round(price1, 1),
                        "buy1_pred": round(float(buy1.get(metric, 0.0)), 3),

                        "sell2": row2["web_name"],
                        "sell2_team": row2["team_short"],
                        "sell2_price": round(float(sell_price2), 1),
                        "sell2_pred": round(current_pred2, 3),

                        "buy2": buy2["web_name"],
                        "buy2_team": buy2["team_short"],
                        "buy2_price": round(price2, 1),
                        "buy2_pred": round(float(buy2.get(metric, 0.0)), 3),

                        "gain": round(float(gain), 3),
                        "hit_points": hit_points,
                        "net_gain": round(float(net_gain), 3),
                    })

    if not suggestions:
        return None

    suggestions_df = pd.DataFrame(suggestions).sort_values("net_gain", ascending=False)
    return suggestions_df

from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus


def optimise_wildcard_squad_milp(preds_all, total_budget_m, max_per_team=3):
    """
    MILP: choose the best 15-man squad (2 GKP, 5 DEF, 5 MID, 3 FWD)
    within a given budget using robust multi-GW points.

    - preds_all: DataFrame with at least:
        ["id", "web_name", "team_short", "position", "price",
         "multi_gw_points_robust" or "multi_gw_points"]
    - total_budget_m: budget in millions (float), e.g. 100.0
    """

    df = preds_all.copy()

    # Use robust EP if available, otherwise raw
    metric = "multi_gw_points_robust" if "multi_gw_points_robust" in df.columns else "multi_gw_points"
    if metric not in df.columns:
        raise ValueError("Expected multi_gw_points(_robust) in preds_all for MILP optimiser.")

    # Basic cleaning
    df = df.dropna(subset=[metric, "price", "position", "team_short"])
    df.reset_index(drop=True, inplace=True)

    n = len(df)
    if n == 0:
        print("No players available for MILP optimisation.")
        return None

    # Position requirements (standard FPL)
    POS_REQ = {
        "GKP": 2,
        "GK": 2,   # in case goalkeeper is labelled GK
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }

    # --- Define the MILP problem ---
    prob = LpProblem("FPL_Wildcard_MILP", LpMaximize)

    # Binary decision variable x_i: 1 if player i is in the final squad
    x = [
        LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(n)
    ]

    # --- Objective: maximise total robust expected points ---
    prob += lpSum(df[metric].iloc[i] * x[i] for i in range(n)), "Total_Robust_EP"

    # --- Constraint: total players = 15 ---
    prob += lpSum(x) == 15, "TotalPlayers"

    # --- Constraints: position counts ---
    for pos, req in POS_REQ.items():
        idx = df[df["position"] == pos].index.tolist()
        if pos in ("GKP", "GK"):
            # Treat GK and GKP as the same pool
            idx = df[df["position"].isin(["GKP", "GK"])].index.tolist()
        if not idx:
            continue
        prob += lpSum(x[i] for i in idx) == req, f"Pos_{pos}"

    # --- Constraint: max 3 players per team ---
    for team in df["team_short"].unique():
        idx = df[df["team_short"] == team].index.tolist()
        prob += lpSum(x[i] for i in idx) <= max_per_team, f"Team_{team}_limit"

    # --- Constraint: budget ---
    prob += lpSum(df["price"].iloc[i] * x[i] for i in range(n)) <= total_budget_m, "Budget"

    # --- Solve ---
    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        print(f"MILP did not find an optimal solution (status={LpStatus[prob.status]}).")
        return None

    chosen_idx = [i for i in range(n) if x[i].value() == 1]
    squad = df.iloc[chosen_idx].copy()
    squad = squad.sort_values(["position", metric], ascending=[True, False])

    # Add some summary columns
    squad["selected_metric"] = metric
    squad["budget"] = total_budget_m
    squad["total_metric"] = squad[metric].sum()
    squad["total_price"] = squad["price"].sum()

    return squad

from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus


def optimise_transfers_milp(
    preds_all,
    preds_myteam,
    bank_tenths,
    value_tenths,
    free_transfers,
    hit_cost,
    max_additional_transfers=4,
    max_per_team=3,
):
    """
    MILP transfer optimiser.

    Decide which players to keep / sell / buy to form the best final 15-man squad
    (2 GK, 5 DEF, 5 MID, 3 FWD) subject to:
      - Budget (current squad value + bank)
      - Max 3 per team
      - Limited number of transfers with hit cost

    Uses multi_gw_points_robust if present, otherwise multi_gw_points.

    Returns:
      final_squad_df, transfers_df  (or (None, None) if no solution)
    """

    if preds_myteam is None or preds_myteam.empty:
        print("No current team data for MILP transfers.")
        return None, None

    df = preds_all.copy()

    # Use robust EP if available, otherwise raw
    metric = (
        "multi_gw_points_robust"
        if "multi_gw_points_robust" in df.columns
        else "multi_gw_points"
    )
    if metric not in df.columns:
        raise ValueError(
            "Expected multi_gw_points or multi_gw_points_robust in preds_all."
        )

    # Only consider playable players (a = available, d = doubt)
    df = df[df["status"].isin(["a", "d"])].copy()

    # We need these columns
    required_cols = ["id", "team_short", "position", "price", "owned"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"MILP transfer optimiser needs column '{c}' in preds_all.")

    # If sell_price missing, fall back to price
    if "sell_price" not in df.columns:
        df["sell_price"] = df["price"]

    # Clean + reset index
    df.reset_index(drop=True, inplace=True)
    n = len(df)
    if n == 0:
        print("No players available for MILP transfers.")
        return None, None

    # Position requirements
    POS_REQ = {
        "GKP": 2,
        "GK": 2,   # handle GK vs GKP naming
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }

    # Budget in millions: current squad value + bank
    if value_tenths is None or bank_tenths is None:
        total_budget_m = 100.0
    else:
        total_budget_m = (value_tenths + bank_tenths) / 10.0

    # Indices by ownership
    owned_mask = df["owned"].astype(bool)
    idx_owned = df[owned_mask].index.tolist()
    idx_not_owned = df[~owned_mask].index.tolist()

    # --- Define the MILP problem ---
    prob = LpProblem("FPL_Transfers_MILP", LpMaximize)

    # Decision variable: x[i] = 1 if player i is in the final squad
    x = [
        LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in range(n)
    ]

    # out[i] for currently owned players: 1 if we sell them
    out = {
        i: LpVariable(f"out_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in idx_owned
    }

    # in[j] for not-owned players: 1 if we buy them
    _in = {
        j: LpVariable(f"in_{j}", lowBound=0, upBound=1, cat="Binary")
        for j in idx_not_owned
    }

    # extra transfers beyond free_transfers
    extra = LpVariable("extra_transfers", lowBound=0, cat="Integer")

    # --- Objective: maximise total robust EP minus hit cost ---
    prob += (
        lpSum(df[metric].iloc[i] * x[i] for i in range(n))
        - hit_cost * extra
    ), "Total_Robust_EP_minus_Hits"

    # --- Squad size: 15 players ---
    prob += lpSum(x) == 15, "TotalPlayers"

    # --- Position constraints ---
    # GK/GKP combined handling
    gk_idx = df[df["position"].isin(["GK", "GKP"])].index.tolist()
    if gk_idx:
        prob += lpSum(x[i] for i in gk_idx) == 2, "GK_Count"

    for pos in ["DEF", "MID", "FWD"]:
        idx_pos = df[df["position"] == pos].index.tolist()
        if idx_pos:
            prob += lpSum(x[i] for i in idx_pos) == POS_REQ[pos], f"{pos}_Count"

    # --- Max 3 per team ---
    for team in df["team_short"].unique():
        idx_team = df[df["team_short"] == team].index.tolist()
        prob += lpSum(x[i] for i in idx_team) <= max_per_team, f"Team_{team}_Limit"

    # --- Ownership flow constraints ---
    # For owned players: either kept (x=1) or sold (out=1)
    for i in idx_owned:
        prob += x[i] + out[i] == 1, f"Owned_keep_or_sell_{i}"

    # For not-owned players: if in final squad, must be bought (x == in)
    for j in idx_not_owned:
        prob += x[j] - _in[j] <= 0, f"NotOwned_in_if_selected1_{j}"
        prob += _in[j] - x[j] <= 0, f"NotOwned_in_if_selected2_{j}"

    # Transfers in and out must balance
    prob += lpSum(_in[j] for j in idx_not_owned) == lpSum(out[i] for i in idx_owned), "Buy_Sell_Balance"

    # Total transfers
    total_transfers = lpSum(_in[j] for j in idx_not_owned)

    # Limit total transfers to free + max_additional_transfers
    prob += total_transfers <= free_transfers + max_additional_transfers, "MaxTransfers"

    # extra >= transfers - free_transfers, extra >= 0
    prob += extra >= total_transfers - free_transfers, "ExtraTransfersLowerBound"
    prob += extra >= 0, "ExtraTransfersNonNeg"

    # --- Budget constraint ---
    prob += lpSum(df["price"].iloc[i] * x[i] for i in range(n)) <= total_budget_m, "Budget"

    # --- Solve ---
    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        print(f"MILP transfers: no optimal solution (status={LpStatus[prob.status]}).")
        return None, None

    # Build final squad
    chosen_idx = [i for i in range(n) if x[i].value() == 1]
    final_squad = df.iloc[chosen_idx].copy()
    final_squad = final_squad.sort_values(["position", metric], ascending=[True, False])

    # Build transfers dataframe
    transfers = []

    # Sold players (out=1)
    for i in idx_owned:
        if out[i].value() >= 0.5:
            row = df.iloc[i]
            transfers.append(
                {
                    "type": "SELL",
                    "id": int(row["id"]),
                    "web_name": row["web_name"],
                    "team_short": row["team_short"],
                    "position": row["position"],
                    "sell_price": float(row["sell_price"]),
                }
            )

    # Bought players (in=1)
    for j in idx_not_owned:
        if _in[j].value() >= 0.5:
            row = df.iloc[j]
            transfers.append(
                {
                    "type": "BUY",
                    "id": int(row["id"]),
                    "web_name": row["web_name"],
                    "team_short": row["team_short"],
                    "position": row["position"],
                    "buy_price": float(row["price"]),
                }
            )

    transfers_df = pd.DataFrame(transfers)

    # Add some summary info to final_squad
    final_squad["selected_metric"] = metric
    final_squad["total_metric"] = final_squad[metric].sum()
    final_squad["total_price"] = final_squad["price"].sum()
    final_squad["budget_limit"] = total_budget_m

    return final_squad, transfers_df

def optimise_transfers_milp_multi_horizon(
    preds_all,
    preds_myteam,
    bank_tenths,
    value_tenths,
    plan_gw,
    num_gws,
    free_transfers_first_gw,
    hit_cost,
    free_transfers_subsequent=1,
    max_extra_hits=8,
    max_per_team=3,
):
    """
    Multi-GW MILP transfer optimiser (sliding horizon, Path B).

    Plans transfers over a horizon of num_gws starting at plan_gw:
      GWs = [plan_gw, plan_gw+1, ..., plan_gw+num_gws-1]

    Decision variables:
      - x[i,g]   : 1 if player i is in your squad AFTER GW g (g=0..G)
      - in[i,g]  : 1 if player i is bought for GW g (g=1..G)
      - out[i,g] : 1 if player i is sold for GW g (g=1..G)
      - extra_g  : extra transfers for GW g beyond free transfers

    g=0 is your current squad (pre-plan); x[i,0] is fixed by 'owned'.

    Objective:
      maximise sum_g ( robust points for that GW * x[i,g] ) - hit_cost * sum_g extra_g

    Constraints:
      - Squad always 15 players each GW, with correct position counts
      - Max 3 per club each GW
      - Total squad price each GW <= (value + bank) in millions
      - Transfer flow: x[i,g] = x[i,g-1] + in[i,g] - out[i,g]
      - Transfers_g = sum(in[i,g]) = sum(out[i,g])
      - extra_g >= transfers_g - free_transfers_g
      - Sum of all extra_g <= max_extra_hits / hit_cost
    """

    if preds_myteam is None or preds_myteam.empty:
        print("No current team data for multi-GW MILP transfers.")
        return None, None

    df = preds_all.copy()

    # --------- choose per-GW scoring columns ----------
    horizon_gws = [plan_gw + k for k in range(num_gws)]
    gw_metric_cols = {}

    for gw in horizon_gws:
        robust_col = f"gw{gw}_points_robust"
        raw_col = f"gw{gw}_points"
        if robust_col in df.columns:
            gw_metric_cols[gw] = robust_col
        elif raw_col in df.columns:
            gw_metric_cols[gw] = raw_col
        else:
            raise ValueError(f"Missing per-GW EP columns for GW{gw} "
                             f"(expected {robust_col} or {raw_col}).")

    # Only consider available / doubtful players
    df = df[df["status"].isin(["a", "d"])].copy()

    required_cols = ["id", "team_short", "position", "price", "owned"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"multi-GW MILP needs column '{c}' in preds_all.")

    if "sell_price" not in df.columns:
        df["sell_price"] = df["price"]

    df.reset_index(drop=True, inplace=True)
    n = len(df)
    if n == 0:
        print("No players available for multi-GW MILP transfers.")
        return None, None

    # Position requirements
    POS_REQ = {"GKP": 2, "GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    # Budget (millions)
    if value_tenths is None or bank_tenths is None:
        total_budget_m = 100.0
    else:
        total_budget_m = (value_tenths + bank_tenths) / 10.0

    owned_mask = df["owned"].astype(bool)
    idx_owned = df[owned_mask].index.tolist()

    # ------------- set up MILP -------------
    G = len(horizon_gws)  # number of future GWs (1..G)

    prob = LpProblem("FPL_Transfers_MultiGW_MILP", LpMaximize)

    # Binary squad variable x[i,g], g=0..G (g=0 is pre-plan squad)
    x = {}
    for i in range(n):
        for g in range(G + 1):
            x[(i, g)] = LpVariable(f"x_{i}_{g}", lowBound=0, upBound=1, cat="Binary")

    # Transfers in/out for each GW (g=1..G)
    in_var = {}
    out_var = {}
    for i in range(n):
        for g in range(1, G + 1):
            in_var[(i, g)] = LpVariable(f"in_{i}_{g}", lowBound=0, upBound=1, cat="Binary")
            out_var[(i, g)] = LpVariable(f"out_{i}_{g}", lowBound=0, upBound=1, cat="Binary")

    # Extra transfers (beyond free) per GW
    extra_g = {
        g: LpVariable(f"extra_transfers_{g}", lowBound=0, cat="Integer")
        for g in range(1, G + 1)
    }

    # -------- objective: sum over GWs of EP - hit cost on all extra transfers ----------
    obj_terms = []
    for g_idx, gw in enumerate(horizon_gws, start=1):
        col = gw_metric_cols[gw]
        for i in range(n):
            obj_terms.append(df[col].iloc[i] * x[(i, g_idx)])

    obj_hits = hit_cost * lpSum(extra_g[g] for g in range(1, G + 1))
    prob += lpSum(obj_terms) - obj_hits, "Total_Robust_EP_minus_Hits_MultiGW"

    # -------- baseline squad at g=0 must be your current squad ----------
    for i in range(n):
        prob += x[(i, 0)] == (1.0 if i in idx_owned else 0.0), f"BaselineSquad_{i}"

    # -------- squad evolution + transfer counts per GW ----------
    for g in range(1, G + 1):

        # Flow: x[i,g] = x[i,g-1] + in - out
        for i in range(n):
            prob += (
                x[(i, g)]
                == x[(i, g - 1)] + in_var[(i, g)] - out_var[(i, g)]
            ), f"Flow_{i}_{g}"

        # Transfers in/out must balance
        prob += (
            lpSum(in_var[(i, g)] for i in range(n))
            == lpSum(out_var[(i, g)] for i in range(n))
        ), f"BuySellBalance_{g}"

        transfers_g = lpSum(in_var[(i, g)] for i in range(n))

        # Free transfers this GW
        free_g = free_transfers_first_gw if g == 1 else free_transfers_subsequent

        # extra_g >= transfers_g - free_g
        prob += extra_g[g] >= transfers_g - free_g, f"ExtraLB_{g}"
        prob += extra_g[g] >= 0, f"ExtraNonNeg_{g}"

    # Cap total extra transfers converted to hits
    max_extra_transfers_total = max_extra_hits / float(hit_cost) if hit_cost > 0 else 0.0
    prob += (
        lpSum(extra_g[g] for g in range(1, G + 1)) <= max_extra_transfers_total
    ), "MaxExtraTransfersTotal"

    # -------- squad composition + budget each GW ----------
    teams = df["team_short"].unique().tolist()

    # Precompute index sets
    gk_idx = df[df["position"].isin(["GK", "GKP"])].index.tolist()
    pos_idx = {pos: df[df["position"] == pos].index.tolist() for pos in ["DEF", "MID", "FWD"]}
    team_idx = {team: df[df["team_short"] == team].index.tolist() for team in teams}

    for g in range(1, G + 1):

        # 15-man squad
        prob += lpSum(x[(i, g)] for i in range(n)) == 15, f"TotalPlayers_{g}"

        # GK count
        if gk_idx:
            prob += (
                lpSum(x[(i, g)] for i in gk_idx) == 2
            ), f"GK_Count_{g}"

        # DEF/MID/FWD counts
        for pos in ["DEF", "MID", "FWD"]:
            idx_pos = pos_idx[pos]
            if idx_pos:
                prob += (
                    lpSum(x[(i, g)] for i in idx_pos) == POS_REQ[pos]
                ), f"{pos}_Count_{g}"

        # Max 3 per club
        for team in teams:
            idx_team = team_idx[team]
            prob += (
                lpSum(x[(i, g)] for i in idx_team) <= max_per_team
            ), f"Team_{team}_Limit_{g}"

        # Budget per GW
        prob += (
            lpSum(df["price"].iloc[i] * x[(i, g)] for i in range(n)) <= total_budget_m
        ), f"Budget_{g}"

    # -------- solve ----------
    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        print(f"Multi-GW MILP: no optimal solution (status={LpStatus[prob.status]}).")
        return None, None

    # -------- build results: squads & transfers per GW ----------
    squads_by_gw = {}
    for g_idx, gw in enumerate(horizon_gws, start=1):
        chosen_idx = [i for i in range(n) if x[(i, g_idx)].value() >= 0.5]
        squad = df.iloc[chosen_idx].copy()
        squad["gw"] = gw
        squads_by_gw[gw] = squad.sort_values(
            ["position", gw_metric_cols[gw]], ascending=[True, False]
        )

    transfers_rows = []
    for g_idx, gw in enumerate(horizon_gws, start=1):
        # buys
        for i in range(n):
            if in_var[(i, g_idx)].value() >= 0.5:
                row = df.iloc[i]
                transfers_rows.append(
                    {
                        "gw": gw,
                        "type": "BUY",
                        "id": int(row["id"]),
                        "web_name": row["web_name"],
                        "team_short": row["team_short"],
                        "position": row["position"],
                        "price": float(row["price"]),
                    }
                )
        # sells
        for i in range(n):
            if out_var[(i, g_idx)].value() >= 0.5:
                row = df.iloc[i]
                transfers_rows.append(
                    {
                        "gw": gw,
                        "type": "SELL",
                        "id": int(row["id"]),
                        "web_name": row["web_name"],
                        "team_short": row["team_short"],
                        "position": row["position"],
                        "price": float(row["sell_price"]),
                    }
                )

    transfers_df = pd.DataFrame(transfers_rows).sort_values(["gw", "type", "web_name"])

    return squads_by_gw, transfers_df



def main():
    print("=== FPL Prediction + Optimisation Engine (Auto-Mode) ===")

    # -------------------------------------------------------
    # 1. Load core data
    # -------------------------------------------------------
    data = fetch_bootstrap()
    fixtures = fetch_fixtures()
    players = build_players_df(data)

    # -------------------------------------------------------
    # 2. Detect CURRENT and NEXT gameweek automatically
    # -------------------------------------------------------
    events = data["events"]
    current_gw = None
    next_gw = None

    for ev in events:
        if ev.get("is_current"):
            current_gw = ev["id"]
        if ev.get("is_next"):
            next_gw = ev["id"]

    # Planning GW: normally the NEXT GW (upcoming deadline)
    if next_gw is not None:
        PLAN_GW = next_gw          # e.g. 13
        BASIS_GW = max(1, PLAN_GW - 1)  # use previous GW squad as pre-deadline team
    else:
        PLAN_GW = current_gw
        BASIS_GW = max(1, PLAN_GW - 1)

    print(f"Detected current GW: {current_gw}, next GW: {next_gw}")
    print(f"Planning from GW{PLAN_GW} onwards.")
    print(
        f"Fetching your squad from GW{BASIS_GW} as the basis "
        f"(your true pre-GW{PLAN_GW} team)."
    )

    # This is the picks GW we actually use
    PICKS_GW = BASIS_GW
    picks_df, bank_tenths, value_tenths = fetch_manager_picks_and_bank(
        TEAM_ID, PICKS_GW
    )

    # -------------------------------------------------------
    # 3. Auto-generate output filenames for this run
    # -------------------------------------------------------
    pred_start = PLAN_GW
    pred_end = PLAN_GW + NUM_GWS - 1

    global OUTPUT_ALL_CSV, OUTPUT_MYTEAM_CSV, OUTPUT_TRANSFERS_CSV, OUTPUT_DOUBLE_CSV, OUTPUT_MILP_PLAN_CSV

    OUTPUT_ALL_CSV = f"gw{pred_start}_to_gw{pred_end}_predictions_all.csv"
    OUTPUT_MYTEAM_CSV = f"gw{pred_start}_to_gw{pred_end}_predictions_myteam.csv"
    OUTPUT_TRANSFERS_CSV = f"gw{pred_start}_to_gw{pred_end}_transfer_suggestions.csv"
    OUTPUT_DOUBLE_CSV = f"gw{pred_start}_to_gw{pred_end}_double_transfers.csv"
    OUTPUT_MILP_PLAN_CSV = (
        f"gw{pred_start}_to_gw{pred_end}_multiGW_milp_plan.csv"
    )

    print(f"Predictions horizon: GW{pred_start}–GW{pred_end}")
    print("Output files will use this naming automatically.")

    # -------------------------------------------------------
    # 4. Predict points for the horizon (PLAN_GW → PLAN_GW+NUM_GWS-1)
    # -------------------------------------------------------
    preds = multi_gw_predictions(players, fixtures, PLAN_GW, NUM_GWS)

    # -------------------------------------------------------
    # 5. Attach your team view (if picks are available)
    # -------------------------------------------------------
    preds_all, preds_myteam = attach_manager_view(preds, picks_df)

    # === Choose which columns to export ===
    gw_cols = [f"gw{gw}_points" for gw in range(PLAN_GW, PLAN_GW + NUM_GWS)]
    gw_cols_robust = [
        f"gw{gw}_points_robust" for gw in range(PLAN_GW, PLAN_GW + NUM_GWS)
    ]

    base_cols = [
        "id",
        "web_name",
        "team_name",
        "team_short",
        "position",
        "price",
        "multi_gw_points",          # raw total EP
        "multi_gw_points_robust",   # robust (risk-adjusted) total EP
    ] + gw_cols + gw_cols_robust

    all_cols = base_cols + [
        "selected_by_percent",
        "status",
        "points_per_game",
        "form",
        "ga_per90",
        "volatility_score",
        "owned",
        "squad_position",
        "is_captain",
        "is_vice_captain",
        "sell_price",
    ]

    existing_cols = [c for c in all_cols if c in preds_all.columns]
    sort_metric_all = (
        "multi_gw_points_robust"
        if "multi_gw_points_robust" in preds_all.columns
        else "multi_gw_points"
    )
    preds_all_out = preds_all[existing_cols].sort_values(
        sort_metric_all, ascending=False
    )

    print(f"Saving ALL player predictions → {OUTPUT_ALL_CSV}")
    preds_all_out.to_csv(OUTPUT_ALL_CSV, index=False)

    # -------------------------------------------------------
    # 6. MILP Wildcard Optimisation (best possible squad)
    # -------------------------------------------------------
    try:
        if value_tenths is not None and bank_tenths is not None:
            total_budget_m = (value_tenths + bank_tenths) / 10.0
        else:
            total_budget_m = 100.0

        print(f"\n=== MILP Wildcard Squad (Budget ≈ £{total_budget_m:.1f}m) ===")
        milp_squad_wc = optimise_wildcard_squad_milp(preds_all, total_budget_m)

        if milp_squad_wc is not None:
            cols_to_show = [
                "web_name",
                "team_short",
                "position",
                "price",
                "multi_gw_points",
            ]
            if "multi_gw_points_robust" in milp_squad_wc.columns:
                cols_to_show.append("multi_gw_points_robust")

            print(milp_squad_wc[cols_to_show])
            print(
                f"\nMILP total {milp_squad_wc['selected_metric'].iloc[0]} "
                f"= {milp_squad_wc['total_metric'].iloc[0]:.2f}, "
                f"Total price = £{milp_squad_wc['total_price'].iloc[0]:.1f}m"
            )
        else:
            print("MILP squad could not be constructed.")
    except Exception as e:
        print(f"MILP optimisation failed with error: {e}")

    # -------------------------------------------------------
    # 7. If we have your team, save personalised output + transfers
    # -------------------------------------------------------
    if not preds_myteam.empty:
        existing_my_cols = [c for c in all_cols if c in preds_myteam.columns]
        sort_metric_my = (
            "multi_gw_points_robust"
            if "multi_gw_points_robust" in preds_myteam.columns
            else "multi_gw_points"
        )
        preds_myteam_out = preds_myteam[existing_my_cols].sort_values(
            sort_metric_my, ascending=False
        )

        print(f"Saving YOUR TEAM predictions → {OUTPUT_MYTEAM_CSV}")
        preds_myteam_out.to_csv(OUTPUT_MYTEAM_CSV, index=False)

        # Captain choices
        print("\n=== Suggested Captain Choices (Your Squad) ===")
        top_owned = preds_myteam_out.head(5)
        print(top_owned[["web_name", "team_short", "position", "multi_gw_points"]])

        current_cap = preds_myteam[preds_myteam["is_captain"]]
        if not current_cap.empty:
            print("\nYour CURRENT captain:")
            print(
                current_cap[
                    ["web_name", "team_short", "position", "multi_gw_points"]
                ]
            )

        # Single-transfer ideas
        print("\n=== Single-Transfer Suggestions ===")
        suggestions_df = suggest_best_single_transfers_multi_gw(
            preds_all, preds_myteam, bank_tenths
        )
        if suggestions_df is not None:
            print(suggestions_df.head(10))
            print(f"Saving → {OUTPUT_TRANSFERS_CSV}")
            suggestions_df.to_csv(OUTPUT_TRANSFERS_CSV, index=False)
        else:
            print("No profitable single-transfer upgrades found.")

        # Double-transfer ideas (hybrid option C)
        print("\n=== Double-Transfer Suggestions ===")
        double_df = suggest_best_double_transfers_multi_gw(
            preds_all,
            preds_myteam,
            bank_tenths,
            hit_cost=HIT_COST,
            free_transfers=FREE_TRANSFERS,
        )
        if double_df is not None:
            double_df = double_df[double_df["net_gain"] > 0]
            print(double_df.head(5))
            print(f"Saving → {OUTPUT_DOUBLE_CSV}")
            double_df.to_csv(OUTPUT_DOUBLE_CSV, index=False)
        else:
            print("No profitable 2-transfer combos found.")

        # ---------------------------------------------------
        # 8. MILP transfer optimiser (full-team optimisation)
        # ---------------------------------------------------
        print("\n=== MILP Transfer Optimiser (full squad) ===")
        try:
            milp_squad, milp_transfers = optimise_transfers_milp(
                preds_all,
                preds_myteam,
                bank_tenths,
                value_tenths,
                FREE_TRANSFERS,
                HIT_COST,
                max_additional_transfers=MAX_ADDITIONAL_TRANSFERS,
            )

            if milp_squad is not None:
                show_cols = [
                    "web_name",
                    "team_short",
                    "position",
                    "price",
                    "multi_gw_points",
                ]
                if "multi_gw_points_robust" in milp_squad.columns:
                    show_cols.append("multi_gw_points_robust")

                print("\nOptimal final squad (MILP):")
                print(milp_squad[show_cols])

                print(
                    f"\nMILP total {milp_squad['selected_metric'].iloc[0]} "
                    f"= {milp_squad['total_metric'].iloc[0]:.2f}, "
                    f"Total price = £{milp_squad['total_price'].iloc[0]:.1f}m "
                    f"(budget limit £{milp_squad['budget_limit'].iloc[0]:.1f}m)"
                )

                if milp_transfers is not None and not milp_transfers.empty:
                    print("\nMILP suggested transfers:")
                    print(milp_transfers)
            else:
                print("MILP transfer optimisation did not return a squad.")
        except Exception as e:
            print(f"MILP transfer optimiser failed with error: {e}")

        print(
            f"\nBank: £{bank_tenths/10.0:.1f}m   "
            f"Squad Value: £{value_tenths/10.0:.1f}m"
        )

        # ---------------------------------------------------
        # 9. Multi-GW MILP transfer plan (sliding horizon)
        # ---------------------------------------------------
        print("\n=== Multi-GW MILP Transfer Plan (sliding horizon) ===")
        try:
            squads_by_gw, milp_plan_df = optimise_transfers_milp_multi_horizon(
                preds_all=preds_all,
                preds_myteam=preds_myteam,
                bank_tenths=bank_tenths,
                value_tenths=value_tenths,
                plan_gw=PLAN_GW,
                num_gws=NUM_GWS,
                free_transfers_first_gw=FREE_TRANSFERS,
                hit_cost=HIT_COST,
                free_transfers_subsequent=1,  # normal FPL behaviour
                max_extra_hits=8,              # "never more than –8"
                max_per_team=3,
            )

            if milp_plan_df is not None and not milp_plan_df.empty:
                print("\nMulti-GW MILP transfer plan (first 20 rows):")
                print(milp_plan_df.head(20))

                print(f"\nSaving multi-GW MILP plan → {OUTPUT_MILP_PLAN_CSV}")
                milp_plan_df.to_csv(OUTPUT_MILP_PLAN_CSV, index=False)
            else:
                print("Multi-GW MILP optimiser did not return any transfer plan.")
        except Exception as e:
            print(f"Multi-GW MILP optimiser failed with error: {e}")

    else:
        print("\n⚠ No picks available for your GW. Could not build personalised view.")

    print("\nDone.")

if __name__ == "__main__":
    main()
