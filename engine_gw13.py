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
# (higher = more risk averse; 0.3–0.7 is a good range)
ROBUST_ALPHA = 0.3

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
        "id",
        "web_name",
        "team",
        "team_name",
        "team_short",
        "position",
        "now_cost",
        "total_points",
        "minutes",
        "points_per_game",
        "form",
        "goals_scored",
        "assists",
        "clean_sheets",
        "chance_of_playing_next_round",
        "selected_by_percent",
        "status",
    ]
    players = players[keep_cols]

    players["price"] = players["now_cost"] / 10.0
    players["points_per_game"] = pd.to_numeric(
        players["points_per_game"], errors="coerce"
    ).fillna(0.0)
    players["form"] = pd.to_numeric(players["form"], errors="coerce").fillna(0.0)
    players["selected_by_percent"] = pd.to_numeric(
        players["selected_by_percent"], errors="coerce"
    ).fillna(0.0)

    minutes = players["minutes"].replace(0, np.nan)
    players["g_per90"] = players["goals_scored"] / minutes * 90
    players["a_per90"] = players["assists"] / minutes * 90

    players["g_per90"] = (
        players["g_per90"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    players["a_per90"] = (
        players["a_per90"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    # Cap per-90 rates to realistic ranges
    players["g_per90"] = players["g_per90"].clip(0, 1.2)
    players["a_per90"] = players["a_per90"].clip(0, 1.2)

    players["ga_per90"] = players["g_per90"] + players["a_per90"]

    # --- VOLATILITY SCORE ---
    # Simple proxy: attacking involvement + instability in recent form
    vol = 0.5 * players["ga_per90"].fillna(0.0) + 0.5 * (
        players["form"] - players["points_per_game"]
    ).abs()
    players["volatility_score"] = vol.clip(0.0, 4.0)
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
        return pd.DataFrame(
            {"team_id": teams, "att_strength": 1.0, "def_strength": 1.0}
        )

    # Restrict to last N gameweeks
    if "event" in hist.columns:
        max_gw = hist["event"].max()
        hist = hist[hist["event"] >= max_gw - window_gws + 1]

    rows = []
    for _, r in hist.iterrows():
        rows.append(
            {
                "team_id": r["team_h"],
                "goals_for": r["team_h_score"],
                "goals_against": r["team_a_score"],
            }
        )
        rows.append(
            {
                "team_id": r["team_a"],
                "goals_for": r["team_a_score"],
                "goals_against": r["team_h_score"],
            }
        )
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
        return pd.DataFrame(
            {"team_id": teams, "att_strength": 1.0, "def_strength": 1.0}
        )

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
    tm_a = team_model.rename(
        columns={
            "team_id": "team_a",
            "att_strength": "att_strength_a",
            "def_strength": "def_strength_a",
        }
    )
    gw_fx = gw_fx.merge(tm_a, on="team_a", how="left")

    # Fill missing strengths with neutral 1.0
    for col in ["att_strength", "def_strength", "att_strength_a", "def_strength_a"]:
        if col not in gw_fx.columns:
            gw_fx[col] = 1.0
    gw_fx[
        ["att_strength", "def_strength", "att_strength_a", "def_strength_a"]
    ] = gw_fx[
        ["att_strength", "def_strength", "att_strength_a", "def_strength_a"]
    ].fillna(
        1.0
    )

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
        rows.append(
            {
                "team_id": r["team_h"],
                "opponent_id": r["team_a"],
                "home_away": "H",
                "lambda_for": lam_home,
                "lambda_against": lam_away,
                "scoring_factor": lam_home / base_lambda,
                "cs_prob": float(np.exp(-lam_away)),  # P(0 goals against)
            }
        )
        # Away side record
        rows.append(
            {
                "team_id": r["team_a"],
                "opponent_id": r["team_h"],
                "home_away": "A",
                "lambda_for": lam_away,
                "lambda_against": lam_home,
                "scoring_factor": lam_away / base_lambda,
                "cs_prob": float(np.exp(-lam_home)),
            }
        )

    df = pd.DataFrame(rows)
    df = df.groupby("team_id", as_index=False).first()
    return df


def difficulty_to_multiplier(difficulty):
    if pd.isna(difficulty):
        return 1.0
    d = float(difficulty)
    if d <= 1:
        return 1.25
    elif d == 2:
        return 1.12
    elif d == 3:
        return 1.00
    elif d == 4:
        return 0.90
    else:
        return 0.80


def minutes_factor(row):
    status = row["status"]
    if status in ("i", "s"):
        return 0.0
    cop = row.get("chance_of_playing_next_round", None)
    if pd.notna(cop):
        return float(cop) / 100.0
    minutes = row.get("minutes", 0)
    if minutes <= 0:
        return 0.0
    if minutes >= 900:
        return 0.9
    elif minutes >= 450:
        return 0.7
    elif minutes >= 180:
        return 0.5
    else:
        return 0.3


def base_ep_from_poisson(row):
    """
    Expected FPL points for a *full match* using:
      - team attacking λ_for / λ_against
      - clean sheet probability
      - player attacking involvement (xGI + historical GA)
      - small form/PPG bump

    xG/xA logic:
      - If xg_coverage_flag == 1 -> involvement = 0.6 * xGI + 0.4 * GA_per90
      - Else -> involvement = GA_per90 only (no xG penalty).

    Uses shrunk/modelled xGI/90 if available to avoid tiny-sample explosions.
    """
    pos = row["position"]

    lam_for = float(row.get("lambda_for", 1.2) or 1.2)
    lam_against = float(row.get("lambda_against", 1.2) or 1.2)
    cs_prob = float(row.get("cs_prob", 0.0) or 0.0)

    # Prefer modelled (shrunk/capped) xGI/90 if present
    xgi90 = float(
        row.get("xgi_per90_model", row.get("xgi_per90", 0.0)) or 0.0
    )
    ga90 = float(row.get("ga_per90", 0.0) or 0.0)
    has_xg = int(row.get("xg_coverage_flag", 0) or 0)

    if has_xg:
        involvement = 0.6 * xgi90 + 0.4 * ga90
    else:
        involvement = ga90

    # Position-specific scoring factors
    if pos in ("FWD",):
        goal_factor = 4.0
        assist_factor = 3.0
        base_points = 2.5
    elif pos in ("MID",):
        goal_factor = 5.0
        assist_factor = 3.0
        base_points = 2.7
    elif pos in ("DEF", "GKP", "GK"):
        goal_factor = 6.0
        assist_factor = 3.0
        base_points = 3.0
    else:
        goal_factor = 4.0
        assist_factor = 3.0
        base_points = 2.0

    expected_goals = involvement * 0.6
    expected_assists = involvement * 0.4

    atk_points = expected_goals * goal_factor + expected_assists * assist_factor

    # Clean sheet component
    cs_points = 0.0
    if pos in ("DEF", "GKP", "GK"):
        cs_points = cs_prob * 3.5
    elif pos == "MID":
        cs_points = cs_prob * 0.8

    defence_penalty = max(0.0, (lam_against - 1.2)) * 0.4

    # Mild form/PPG bump
    form = float(row.get("form", 0.0) or 0.0)
    ppg = float(row.get("points_per_game", 0.0) or 0.0)
    form_boost = 0.05 * form + 0.03 * ppg

    ep_full_match = base_points + atk_points + cs_points - defence_penalty + form_boost
    return max(ep_full_match, 0.0)


def predict_gw_points(players, fixture_df):
    """
    Merge player data with fixture Poisson model and compute per-GW EP.
    FULL PLAYER DATA IS PRESERVED (critical for xgi_per90_model, ga_per90, minutes, etc.)
    """

    # Merge full player dataframe with fixture difficulty model
    preds = players.merge(
        fixture_df,
        left_on="team",
        right_on="team_id",
        how="left",
    )

    # Fill missing fixture model values
    preds["scoring_factor"] = preds["scoring_factor"].fillna(1.0).clip(0.6, 1.4)
    preds["cs_prob"] = preds["cs_prob"].fillna(0.0).clip(0.0, 1.0)

    # Ensure volatility exists
    preds["volatility_score"] = preds.get("volatility_score", 0.0).fillna(0.0)

    # Compute predicted minutes played (0–90 scale)
    preds["minutes_multiplier"] = preds.apply(minutes_factor, axis=1)

    # FULL EP (shrunk xGI model is used inside base_ep_from_poisson)
    preds["base_ep"] = preds.apply(base_ep_from_poisson, axis=1)

    # Multiply by expected minutes
    preds["predicted_points"] = preds["base_ep"] * preds["minutes_multiplier"]

    # Robust (risk-adjusted) EP
    alpha = ROBUST_ALPHA
    preds["robust_predicted_points"] = (
        preds["predicted_points"] - alpha * preds["volatility_score"]
    ).clip(lower=0.0)

    # DO NOT DROP ANY COLUMNS — return entire enriched preds
    return preds


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
            how="left",
        )

        results[colname] = results["predicted_points"].fillna(0.0)
        results[robust_colname] = results["robust_predicted_points"].fillna(0.0)

        total_raw += results[colname]
        total_robust += results[robust_colname]

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
        picks_df[
            ["id", "position", "multiplier", "is_captain", "is_vice_captain", "sell_price"]
        ],
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


def suggest_best_single_transfers_multi_gw(
    preds_all,
    myteam,
    bank_tenths,
    min_minutes: int = 300,
    min_selected_by: float = 0.5,
    min_form: float = 0.0,
    min_ppg: float = 0.0,
):
    """1-transfer optimiser (multi GW), enforcing max 3 per club.
       Uses robust EP if available. Includes safety filters.
    """
    if myteam.empty:
        return None

    metric = (
        "multi_gw_points_robust"
        if "multi_gw_points_robust" in preds_all.columns
        else "multi_gw_points"
    )

    bank = (bank_tenths or 0) / 10.0
    candidates_all = preds_all.copy()

    candidates_all["minutes"] = pd.to_numeric(
        candidates_all.get("minutes", 0), errors="coerce"
    ).fillna(0)
    candidates_all["selected_by_percent"] = pd.to_numeric(
        candidates_all.get("selected_by_percent", 0.0), errors="coerce"
    ).fillna(0.0)
    candidates_all["form"] = pd.to_numeric(
        candidates_all.get("form", 0.0), errors="coerce"
    ).fillna(0.0)
    candidates_all["points_per_game"] = pd.to_numeric(
        candidates_all.get("points_per_game", 0.0), errors="coerce"
    ).fillna(0.0)

    candidates_all = candidates_all[
        (candidates_all["minutes"] >= min_minutes)
        & (candidates_all["selected_by_percent"] >= min_selected_by)
        & (candidates_all["form"] >= min_form)
        & (candidates_all["points_per_game"] >= min_ppg)
    ]

    candidates_all = candidates_all.sort_values(metric, ascending=False)

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
    Hybrid 2-transfer optimiser (multi-GW), with safety filters.
    """
    if myteam.empty or len(myteam) < 2:
        return None

    metric = (
        "multi_gw_points_robust"
        if "multi_gw_points_robust" in preds_all.columns
        else "multi_gw_points"
    )

    bank = (bank_tenths or 0) / 10.0
    candidates_all = preds_all.sort_values(metric, ascending=False).copy()

    candidates_all["minutes"] = pd.to_numeric(
        candidates_all.get("minutes", 0), errors="coerce"
    ).fillna(0)
    candidates_all["selected_by_percent"] = pd.to_numeric(
        candidates_all.get("selected_by_percent", 0.0), errors="coerce"
    ).fillna(0.0)
    candidates_all["form"] = pd.to_numeric(
        candidates_all.get("form", 0.0), errors="coerce"
    ).fillna(0.0)
    candidates_all["points_per_game"] = pd.to_numeric(
        candidates_all.get("points_per_game", 0.0), errors="coerce"
    ).fillna(0.0)

    MIN_MINUTES = 300
    MIN_SELECTED = 0.5
    MIN_FORM = 0.0
    MIN_PPG = 0.0

    candidates_all = candidates_all[
        (candidates_all["minutes"] >= MIN_MINUTES)
        & (candidates_all["selected_by_percent"] >= MIN_SELECTED)
        & (candidates_all["form"] >= MIN_FORM)
        & (candidates_all["points_per_game"] >= MIN_PPG)
    ]

    pos_top = {}
    for pos in ["GKP", "GK", "DEF", "MID", "FWD"]:
        pos_df = candidates_all[candidates_all["position"] == pos]
        if not pos_df.empty:
            pos_top[pos] = pos_df.head(max_candidates_per_position).copy()

    club_counts = myteam["team_short"].value_counts().to_dict()
    suggestions = []

    team_df = myteam.reset_index(drop=True)
    n = len(team_df)

    for i in range(n):
        row1 = team_df.iloc[i]
        for j in range(i + 1, n):
            row2 = team_df.iloc[j]

            pos1, pos2 = row1["position"], row2["position"]
            if pos1 not in pos_top or pos2 not in pos_top:
                continue

            current_pred1 = float(row1.get(metric, 0.0))
            current_pred2 = float(row2.get(metric, 0.0))
            current_points = current_pred1 + current_pred2

            sell_price1 = row1.get("sell_price", np.nan)
            sell_price2 = row2.get("sell_price", np.nan)
            if pd.isna(sell_price1):
                sell_price1 = row1["price"]
            if pd.isna(sell_price2):
                sell_price2 = row2["price"]

            total_budget = sell_price1 + sell_price2 + bank

            cc_after_sell = club_counts.copy()
            cc_after_sell[row1["team_short"]] = cc_after_sell.get(
                row1["team_short"], 1
            ) - 1
            cc_after_sell[row2["team_short"]] = cc_after_sell.get(
                row2["team_short"], 1
            ) - 1

            pool1 = pos_top[pos1].copy()
            pool2 = pos_top[pos2].copy()

            pool1 = pool1[pool1["id"] != row1["id"]]
            pool2 = pool2[pool2["id"] != row2["id"]]

            pool1 = pool1[pool1["status"].isin(["a", "d"])]
            pool2 = pool2[pool2["status"].isin(["a", "d"])]

            if pool1.empty or pool2.empty:
                continue

            for _, buy1 in pool1.iterrows():
                for _, buy2 in pool2.iterrows():
                    if buy1["id"] == buy2["id"]:
                        continue

                    price1 = float(buy1["price"])
                    price2 = float(buy2["price"])

                    if price1 + price2 > total_budget + 1e-6:
                        continue

                    tmp_counts = cc_after_sell.copy()
                    t1 = buy1["team_short"]
                    t2 = buy2["team_short"]
                    tmp_counts[t1] = tmp_counts.get(t1, 0) + 1
                    tmp_counts[t2] = tmp_counts.get(t2, 0) + 1

                    if max(tmp_counts.values()) > 3:
                        continue

                    new_points = float(buy1.get(metric, 0.0)) + float(
                        buy2.get(metric, 0.0)
                    )
                    gain = new_points - current_points
                    if gain <= 0:
                        continue

                    n_transfers = 2
                    extra_transfers = max(0, n_transfers - free_transfers)
                    hit_points = extra_transfers * hit_cost
                    net_gain = gain - hit_points

                    suggestions.append(
                        {
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
                        }
                    )

    if not suggestions:
        return None

    suggestions_df = pd.DataFrame(suggestions).sort_values(
        "net_gain", ascending=False
    )
    return suggestions_df


from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus


def optimise_wildcard_squad_milp(preds_all, total_budget_m, max_per_team=3):
    """
    MILP: choose the best 15-man squad (2 GKP, 5 DEF, 5 MID, 3 FWD)
    within a given budget using robust multi-GW points.
    """
    df = preds_all.copy()

    metric = (
        "multi_gw_points_robust"
        if "multi_gw_points_robust" in df.columns
        else "multi_gw_points"
    )
    if metric not in df.columns:
        raise ValueError(
            "Expected multi_gw_points(_robust) in preds_all for MILP optimiser."
        )

    df = df.dropna(subset=[metric, "price", "position", "team_short"])
    df.reset_index(drop=True, inplace=True)

    n = len(df)
    if n == 0:
        print("No players available for MILP optimisation.")
        return None

    POS_REQ = {
        "GKP": 2,
        "GK": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }

    prob = LpProblem("FPL_Wildcard_MILP", LpMaximize)
    x = [LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(n)]

    prob += lpSum(df[metric].iloc[i] * x[i] for i in range(n)), "Total_Robust_EP"
    prob += lpSum(x) == 15, "TotalPlayers"

    for pos, req in POS_REQ.items():
        idx = df[df["position"] == pos].index.tolist()
        if pos in ("GKP", "GK"):
            idx = df[df["position"].isin(["GKP", "GK"])].index.tolist()
        if not idx:
            continue
        prob += lpSum(x[i] for i in idx) == req, f"Pos_{pos}"

    for team in df["team_short"].unique():
        idx = df[df["team_short"] == team].index.tolist()
        prob += lpSum(x[i] for i in idx) <= max_per_team, f"Team_{team}_limit"

    prob += (
        lpSum(df["price"].iloc[i] * x[i] for i in range(n)) <= total_budget_m
    ), "Budget"

    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        print(f"MILP did not find an optimal solution (status={LpStatus[prob.status]}).")
        return None

    chosen_idx = [i for i in range(n) if x[i].value() == 1]
    squad = df.iloc[chosen_idx].copy()
    squad = squad.sort_values(["position", metric], ascending=[True, False])

    squad["selected_metric"] = metric
    squad["budget"] = total_budget_m
    squad["total_metric"] = squad[metric].sum()
    squad["total_price"] = squad["price"].sum()

    return squad


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
    MILP transfer optimiser (single-horizon).
    """
    if preds_myteam is None or preds_myteam.empty:
        print("No current team data for MILP transfers.")
        return None, None

    df = preds_all.copy()

    metric = (
        "multi_gw_points_robust"
        if "multi_gw_points_robust" in df.columns
        else "multi_gw_points"
    )
    if metric not in df.columns:
        raise ValueError(
            "Expected multi_gw_points or multi_gw_points_robust in preds_all."
        )

    df = df[df["status"].isin(["a", "d"])].copy()

    required_cols = ["id", "team_short", "position", "price", "owned"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(
                f"MILP transfer optimiser needs column '{c}' in preds_all."
            )

    if "sell_price" not in df.columns:
        df["sell_price"] = df["price"]

    df.reset_index(drop=True, inplace=True)
    n = len(df)
    if n == 0:
        print("No players available for MILP transfers.")
        return None, None

    POS_REQ = {
        "GKP": 2,
        "GK": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }

    if value_tenths is None or bank_tenths is None:
        total_budget_m = 100.0
    else:
        total_budget_m = (value_tenths + bank_tenths) / 10.0

    owned_mask = df["owned"].astype(bool)
    idx_owned = df[owned_mask].index.tolist()
    idx_not_owned = df[~owned_mask].index.tolist()

    prob = LpProblem("FPL_Transfers_MILP", LpMaximize)

    x = [LpVariable(f"x_{i}", lowBound=0, upBound=1, cat="Binary") for i in range(n)]

    out = {
        i: LpVariable(f"out_{i}", lowBound=0, upBound=1, cat="Binary")
        for i in idx_owned
    }
    _in = {
        j: LpVariable(f"in_{j}", lowBound=0, upBound=1, cat="Binary")
        for j in idx_not_owned
    }

    extra = LpVariable("extra_transfers", lowBound=0, cat="Integer")

    prob += (
        lpSum(df[metric].iloc[i] * x[i] for i in range(n)) - hit_cost * extra
    ), "Total_Robust_EP_minus_Hits"

    prob += lpSum(x) == 15, "TotalPlayers"

    gk_idx = df[df["position"].isin(["GK", "GKP"])].index.tolist()
    if gk_idx:
        prob += lpSum(x[i] for i in gk_idx) == 2, "GK_Count"

    for pos in ["DEF", "MID", "FWD"]:
        idx_pos = df[df["position"] == pos].index.tolist()
        if idx_pos:
            prob += lpSum(x[i] for i in idx_pos) == POS_REQ[pos], f"{pos}_Count"

    for team in df["team_short"].unique():
        idx_team = df[df["team_short"] == team].index.tolist()
        prob += lpSum(x[i] for i in idx_team) <= max_per_team, f"Team_{team}_Limit"

    for i in idx_owned:
        prob += x[i] + out[i] == 1, f"Owned_keep_or_sell_{i}"

    for j in idx_not_owned:
        prob += x[j] - _in[j] <= 0, f"NotOwned_in_if_selected1_{j}"
        prob += _in[j] - x[j] <= 0, f"NotOwned_in_if_selected2_{j}"

    prob += (
        lpSum(_in[j] for j in idx_not_owned)
        == lpSum(out[i] for i in idx_owned)
    ), "Buy_Sell_Balance"

    total_transfers = lpSum(_in[j] for j in idx_not_owned)

    prob += (
        total_transfers <= free_transfers + max_additional_transfers
    ), "MaxTransfers"

    prob += extra >= total_transfers - free_transfers, "ExtraTransfersLowerBound"
    prob += extra >= 0, "ExtraTransfersNonNeg"

    prob += (
        lpSum(df["price"].iloc[i] * x[i] for i in range(n)) <= total_budget_m
    ), "Budget"

    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        print(
            f"MILP transfers: no optimal solution (status={LpStatus[prob.status]})."
        )
        return None, None

    chosen_idx = [i for i in range(n) if x[i].value() == 1]
    final_squad = df.iloc[chosen_idx].copy()
    final_squad = final_squad.sort_values(["position", metric], ascending=[True, False])

    transfers = []

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
    Multi-GW MILP transfer optimiser (sliding horizon).
    """
    if preds_myteam is None or preds_myteam.empty:
        print("No current team data for multi-GW MILP transfers.")
        return None, None

    df = preds_all.copy()

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
            raise ValueError(
                f"Missing per-GW EP columns for GW{gw} "
                f"(expected {robust_col} or {raw_col})."
            )

    df = df[df["status"].isin(["a", "d"])].copy()

    required_cols = ["id", "team_short", "position", "price", "owned", "sell_price"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"multi-GW MILP needs column '{c}' in preds_all.")

    df.reset_index(drop=True, inplace=True)
    n = len(df)
    if n == 0:
        print("No players available for multi-GW MILP transfers.")
        return None, None

    POS_REQ = {"GKP": 2, "GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

    if value_tenths is None or bank_tenths is None:
        total_budget_m = 100.0
    else:
        total_budget_m = (value_tenths + bank_tenths) / 10.0

    owned_mask = df["owned"].astype(bool)
    idx_owned = df[owned_mask].index.tolist()

    G = len(horizon_gws)

    prob = LpProblem("FPL_Transfers_MultiGW_MILP", LpMaximize)

    x = {}
    for i in range(n):
        for g in range(G + 1):
            x[(i, g)] = LpVariable(f"x_{i}_{g}", lowBound=0, upBound=1, cat="Binary")

    in_var = {}
    out_var = {}
    for i in range(n):
        for g in range(1, G + 1):
            in_var[(i, g)] = LpVariable(
                f"in_{i}_{g}", lowBound=0, upBound=1, cat="Binary"
            )
            out_var[(i, g)] = LpVariable(
                f"out_{i}_{g}", lowBound=0, upBound=1, cat="Binary"
            )

    extra_g = {
        g: LpVariable(f"extra_transfers_{g}", lowBound=0, cat="Integer")
        for g in range(1, G + 1)
    }

    obj_terms = []
    for g_idx, gw in enumerate(horizon_gws, start=1):
        col = gw_metric_cols[gw]
        for i in range(n):
            obj_terms.append(df[col].iloc[i] * x[(i, g_idx)])

    obj_hits = hit_cost * lpSum(extra_g[g] for g in range(1, G + 1))
    prob += lpSum(obj_terms) - obj_hits, "Total_Robust_EP_minus_Hits_MultiGW"

    for i in range(n):
        prob += x[(i, 0)] == (1.0 if i in idx_owned else 0.0), f"BaselineSquad_{i}"

    for g in range(1, G + 1):
        for i in range(n):
            prob += (
                x[(i, g)] == x[(i, g - 1)] + in_var[(i, g)] - out_var[(i, g)]
            ), f"Flow_{i}_{g}"

        prob += (
            lpSum(in_var[(i, g)] for i in range(n))
            == lpSum(out_var[(i, g)] for i in range(n))
        ), f"BuySellBalance_{g}"

        transfers_g = lpSum(in_var[(i, g)] for i in range(n))
        free_g = free_transfers_first_gw if g == 1 else free_transfers_subsequent

        prob += extra_g[g] >= transfers_g - free_g, f"ExtraLB_{g}"
        prob += extra_g[g] >= 0, f"ExtraNonNeg_{g}"

    max_extra_transfers_total = (
        max_extra_hits / float(hit_cost) if hit_cost > 0 else 0.0
    )
    prob += (
        lpSum(extra_g[g] for g in range(1, G + 1)) <= max_extra_transfers_total
    ), "MaxExtraTransfersTotal"

    teams = df["team_short"].unique().tolist()
    gk_idx = df[df["position"].isin(["GK", "GKP"])].index.tolist()
    pos_idx = {pos: df[df["position"] == pos].index.tolist() for pos in ["DEF", "MID", "FWD"]}
    team_idx = {team: df[df["team_short"] == team].index.tolist() for team in teams}

    for g in range(1, G + 1):
        prob += lpSum(x[(i, g)] for i in range(n)) == 15, f"TotalPlayers_{g}"

        if gk_idx:
            prob += lpSum(x[(i, g)] for i in gk_idx) == 2, f"GK_Count_{g}"

        for pos in ["DEF", "MID", "FWD"]:
            idx_pos = pos_idx[pos]
            if idx_pos:
                prob += (
                    lpSum(x[(i, g)] for i in idx_pos) == POS_REQ[pos]
                ), f"{pos}_Count_{g}"

        for team in teams:
            idx_team = team_idx[team]
            prob += (
                lpSum(x[(i, g)] for i in idx_team) <= max_per_team
            ), f"Team_{team}_Limit_{g}"

        prob += (
            lpSum(df["price"].iloc[i] * x[(i, g)] for i in range(n)) <= total_budget_m
        ), f"Budget_{g}"

    prob.solve()

    if LpStatus[prob.status] != "Optimal":
        print(f"Multi-GW MILP: no optimal solution (status={LpStatus[prob.status]}).")
        return None, None

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

    transfers_df = pd.DataFrame(transfers_rows).sort_values(
        ["gw", "type", "web_name"]
    )

    return squads_by_gw, transfers_df


def generate_email_summary(preds_myteam, preds_all, single_df, double_df, milp_squad):
    """
    Build a clean, human-readable summary for the email body.
    """
    lines = []
    lines.append("=== FPL Engine Summary ===\n")

    if not preds_myteam.empty:
        cap_metric = (
            "multi_gw_points_robust"
            if "multi_gw_points_robust" in preds_myteam.columns
            else "multi_gw_points"
        )
        best_cap = preds_myteam.sort_values(cap_metric, ascending=False).iloc[0]
        lines.append(f"⭐ Captain: {best_cap['web_name']} ({best_cap['team_short']})")
        lines.append(f"   Expected points: {best_cap[cap_metric]:.2f}\n")
    else:
        lines.append("⭐ Captain: No data available\n")

    if single_df is not None and not single_df.empty:
        best = single_df.iloc[0]
        lines.append("🔁 Best Single Transfer:")
        lines.append(f"   SELL: {best['sell_name']} ({best['sell_team']})")
        lines.append(f"   BUY:  {best['buy_name']} ({best['buy_team']})")
        lines.append(f"   Gain: {best['gain']:.2f} pts over horizon\n")
    else:
        lines.append("🔁 Best Single Transfer: None found\n")

    if double_df is not None and not double_df.empty:
        best2 = double_df.sort_values("net_gain", ascending=False).iloc[0]
        lines.append("🔁 Best Double Transfer:")
        lines.append(f"   Net gain: {best2['net_gain']:.2f} pts")
        lines.append("   (See CSV for full details)\n")
    else:
        lines.append("🔁 Best Double Transfer: None found\n")

    if milp_squad is not None and not milp_squad.empty:
        metric = milp_squad["selected_metric"].iloc[0]
        total = milp_squad["total_metric"].iloc[0]
        price = milp_squad["total_price"].iloc[0]
        lines.append("🧠 MILP Optimised Squad:")
        lines.append(f"   Optimises: {metric}")
        lines.append(f"   Total expected points: {total:.2f}")
        lines.append(f"   Total price: £{price:.1f}m\n")
    else:
        lines.append("🧠 MILP Squad: Not available\n")

    return "\n".join(lines)


def augment_players_with_expected_stats(
    players: pd.DataFrame,
    stats_path: str = "expected_stats_latest.csv",
):
    """
    Augment the players DataFrame with FPL-based expected stats
    from the pre-built CSV `expected_stats_latest.csv`.

    Adds:
      xg_per90, xa_per90, xgi_per90,
      xg_coverage_flag,
      xg_per90_model, xa_per90_model, xgi_per90_model (shrunk/capped).
    """
    try:
        stats = pd.read_csv(stats_path)
    except FileNotFoundError:
        print(
            f"[expected_stats] No file '{stats_path}' found. "
            "Proceeding without enrichment."
        )
        players["xg_per90"] = 0.0
        players["xa_per90"] = 0.0
        players["xgi_per90"] = 0.0
        players["xg_coverage_flag"] = 0.0
        players["xg_per90_model"] = 0.0
        players["xa_per90_model"] = 0.0
        players["xgi_per90_model"] = 0.0
        return players
    except Exception as e:
        print(f"[expected_stats] Failed to read '{stats_path}': {e}")
        players["xg_per90"] = 0.0
        players["xa_per90"] = 0.0
        players["xgi_per90"] = 0.0
        players["xg_coverage_flag"] = 0.0
        players["xg_per90_model"] = 0.0
        players["xa_per90_model"] = 0.0
        players["xgi_per90_model"] = 0.0
        return players

    for col in ["minutes", "xg", "xa", "xgi", "xg_per90", "xa_per90", "xgi_per90"]:
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce").fillna(0.0)

    if "xgi_per90" not in stats.columns or stats["xgi_per90"].sum() == 0:
        if {"xg", "xa", "minutes"}.issubset(stats.columns):
            minutes = stats["minutes"].replace(0, np.nan)
            xg90 = stats["xg"] / (minutes / 90.0)
            xa90 = stats["xa"] / (minutes / 90.0)
            stats["xg_per90"] = (
                xg90.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )
            stats["xa_per90"] = (
                xa90.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )
            stats["xgi_per90"] = stats["xg_per90"] + stats["xa_per90"]
        else:
            stats["xg_per90"] = 0.0
            stats["xa_per90"] = 0.0
            stats["xgi_per90"] = 0.0

    if "minutes" in stats.columns and "xgi_per90" in stats.columns:
        stats["xg_coverage_flag"] = (
            (stats["minutes"] >= 300) & (stats["xgi_per90"] > 0)
        ).astype(float)
    else:
        stats["xg_coverage_flag"] = 0.0

    # --- MINUTES-BASED SHRINKAGE & MODELLED PER-90 STATS ---
    if "minutes" in stats.columns and "xgi_per90" in stats.columns:
        MINUTES_PRIOR = 900.0

        total_minutes = float(stats["minutes"].sum())
        if "xgi" in stats.columns and stats["xgi"].sum() > 0 and total_minutes > 0:
            total_xgi = float(stats["xgi"].sum())
            league_xgi_per90 = total_xgi / (total_minutes / 90.0)
        elif total_minutes > 0:
            league_xgi_per90 = float(stats["xgi_per90"].mean())
        else:
            league_xgi_per90 = 0.25

        minutes = stats["minutes"].clip(lower=0.0)
        weight = minutes / (minutes + MINUTES_PRIOR)

        stats["xgi_per90_shrunk"] = league_xgi_per90 + weight * (
            stats["xgi_per90"] - league_xgi_per90
        )

        if "xg" in stats.columns and "xa" in stats.columns and "xgi" in stats.columns:
            denom = stats["xgi"].replace(0, 1e-6)
            xg_prop = stats["xg"] / denom
            xa_prop = stats["xa"] / denom
        else:
            denom = stats["xgi_per90"].replace(0, 1e-6)
            xg_prop = stats["xg_per90"] / denom
            xa_prop = stats["xa_per90"] / denom

        stats["xg_per90_shrunk"] = stats["xgi_per90_shrunk"] * xg_prop
        stats["xa_per90_shrunk"] = stats["xgi_per90_shrunk"] * xa_prop

        stats["xgi_per90_model"] = stats["xgi_per90_shrunk"].clip(upper=1.00)
        stats["xg_per90_model"] = stats["xg_per90_shrunk"].clip(upper=0.70)
        stats["xa_per90_model"] = stats["xa_per90_shrunk"].clip(upper=0.70)
    else:
        stats["xgi_per90_model"] = 0.0
        stats["xg_per90_model"] = 0.0
        stats["xa_per90_model"] = 0.0
    # -------------------------------------------------------

    cols_keep = [
        "web_name",
        "team_short",
        "xg_per90",
        "xa_per90",
        "xgi_per90",
        "xg_coverage_flag",
        "xg_per90_model",
        "xa_per90_model",
        "xgi_per90_model",
    ]
    stats = stats[[c for c in cols_keep if c in stats.columns]]

    merged = players.merge(
        stats,
        on=["web_name", "team_short"],
        how="left",
        suffixes=("", "_exp"),
    )

    for col, default in [
        ("xg_per90", 0.0),
        ("xa_per90", 0.0),
        ("xgi_per90", 0.0),
        ("xg_coverage_flag", 0.0),
        ("xg_per90_model", 0.0),
        ("xa_per90_model", 0.0),
        ("xgi_per90_model", 0.0),
    ]:
        if col not in merged.columns:
            merged[col] = default
        else:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(default)

    print("[expected_stats] Successfully merged FPL expected-stats into players.")
    print("[expected_stats] Applied minutes-based shrinkage to xGI per 90.")
    return merged


def main():
    print("=== FPL Prediction + Optimisation Engine (Auto-Mode) ===")

    data = fetch_bootstrap()
    fixtures = fetch_fixtures()
    players = build_players_df(data)
    players = augment_players_with_expected_stats(
        players, stats_path="expected_stats_latest.csv"
    )

    events = data["events"]
    current_gw = None
    next_gw = None

    for ev in events:
        if ev.get("is_current"):
            current_gw = ev["id"]
        if ev.get("is_next"):
            next_gw = ev["id"]

    if next_gw is not None:
        PLAN_GW = next_gw
        BASIS_GW = max(1, PLAN_GW - 1)
    else:
        PLAN_GW = current_gw
        BASIS_GW = max(1, PLAN_GW - 1)

    print(f"Detected current GW: {current_gw}, next GW: {next_gw}")
    print(f"Planning from GW{PLAN_GW} onwards.")
    print(
        f"Fetching your squad from GW{BASIS_GW} as the basis "
        f"(your true pre-GW{PLAN_GW} team)."
    )

    PICKS_GW = BASIS_GW
    picks_df, bank_tenths, value_tenths = fetch_manager_picks_and_bank(
        TEAM_ID, PICKS_GW
    )

    pred_start = PLAN_GW
    pred_end = PLAN_GW + NUM_GWS - 1

    global OUTPUT_ALL_CSV, OUTPUT_MYTEAM_CSV, OUTPUT_TRANSFERS_CSV, OUTPUT_DOUBLE_CSV, OUTPUT_MILP_PLAN_CSV
    OUTPUT_ALL_CSV = f"gw{pred_start}_to_gw{pred_end}_predictions_all.csv"
    OUTPUT_MYTEAM_CSV = f"gw{pred_start}_to_gw{pred_end}_predictions_myteam.csv"
    OUTPUT_TRANSFERS_CSV = (
        f"gw{pred_start}_to_gw{pred_end}_transfer_suggestions.csv"
    )
    OUTPUT_DOUBLE_CSV = (
        f"gw{pred_start}_to_gw{pred_end}_double_transfers.csv"
    )
    OUTPUT_MILP_PLAN_CSV = (
        f"gw{pred_start}_to_gw{pred_end}_multiGW_milp_plan.csv"
    )

    print(f"Predictions horizon: GW{pred_start}–GW{pred_end}")
    print("Output files will use this naming automatically.")

    preds = multi_gw_predictions(players, fixtures, PLAN_GW, NUM_GWS)

    preds_all, preds_myteam = attach_manager_view(preds, picks_df)

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
        "multi_gw_points",
        "multi_gw_points_robust",
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

    # MILP Wildcard
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
                free_transfers_subsequent=1,
                max_extra_hits=8,
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
        print(
            "\n⚠ No picks available for your GW. Could not build personalised view."
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
