"""
Serie A 2025-26 Predictor - VERSIONE V19 DIVERSIFIED STRATEGY
FEATURES:
1. Logica V14 (RandomForest + Poisson Tuned)
2. VISIBILITÀ TOTALE: Match mostrati in ogni step.
3. PORTAFOGLIO DIVERSIFICATO: Algoritmo che evita di ripetere le stesse partite.
"""

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import poisson
import itertools
import pickle
import warnings
import sys
import io
import traceback
import time

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass

print("=" * 90, flush=True)
print("[START] Serie A 2025-26 Predictor - V19 DIVERSIFIED STRATEGY", flush=True)
print("=" * 90, flush=True)

# =======================
# CONFIG
# =======================
API_KEY = "f65cdbbd6d67477883d3f468626a19cf"
COMPETITION_ID = 2019
SEASONS_TRAIN = [2024, 2023]
SEASONS_CURRENT = [2025]
PREDICT_SEASON = 2025
SEED = 42
np.random.seed(SEED)

BUDGET_TOTALE = 100.0 

RATE_LIMIT_DELAY = 5.0
API_CALL_COUNT = 0
LAST_API_CALL_TIME = None

DEBUG_MODE = True
DEBUG_LAST_MATCHDAY = 12

TOP_TEAMS = ['Inter', 'Juventus', 'Milan', 'Napoli', 'Atalanta', 'Roma', 'Lazio']
WEAK_ATTACKS = ['Lecce', 'Cagliari', 'Empoli', 'Monza', 'Venezia', 'Genoa', 'Verona', 'Udinese', 'Cremonese', 'Pisa', 'Como']

# =======================
# RATE LIMITING
# =======================
def respect_rate_limit():
    global API_CALL_COUNT, LAST_API_CALL_TIME
    current_time = time.time()
    if LAST_API_CALL_TIME is not None and (current_time - LAST_API_CALL_TIME) > 60:
        API_CALL_COUNT = 0
    if API_CALL_COUNT >= 12:
        wait_time = 60 - (current_time - LAST_API_CALL_TIME)
        if wait_time > 0:
            time.sleep(wait_time)
            API_CALL_COUNT = 0
            LAST_API_CALL_TIME = time.time()
            return
    if LAST_API_CALL_TIME is not None:
        time_since_last = current_time - LAST_API_CALL_TIME
        if time_since_last < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last)
    LAST_API_CALL_TIME = time.time()
    API_CALL_COUNT += 1

# =======================
# FETCH DATA
# =======================
def fetch_matches(season):
    respect_rate_limit()
    url = f"https://api.football-data.org/v4/competitions/{COMPETITION_ID}/matches"
    headers = {"X-Auth-Token": API_KEY}
    params = {"season": season}
    print(f"[API #{API_CALL_COUNT}] Scaricamento stagione {season}-{season+1}...", flush=True)
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
        print(f"[OK] {len(matches)} partite scaricate", flush=True)
        return matches
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return []

# =======================
# ODDS
# =======================
def fetch_odds_per_matchday(df_matches):
    print("[QUOTE] Caricamento quote per la giornata...", flush=True)
    print("-" * 80, flush=True)
    odds_list = []
    default_odds = get_default_odds()
    for idx, row in df_matches.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        print(f"  [{idx+1}/{len(df_matches)}] {home:20} vs {away:20}", end="", flush=True)
        fallback_idx = idx % len(default_odds)
        print(f" ✓ Default #{fallback_idx+1}", flush=True)
        odds_list.append(default_odds[fallback_idx])
    print(f"[OK] {len(odds_list)} set di quote caricati\n", flush=True)
    return odds_list

def get_default_odds():
    return [
        {'1': 1.63, 'X': 3.90, '2': 5.25, '1X': 1.14, '2X': 2.20, 'GG': 1.85, 'NG': 1.87},
        {'1': 2.65, 'X': 3.10, '2': 2.75, '1X': 1.43, '2X': 1.45, 'GG': 1.83, 'NG': 1.88},
        {'1': 2.05, 'X': 3.00, '2': 4.10, '1X': 1.22, '2X': 1.73, 'GG': 2.15, 'NG': 1.63},
        {'1': 1.32, 'X': 5.00, '2': 9.50, '1X': 1.04, '2X': 3.25, 'GG': 2.30, 'NG': 1.55},
        {'1': 1.60, 'X': 4.00, '2': 5.50, '1X': 1.13, '2X': 2.25, 'GG': 1.90, 'NG': 1.80},
        {'1': 3.00, 'X': 2.85, '2': 2.60, '1X': 1.45, '2X': 1.35, 'GG': 2.05, 'NG': 1.70},
        {'1': 9.00, 'X': 5.25, '2': 1.30, '1X': 3.25, '2X': 1.04, 'GG': 2.15, 'NG': 1.63},
        {'1': 1.70, 'X': 3.75, '2': 4.60, '1X': 1.17, '2X': 2.05, 'GG': 1.73, 'NG': 2.00},
        {'1': 2.55, 'X': 2.80, '2': 3.10, '1X': 1.33, '2X': 1.47, 'GG': 1.92, 'NG': 1.77},
        {'1': 1.45, 'X': 4.10, '2': 7.00, '1X': 1.07, '2X': 2.60, 'GG': 2.20, 'NG': 1.60},
    ]

def print_odds_table(df_matches, odds_list):
    print("\n" + "=" * 130, flush=True)
    print("[📊 TABELLA QUOTE DISPONIBILI]", flush=True)
    print("=" * 130, flush=True)
    header = f"{'#':>2} | {'Home':^18} | {'Away':^18} | {'1':>5} | {'X':>5} | {'2':>5} | {'GG':>5} | {'NG':>5}"
    print(header, flush=True)
    print("-" * 130, flush=True)
    for idx, row in df_matches.iterrows():
        if idx < len(odds_list):
            odds = odds_list[idx]
            print(f"{idx+1:2d} | {row['home_team']:^18} | {row['away_team']:^18} | "
                  f"{odds['1']:5.2f} | {odds['X']:5.2f} | {odds['2']:5.2f} | "
                  f"{odds['GG']:5.2f} | {odds['NG']:5.2f}", flush=True)
    print("=" * 130 + "\n", flush=True)

def export_odds_to_csv(df_matches, odds_list, filename="odds_export.csv"):
    rows = []
    for idx, row in df_matches.iterrows():
        if idx < len(odds_list):
            odds = odds_list[idx]
            rows.append({
                "matchday": int(row.get("matchday", 0)),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "quota_1": odds["1"],
                "quota_X": odds["X"],
                "quota_2": odds["2"],
                "quota_GG": odds["GG"],
                "quota_NG": odds["NG"]
            })
    df_odds = pd.DataFrame(rows)
    df_odds.to_csv(filename, index=False)
    print(f"[OK] Quote esportate in: {filename}", flush=True)

# =======================
# DATA & ELO
# =======================
def detect_last_matchday(matches):
    finished_matches = [m for m in matches if m["status"] in ["FINISHED", "LIVE"]]
    if not finished_matches: return None
    return max([m.get("matchday", 0) for m in finished_matches])

def build_match_df(seasons_train, seasons_current, current_matchday):
    print("[1] Scaricamento dati storici...", flush=True)
    rows = []
    for s in seasons_train:
        matches = fetch_matches(s)
        count = 0
        for m in matches:
            if m["status"] in ["FINISHED", "LIVE"]:
                rows.append(parse_match(m, s))
                count += 1
        print(f"    -> Stagione {s}-{s+1}: {count} partite FINISHED")
    for s in seasons_current:
        matches = fetch_matches(s)
        count = 0
        for m in matches:
            if m["status"] in ["FINISHED", "LIVE"]:
                if m.get("matchday", 0) <= current_matchday:
                    rows.append(parse_match(m, s))
                    count += 1
        print(f"    -> Stagione {s}-{s+1}: {count} partite FINISHED (giornate 1-{current_matchday})")
    df = pd.DataFrame(rows)
    if not df.empty: df["date"] = pd.to_datetime(df["date"])
    print(f"[OK] TOTALE: {len(df)} partite caricate", flush=True)
    return df

def parse_match(m, s):
    return {
        "season": s,
        "matchday": m.get("matchday", 0),
        "home_team": m["homeTeam"]["name"],
        "away_team": m["awayTeam"]["name"],
        "date": m["utcDate"],
        "home_goals": m["score"]["fullTime"]["home"],
        "away_goals": m["score"]["fullTime"]["away"],
    }

def compute_elo(df, k=20):
    teams = list(set(df["home_team"]).union(set(df["away_team"])))
    elo = {t: 1500 for t in teams}
    home_elo_list, away_elo_list = [], []
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        home_elo_list.append(elo[home])
        away_elo_list.append(elo[away])
        if hg > ag: sh, sa = 1, 0
        elif hg < ag: sh, sa = 0, 1
        else: sh, sa = 0.5, 0.5
        eh = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        ea = 1 - eh
        elo[home] += k * (sh - eh)
        elo[away] += k * (sa - ea)
    df["elo_home"] = home_elo_list
    df["elo_away"] = away_elo_list
    return df

# =======================
# STATS & HIERARCHY LOGIC
# =======================
def compute_stats_v14(df, team, idx, season):
    df_prev = df[df.index < idx]
    last5_h = df_prev[df_prev["home_team"] == team]
    last5_a = df_prev[df_prev["away_team"] == team]
    last_matches = pd.concat([last5_h, last5_a]).sort_values('date').tail(5)
    
    if len(last_matches) < 2: return 1.4, 1.3, 1.4
        
    recent_scored = 0
    recent_conceded = 0
    for _, m in last_matches.iterrows():
        if m['home_team'] == team:
            recent_scored += m['home_goals']
            recent_conceded += m['away_goals']
        else:
            recent_scored += m['away_goals']
            recent_conceded += m['home_goals']
    
    avg_rec_scored = recent_scored / len(last_matches)
    avg_rec_conceded = recent_conceded / len(last_matches)
    return avg_rec_scored, avg_rec_conceded, avg_rec_scored

def build_features_v14(df_matches):
    print("[2a] Calcolo ELO rating...", flush=True)
    df_matches = compute_elo(df_matches)
    
    X, y, seasons = [], [], []
    df_matches['stat_home_scored'] = 0.0
    df_matches['stat_home_conceded'] = 0.0
    df_matches['stat_away_scored'] = 0.0
    df_matches['stat_away_conceded'] = 0.0
    
    count = 0
    print("[2b] Calcolo Stats Gol (Pesate) e Features...", flush=True)
    
    for idx, row in df_matches.iterrows():
        if pd.isna(row.get("home_goals")) or pd.isna(row.get("away_goals")): continue
        
        h_s, h_c, h_f = compute_stats_v14(df_matches, row["home_team"], idx, row["season"])
        a_s, a_c, a_f = compute_stats_v14(df_matches, row["away_team"], idx, row["season"])
        
        df_matches.at[idx, 'stat_home_scored'] = h_s
        df_matches.at[idx, 'stat_home_conceded'] = h_c
        df_matches.at[idx, 'stat_away_scored'] = a_s
        df_matches.at[idx, 'stat_away_conceded'] = a_c
        
        feats = [
            row["elo_home"], row["elo_away"],
            h_s, h_c, a_s, a_c,
            h_f, a_f,
            row["elo_home"] - row["elo_away"]
        ]
        X.append(feats)
        
        if row["home_goals"] > row["away_goals"]: y.append(2) 
        elif row["home_goals"] < row["away_goals"]: y.append(0) 
        else: y.append(1) 
        seasons.append(row["season"])
        count += 1
        
    print(f"[OK] {count} features generate.", flush=True)
    return np.array(X), np.array(y), np.array(seasons), df_matches

def train_model(X, y):
    print("\n[3] Training Random Forest Classifier...", flush=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split = int(len(X) * 0.85)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=SEED)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[STATS] Accuracy su Test Set: {acc:.3f}", flush=True)
    return model, scaler

# =======================
# POISSON CON HIERARCHY LOGIC V14
# =======================
def is_top_team(team_name):
    for t in TOP_TEAMS:
        if t in team_name: return True
    return False

def is_weak_attack(team_name):
    for t in WEAK_ATTACKS:
        if t in team_name: return True
    return False

def calculate_poisson_metrics_v14(h_name, a_name, avg_h_scored, avg_h_conceded, avg_a_scored, avg_a_conceded):
    LEAGUE_FACTOR = 1.05
    
    lambda_home = ((avg_h_scored * 0.6 + avg_a_conceded * 0.4) + 0.1) * LEAGUE_FACTOR
    lambda_away = ((avg_a_scored * 0.6 + avg_h_conceded * 0.4) - 0.1) * LEAGUE_FACTOR
    
    if is_top_team(h_name) and is_weak_attack(a_name):
        lambda_away *= 0.50
        
    if is_top_team(a_name) and is_weak_attack(h_name):
        lambda_home *= 0.65
        
    if lambda_home < 0.3: lambda_home = 0.3
    if lambda_away < 0.2: lambda_away = 0.2
    
    p0_h = poisson.pmf(0, lambda_home)
    p0_a = poisson.pmf(0, lambda_away)
    
    p_gol_h = 1 - p0_h
    p_gol_a = 1 - p0_a
    
    prob_gg = p_gol_h * p_gol_a
    prob_ng = 1 - prob_gg
    
    return prob_gg, prob_ng, lambda_home, lambda_away

# =======================
# PORTAFOGLIO FINALE DIVERSIFICATO
# =======================
def check_overlap(schedina_a, schedina_b):
    """Restituisce True se due schedine hanno più di 1 partita in comune"""
    matches_a = set([m.split(' vs ')[0] for m in schedina_a['matches']])
    matches_b = set([m.split(' vs ')[0] for m in schedina_b['matches']])
    
    intersection = len(matches_a.intersection(matches_b))
    return intersection > 1 # Tolleranza: Max 1 partita in comune

def print_final_portfolio_diversified(all_std, all_hard, budget):
    print("\n\n")
    print("X" * 100)
    print(f"X   STRATEGIA DI INVESTIMENTO DIVERSIFICATA (BUDGET: {budget}€)   X")
    print("X" * 100)
    
    # 1. BEST SAFE (Standard)
    # Ordiniamo per score
    all_std.sort(key=lambda x: x['score'], reverse=True)
    best_safe = all_std[0] if all_std else None
    
    # 2. BEST VALUE (Hardcore) - NO OVERLAP con Safe
    best_value = None
    all_hard.sort(key=lambda x: x['score'], reverse=True)
    
    if best_safe:
        for s in all_hard:
            if not check_overlap(best_safe, s):
                best_value = s
                break
        # Se non trova nulla senza overlap, prende la migliore disponibile (fallback)
        if not best_value and all_hard:
            best_value = all_hard[0]
            print("   [INFO] Impossibile evitare sovrapposizione totale per Value Bet.")
    else:
        best_value = all_hard[0] if all_hard else None

    # 3. BEST BONUS (Longshot) - NO OVERLAP con Safe e Value
    best_bonus = None
    longshots = [s for s in all_hard if s['total_quota'] > 12.0]
    longshots.sort(key=lambda x: x['roi_percentage'], reverse=True) # Per le longshot guardiamo il ROI
    
    if longshots:
        for s in longshots:
            conflict_safe = check_overlap(best_safe, s) if best_safe else False
            conflict_value = check_overlap(best_value, s) if best_value else False
            
            if not conflict_safe and not conflict_value:
                best_bonus = s
                break
        if not best_bonus:
            best_bonus = longshots[0] # Fallback
    
    # Allocazione
    alloc_safe = budget * 0.50
    alloc_val = budget * 0.35
    alloc_bonus = budget * 0.15
    
    print("\n1. [CASSAFORTE 🛡️] SCHEDINA STANDARD (50% Budget)")
    if best_safe:
        print(f"   Investimento: {alloc_safe:.2f}€ | Possibile Vincita: {alloc_safe * best_safe['total_quota']:.2f}€")
        print(f"   Quota: {best_safe['total_quota']:.2f} | Prob: {best_safe['total_probability']:.1f}% | ROI: {best_safe['roi_percentage']:.1f}%")
        for i in range(len(best_safe['matches'])):
            print(f"   - {best_safe['matches'][i]} -> {best_safe['types'][i]}")
    else:
        print("   Nessuna schedina Safe trovata.")

    print("\n2. [VALORE 💎] SCHEDINA HARDCORE DIVERSIFICATA (35% Budget)")
    if best_value:
        print(f"   Investimento: {alloc_val:.2f}€ | Possibile Vincita: {alloc_val * best_value['total_quota']:.2f}€")
        print(f"   Quota: {best_value['total_quota']:.2f} | Prob: {best_value['total_probability']:.1f}% | ROI: {best_value['roi_percentage']:.1f}%")
        for i in range(len(best_value['matches'])):
            print(f"   - {best_value['matches'][i]} -> {best_value['types'][i]}")
    else:
        print("   Nessuna schedina Value trovata.")

    print("\n3. [COLPO GROSSO 🚀] SCHEDINA BONUS DIVERSIFICATA (15% Budget)")
    if best_bonus:
        print(f"   Investimento: {alloc_bonus:.2f}€ | Possibile Vincita: {alloc_bonus * best_bonus['total_quota']:.2f}€")
        print(f"   Quota: {best_bonus['total_quota']:.2f} | Prob: {best_bonus['total_probability']:.1f}% | ROI: {best_bonus['roi_percentage']:.1f}%")
        for i in range(len(best_bonus['matches'])):
            print(f"   - {best_bonus['matches'][i]} -> {best_bonus['types'][i]}")
    else:
        print("   Nessuna schedina Bonus trovata.")
        
    print("\n" + "="*100)

# =======================
# GENERAZIONE COMBINAZIONI
# =======================
def generate_combinations(match_options, n_matches, min_prob, min_ev):
    if n_matches > len(match_options): return []
    combos = itertools.combinations(match_options, n_matches)
    combos = itertools.islice(combos, 5000) 
    valid_schedine = []
    
    for combo in combos:
        matches_data = []
        for idx in range(len(combo)):
            best_option = combo[idx]['options'][0]
            matches_data.append({
                'match': combo[idx]['match'],
                'type': best_option['type'],
                'prob': best_option['prob'],
                'quota': best_option['quota'],
                'ev': best_option['ev']
            })
        
        total_prob = np.prod([m['prob'] for m in matches_data])
        total_ev = np.prod([m['ev'] for m in matches_data])
        total_quota = np.prod([m['quota'] for m in matches_data])
        
        if total_prob >= min_prob and total_ev > min_ev:
            # Score = Prob * EV^2
            score = total_prob * (total_ev ** 2)
            roi_raw = (total_ev - 1) * 100
            
            valid_schedine.append({
                'num_matches': n_matches,
                'matches': [m['match'] for m in matches_data],
                'types': [m['type'] for m in matches_data],
                'total_quota': float(round(total_quota, 2)),
                'total_probability': float(total_prob * 100),
                'expected_return_per_euro': float(round(total_ev, 2)),
                'roi_percentage': float(round(roi_raw, 2)),
                'score': score
            })
    valid_schedine.sort(key=lambda x: x['score'], reverse=True)
    return valid_schedine

# =======================
# BEST BETS & SCHEDINE
# =======================
def calculate_best_bets(df_next, odds_list, top_n_per_category=3):
    print("\n[SCHEDINE] Generazione Schedine (Standard + Hardcore + Mix)...", flush=True)
    print("-" * 80, flush=True)
    
    all_match_options_standard = []
    all_match_options_nodc = []
    
    for i, row in df_next.iterrows():
        q = odds_list[i] if i < len(odds_list) else odds_list[-1]
        probs_1x2 = row['probs_1x2']
        pa, pd_prob, ph = probs_1x2[0], probs_1x2[1], probs_1x2[2]
        
        h_s = row['stat_home_scored']
        h_c = row['stat_home_conceded']
        a_s = row['stat_away_scored']
        a_c = row['stat_away_conceded']
        
        p_gg, p_ng, _, _ = calculate_poisson_metrics_v14(row['home_team'], row['away_team'], h_s, h_c, a_s, a_c)
        
        def is_good_bet(prob, ev):
            if prob > 0.65 and ev > 0.85: return True 
            if prob > 0.55 and ev > 0.92: return True 
            if ev > 1.02: return True 
            return False

        raw_options = [
            {'type': '1', 'prob': ph, 'quota': q['1'], 'ev': ph * q['1']},
            {'type': 'X', 'prob': pd_prob, 'quota': q['X'], 'ev': pd_prob * q['X']},
            {'type': '2', 'prob': pa, 'quota': q['2'], 'ev': pa * q['2']},
            {'type': '1X', 'prob': ph+pd_prob, 'quota': q.get('1X', 1.05), 'ev': (ph+pd_prob)*q.get('1X', 1.05)},
            {'type': '2X', 'prob': pa+pd_prob, 'quota': q.get('2X', 1.05), 'ev': (pa+pd_prob)*q.get('2X', 1.05)},
            {'type': 'GG', 'prob': p_gg, 'quota': q['GG'], 'ev': p_gg * q['GG']},
            {'type': 'NG', 'prob': p_ng, 'quota': q['NG'], 'ev': p_ng * q['NG']},
        ]
        
        good_ops_std = [o for o in raw_options if is_good_bet(o['prob'], o['ev'])]
        if good_ops_std:
            all_match_options_standard.append({
                'match': row['home_team'] + ' vs ' + row['away_team'],
                'options': sorted(good_ops_std, key=lambda x: x['prob'], reverse=True)
            })
            
        good_ops_nodc = [o for o in good_ops_std if o['type'] not in ['1X', '2X', '12']]
        if good_ops_nodc:
            all_match_options_nodc.append({
                'match': row['home_team'] + ' vs ' + row['away_team'],
                'options': sorted(good_ops_nodc, key=lambda x: x['prob'], reverse=True)
            })

    # GENERAZIONE SCHEDINE
    portfolio_std_all = []
    portfolio_hard_all = []
    
    # Standard
    for n in [2, 3, 4, 5]:
        scheds = generate_combinations(all_match_options_standard, n, min_prob=0.15, min_ev=0.95)
        if scheds:
            portfolio_std_all.extend(scheds)
            print(f"\n[🏆 TOP STANDARD - {n} PARTITE]")
            for s in scheds[:2]: 
                print(f" - Quota: {s['total_quota']:6.2f} | Prob: {s['total_probability']:5.1f}% | ROI: {s['roi_percentage']:.1f}%")
                for j, match in enumerate(s['matches']):
                    print(f"   * {match} ({s['types'][j]})")

    # Hardcore
    for n in [2, 3, 4, 5]:
        scheds = generate_combinations(all_match_options_nodc, n, min_prob=0.08, min_ev=1.00)
        if scheds:
            portfolio_hard_all.extend(scheds)
            print(f"\n[🔥 TOP HARDCORE - {n} PARTITE]")
            for s in scheds[:2]:
                print(f" - Quota: {s['total_quota']:6.2f} | Prob: {s['total_probability']:5.1f}% | ROI: {s['roi_percentage']:.1f}%")
                for j, match in enumerate(s['matches']):
                    print(f"   * {match} ({s['types'][j]})")

    # CALL FINAL DIVERSIFIED STRATEGY
    print_final_portfolio_diversified(portfolio_std_all, portfolio_hard_all, BUDGET_TOTALE)
    
    # CSV Export
    df_export = pd.DataFrame(portfolio_std_all + portfolio_hard_all)
    df_export.to_csv(f"schedine_generate_final.csv", index=False)
    print(f"[OK] Tutte le schedine salvate in CSV.")

# =======================
# MAIN
# =======================
try:
    print("\n")
    print("[0] Rilevamento ultima giornata giocata...", flush=True)
    matches_all = fetch_matches(PREDICT_SEASON)
    last_matchday = detect_last_matchday(matches_all)
    if DEBUG_MODE and DEBUG_LAST_MATCHDAY:
        print(f"[DEBUG] Forzatura ultima giornata a {DEBUG_LAST_MATCHDAY}")
        last_matchday = DEBUG_LAST_MATCHDAY
    next_matchday = last_matchday + 1
    print(f"[OK] Training fino a G{last_matchday}, Previsione G{next_matchday}")
    
    df_history = build_match_df(SEASONS_TRAIN, SEASONS_CURRENT, last_matchday)
    X, y, _, df_history = build_features_v14(df_history)
    model, scaler = train_model(X, y)
    
    print(f"\n[4] Analisi Giornata {next_matchday}...")
    future = [m for m in matches_all if m.get('matchday') == next_matchday]
    if not future: future = [m for m in matches_all if m['status'] == 'SCHEDULED'][:10]
    
    df_next = pd.DataFrame([parse_match(m, PREDICT_SEASON) for m in future])
    X_next_list = []
    
    print("\n" + "=" * 100, flush=True)
    print(f"[MATCHDAY RESULTS - GIORNATA {next_matchday} STAGIONE 2025-26]", flush=True)
    print("=" * 100 + "\n", flush=True)
    
    for i, row in df_next.iterrows():
        h_s, h_c, h_f = compute_stats_v14(df_history, row['home_team'], len(df_history)+1, PREDICT_SEASON)
        a_s, a_c, a_f = compute_stats_v14(df_history, row['away_team'], len(df_history)+1, PREDICT_SEASON)
        df_next.at[i, 'stat_home_scored'] = h_s
        df_next.at[i, 'stat_home_conceded'] = h_c
        df_next.at[i, 'stat_away_scored'] = a_s
        df_next.at[i, 'stat_away_conceded'] = a_c
        
        last_h = df_history[df_history['home_team']==row['home_team']].tail(1)
        elo_h = last_h['elo_home'].values[0] if not last_h.empty else 1500
        last_a = df_history[df_history['away_team']==row['away_team']].tail(1)
        elo_a = last_a['elo_away'].values[0] if not last_a.empty else 1500
        feat = [elo_h, elo_a, h_s, h_c, a_s, a_c, h_f, a_f, elo_h-elo_a]
        X_next_list.append(feat)
    
    X_next_arr = np.array(X_next_list)
    X_next_sc = scaler.transform(X_next_arr)
    probs = model.predict_proba(X_next_sc) 
    df_next['probs_1x2'] = list(probs)
    
    for i, row in df_next.iterrows():
        pr = row['probs_1x2']
        pa, pd_prob, ph = pr[0], pr[1], pr[2]
        if ph > pa and ph > pd_prob: res = "1"
        elif pa > ph and pa > pd_prob: res = "2"
        else: res = "X"
        
        h_s = row['stat_home_scored']
        h_c = row['stat_home_conceded']
        a_s = row['stat_away_scored']
        a_c = row['stat_away_conceded']
        _, p_ng, lambda_h, lambda_a = calculate_poisson_metrics_v14(row['home_team'], row['away_team'], h_s, h_c, a_s, a_c)
        score_pred = f"{int(round(lambda_h))}-{int(round(lambda_a))}"
        
        print(f"{i+1}. {row['home_team']:15} vs {row['away_team']:15}", flush=True)
        print(f"   Pred: {res} ({score_pred}) | 1:{ph*100:.0f}% X:{pd_prob*100:.0f}% 2:{pa*100:.0f}% | NG Prob: {p_ng*100:.0f}%\n", flush=True)
    
    print("=" * 100, flush=True)
    odds_list = fetch_odds_per_matchday(df_next)
    print_odds_table(df_next, odds_list)
    export_odds_to_csv(df_next, odds_list, f"odds_giornata_{next_matchday}.csv")
    calculate_best_bets(df_next, odds_list)
    
    print("\n[SUCCESS] Esecuzione V19 DIVERSIFIED Completata.")
except Exception as e:
    print(f"\n[FATAL ERROR] {e}")
    traceback.print_exc()