"""
Serie A 2025-26 Predictor - VERSIONE V7 ULTIMATE
AUTO + DEBUG MODE + QUOTE SNAI REALI + SCHEDINE INTELLIGENTI + MONEY MANAGEMENT
Con ELO Rating + Forma Recente + Feature Scaling + Cross-Validation
TOP 3 SICURE + TOP CONSIGLIATE PER CATEGORIA + GESTIONE BUDGET
"""

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
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

print("=" * 80, flush=True)
print("[START] Serie A 2025-26 Predictor - FINAL VERSION V7 ULTIMATE", flush=True)
print("=" * 80, flush=True)

# =======================
# CONFIG
# =======================
API_KEY = "f65cdbbd6d67477883d3f468626a19cf"
COMPETITION_ID = 2019
SEASONS_TRAIN = [2024, 2023]
SEASONS_CURRENT = [2025]
SEASON_WEIGHTS = {2025: 0.8, 2024: 0.15, 2023: 0.05}
PREDICT_SEASON = 2025
SEED = 42
np.random.seed(SEED)

# CONFIGURAZIONE BUDGET
BUDGET_TOTALE = 100.0  # Euro da investire

RATE_LIMIT_DELAY = 5.0
API_CALL_COUNT = 0
LAST_API_CALL_TIME = None

DEBUG_MODE = True
DEBUG_LAST_MATCHDAY = 12

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
            print(f"[WAIT] Aspetto {wait_time:.1f}s per rate limit", flush=True)
            time.sleep(wait_time)
            API_CALL_COUNT = 0
            LAST_API_CALL_TIME = time.time()
            return
    
    if LAST_API_CALL_TIME is not None:
        time_since_last = current_time - LAST_API_CALL_TIME
        if time_since_last < RATE_LIMIT_DELAY:
            sleep_time = RATE_LIMIT_DELAY - time_since_last
            time.sleep(sleep_time)
    
    LAST_API_CALL_TIME = time.time()
    API_CALL_COUNT += 1

# =======================
# FETCH MATCH DATA
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
# ODDS MANAGEMENT
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
    """
    Quote REALI da SNAI - Stagione 2025-26 Serie A Giornata 13
    """
    return [
        # 1. Como vs Sassuolo (28/11 20:45)
        {'1': 1.63, 'X': 3.90, '2': 5.25, '1X': 1.14, '2X': 2.20, 'GG': 1.85, 'NG': 1.87},
        # 2. Parma vs Udinese (29/11 15:00)
        {'1': 2.65, 'X': 3.10, '2': 2.75, '1X': 1.43, '2X': 1.45, 'GG': 1.83, 'NG': 1.88},
        # 3. Genoa vs Verona (29/11 15:00)
        {'1': 2.05, 'X': 3.00, '2': 4.10, '1X': 1.22, '2X': 1.73, 'GG': 2.15, 'NG': 1.63},
        # 4. Juventus vs Cagliari (29/11 18:00)
        {'1': 1.32, 'X': 5.00, '2': 9.50, '1X': 1.04, '2X': 3.25, 'GG': 2.30, 'NG': 1.55},
        # 5. Milan vs Lazio (29/11 20:45)
        {'1': 1.60, 'X': 4.00, '2': 5.50, '1X': 1.13, '2X': 2.25, 'GG': 1.90, 'NG': 1.80},
        # 6. Lecce vs Torino (30/11 12:30)
        {'1': 3.00, 'X': 2.85, '2': 2.60, '1X': 1.45, '2X': 1.35, 'GG': 2.05, 'NG': 1.70},
        # 7. Pisa vs Inter (30/11 15:00)
        {'1': 9.00, 'X': 5.25, '2': 1.30, '1X': 3.25, '2X': 1.04, 'GG': 2.15, 'NG': 1.63},
        # 8. Atalanta vs Fiorentina (30/11 18:00)
        {'1': 1.70, 'X': 3.75, '2': 4.60, '1X': 1.17, '2X': 2.05, 'GG': 1.73, 'NG': 2.00},
        # 9. Roma vs Napoli (30/11 20:45)
        {'1': 2.55, 'X': 2.80, '2': 3.10, '1X': 1.33, '2X': 1.47, 'GG': 1.92, 'NG': 1.77},
        # 10. Bologna vs Cremonese (01/12 20:45)
        {'1': 1.45, 'X': 4.10, '2': 7.00, '1X': 1.07, '2X': 2.60, 'GG': 2.20, 'NG': 1.60},
    ]

def print_odds_table(df_matches, odds_list):
    print("\n" + "=" * 160, flush=True)
    print("[📊 TABELLA QUOTE COMPLETE - GIORNATA]", flush=True)
    print("=" * 160, flush=True)
    print()
    
    header = f"{'#':>2} | {'Home':^18} | {'Away':^18} | {'1':>5} | {'X':>5} | {'2':>5} | {'1X':>5} | {'2X':>5} | {'GG':>5} | {'NG':>5}"
    print(header, flush=True)
    print("-" * 160, flush=True)
    
    for idx, row in df_matches.iterrows():
        if idx < len(odds_list):
            odds = odds_list[idx]
            print(f"{idx+1:2d} | {row['home_team']:^18} | {row['away_team']:^18} | "
                  f"{odds['1']:5.2f} | {odds['X']:5.2f} | {odds['2']:5.2f} | "
                  f"{odds.get('1X', 0):5.2f} | {odds.get('2X', 0):5.2f} | "
                  f"{odds['GG']:5.2f} | {odds['NG']:5.2f}", flush=True)
    
    print()
    print("=" * 160, flush=True)
    print()

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
                "quota_1X": odds.get("1X", 0),
                "quota_2X": odds.get("2X", 0),
                "quota_GG": odds["GG"],
                "quota_NG": odds["NG"]
            })
    
    df_odds = pd.DataFrame(rows)
    df_odds.to_csv(filename, index=False)
    print(f"[OK] Quote esportate in: {filename}", flush=True)
    print()

# =======================
# DATA LOADING
# =======================
def detect_last_matchday(matches):
    finished_matches = [m for m in matches if m["status"] in ["FINISHED", "LIVE"]]
    if not finished_matches:
        return None
    last_matchday = max([m.get("matchday", 0) for m in finished_matches])
    return last_matchday

def build_match_df(seasons_train, seasons_current, current_matchday):
    print("[1] Scaricamento dati storici...", flush=True)
    print("-" * 80, flush=True)
    
    rows = []
    
    print("[1a] Stagioni di TRAINING (tutte le giornate)...", flush=True)
    for s in seasons_train:
        matches = fetch_matches(s)
        count = 0
        for m in matches:
            if m["status"] in ["FINISHED", "LIVE"]:
                rows.append({
                    "season": s,
                    "matchday": m.get("matchday", 0),
                    "home_team": m["homeTeam"]["name"],
                    "away_team": m["awayTeam"]["name"],
                    "date": m["utcDate"],
                    "home_goals": m["score"]["fullTime"]["home"],
                    "away_goals": m["score"]["fullTime"]["away"],
                })
                count += 1
        print(f"    -> Stagione {s}-{s+1}: {count} partite FINISHED")
    
    print(f"[1b] Stagione CORRENTE (fino giornata {current_matchday})...", flush=True)
    for s in seasons_current:
        matches = fetch_matches(s)
        count = 0
        for m in matches:
            if m["status"] in ["FINISHED", "LIVE"]:
                matchday = m.get("matchday", 0)
                if matchday <= current_matchday:
                    rows.append({
                        "season": s,
                        "matchday": matchday,
                        "home_team": m["homeTeam"]["name"],
                        "away_team": m["awayTeam"]["name"],
                        "date": m["utcDate"],
                        "home_goals": m["score"]["fullTime"]["home"],
                        "away_goals": m["score"]["fullTime"]["away"],
                    })
                    count += 1
        print(f"    -> Stagione {s}-{s+1}: {count} partite FINISHED (giornate 1-{current_matchday})")
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    
    print(f"[OK] TOTALE: {len(df)} partite caricate", flush=True)
    return df

# =======================
# ELO RATING SYSTEM
# =======================
def compute_elo(df, k=20):
    teams = list(set(df["home_team"]).union(set(df["away_team"])))
    elo = {t: 1500 for t in teams}
    
    home_elo_list, away_elo_list = [], []
    
    for _, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        
        home_elo_list.append(elo[home])
        away_elo_list.append(elo[away])
        
        if hg > ag:
            score_h, score_a = 1, 0
        elif hg < ag:
            score_h, score_a = 0, 1
        else:
            score_h, score_a = 0.5, 0.5
        
        exp_h = 1 / (1 + 10 ** ((elo[away] - elo[home]) / 400))
        exp_a = 1 - exp_h
        
        elo[home] += k * (score_h - exp_h)
        elo[away] += k * (score_a - exp_a)
    
    df["elo_home"] = home_elo_list
    df["elo_away"] = away_elo_list
    return df, elo

# =======================
# RECENT FORM
# =======================
def add_recent_form(df, window=5):
    teams = set(df["home_team"]).union(set(df["away_team"]))
    
    df["home_pts5"] = 0
    df["away_pts5"] = 0
    df["home_gf5"] = 0
    df["home_ga5"] = 0
    df["away_gf5"] = 0
    df["away_ga5"] = 0
    
    match_lists = {t: [] for t in teams}
    
    for i, row in df.iterrows():
        home, away = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        
        df.loc[i, "home_gf5"] = sum([m[0] for m in match_lists[home][-window:]])
        df.loc[i, "home_ga5"] = sum([m[1] for m in match_lists[home][-window:]])
        df.loc[i, "away_gf5"] = sum([m[0] for m in match_lists[away][-window:]])
        df.loc[i, "away_ga5"] = sum([m[1] for m in match_lists[away][-window:]])
        
        df.loc[i, "home_pts5"] = sum(m[2] for m in match_lists[home][-window:])
        df.loc[i, "away_pts5"] = sum(m[2] for m in match_lists[away][-window:])
        
        pts_home = 3 if hg > ag else 1 if hg == ag else 0
        pts_away = 3 if ag > hg else 1 if hg == ag else 0
        
        match_lists[home].append((hg, ag, pts_home))
        match_lists[away].append((ag, hg, pts_away))
    
    return df

# =======================
# FEATURE ENGINEERING V2
# =======================
def compute_stats_v2(df, team, idx, season):
    df_s = df[(df["season"] == season) & (df.index < idx)]
    
    home = df_s[df_s["home_team"] == team]
    away = df_s[df_s["away_team"] == team]
    
    last5 = pd.concat([
        home[["home_goals"]].rename(columns={"home_goals": "g"}),
        away[["away_goals"]].rename(columns={"away_goals": "g"})
    ]).tail(5)
    
    return {
        "home_g_avg": home["home_goals"].mean() if len(home) > 0 else 0.5,
        "away_g_avg": away["away_goals"].mean() if len(away) > 0 else 0.5,
        "recent_form": last5["g"].mean() if len(last5) > 0 else 0.5,
        "matches": len(home) + len(away)
    }

def build_features_v2(df_matches):
    print("[2a] Calcolo ELO rating...", flush=True)
    df_matches, elo_dict = compute_elo(df_matches)
    
    print("[2b] Calcolo forma recente...", flush=True)
    df_matches = add_recent_form(df_matches)
    
    X, y, seasons = [], [], []
    
    count = 0
    for idx, row in df_matches.iterrows():
        if pd.isna(row.get("home_goals")) or pd.isna(row.get("away_goals")):
            continue
        
        hs = compute_stats_v2(df_matches, row["home_team"], idx, row["season"])
        as_ = compute_stats_v2(df_matches, row["away_team"], idx, row["season"])
        
        if hs["matches"] < 3 or as_["matches"] < 3:
            continue
        
        feats = [
            row["elo_home"], row["elo_away"],
            row["home_pts5"], row["away_pts5"],
            row["home_gf5"], row["home_ga5"],
            row["away_gf5"], row["away_ga5"],
            hs["home_g_avg"], as_["home_g_avg"],
            hs["away_g_avg"], as_["away_g_avg"],
            hs["recent_form"], as_["recent_form"],
            row["home_goals"] - row["away_goals"],
            abs(row["home_goals"] - row["away_goals"])
        ]
        
        X.append(feats)
        
        if row["home_goals"] > row["away_goals"]:
            y.append(1)
        elif row["home_goals"] < row["away_goals"]:
            y.append(-1)
        else:
            y.append(0)
        
        seasons.append(row["season"])
        count += 1
    
    print(f"[OK] {count} partite con features complete ({len(X)} features)", flush=True)
    return np.array(X), np.array(y), np.array(seasons), df_matches

# =======================
# SCALER + MODEL TRAINING
# =======================
def train_model_v2(X, y, seasons):
    print("\n[3] Training Ridge model con GridSearch...", flush=True)
    print("-" * 80, flush=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    sample_w = np.array([SEASON_WEIGHTS.get(int(s), 0.1) for s in seasons])
    split = int(len(X) * 0.8)
    
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    sw_train = sample_w[:split]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)
    
    param_grid = {'alpha': [0.10, 0.12, 0.15, 0.18, 0.20]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    
    print(f"[INFO] GridSearch con 5-fold CV...", flush=True)
    grid.fit(X_train, y_train, sample_weight=sw_train)
    
    best_alpha = grid.best_params_['alpha']
    print(f"[OK] Best alpha trovato: {best_alpha}", flush=True)
    
    model = grid.best_estimator_
    
    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    mae_test = mean_absolute_error(y_test, model.predict(X_test))
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    
    print(f"MAE train: {mae_train:.3f}", flush=True)
    print(f"MAE test : {mae_test:.3f}", flush=True)
    print(f"R2 train : {r2_train:.3f}", flush=True)
    print(f"R2 test  : {r2_test:.3f}", flush=True)
    
    return model, scaler, best_alpha

# =======================
# PREDICTION HELPERS
# =======================
def get_probs(pred):
    ph = np.clip((pred + 1) / 2, 0.2, 0.8)
    p_d = (1 - abs(pred)) * 0.28
    pa = 1 - ph - p_d
    return ph, p_d, pa

def label(pred):
    if pred > 0.18:
        return "1"
    if pred < -0.18:
        return "2"
    return "X"

# ==============================================================================
# 💰 NUOVO MODULO: MONEY MANAGEMENT
# ==============================================================================
def apply_money_management(schedine_selezionate, budget_totale=100):
    """
    Calcola lo stake ideale per ogni schedina basandosi sullo SCORE (EV + Prob).
    Usa un sistema a pesi ponderati normalizzati sul budget totale.
    """
    print(f"\n{'='*100}")
    print(f" [💰 MONEY MANAGEMENT STRATEGY] BUDGET: {budget_totale}€ ")
    print(f"{'='*100}")

    schedine_con_stake = []
    somma_pesi = 0
    
    # 1. Assegno un 'peso' a ogni schedina basato sul suo SCORE
    # Lo SCORE calcolato in precedenza già combina Sicurezza ed EV.
    # Eleviamo lo score al quadrato per premiare maggiormente le schedine migliori.
    
    for schedina in schedine_selezionate:
        score = schedina.get('score', 0.5)
        # Se è una schedina ad alto rischio (4-5 partite), riduco leggermente il peso
        num = schedina['num_matches']
        fattore_rischio = 1.0
        if num == 4: fattore_rischio = 0.8
        if num == 5: fattore_rischio = 0.6
        
        peso_finale = (score ** 2) * fattore_rischio
        
        schedina['peso_stake'] = peso_finale
        somma_pesi += peso_finale
        schedine_con_stake.append(schedina)

    # 2. Distribuzione Budget
    print(f"{'TIPO SCHEDINA':<35} | {'QUOTA':<8} | {'PROB':<7} | {'EV':<6} | {'% BUDGET':<10} | {'PUNTATA €':<10}")
    print("-" * 100)

    for schedina in schedine_con_stake:
        if somma_pesi > 0:
            percentuale_reale = (schedina['peso_stake'] / somma_pesi) 
            puntata_euro = percentuale_reale * budget_totale
        else:
            percentuale_reale = 0
            puntata_euro = 0
            
        puntata_euro = round(puntata_euro, 2)
        perc_str = f"{round(percentuale_reale * 100, 1)}%"
        
        nome_display = f"{schedina['num_matches']} Partite (Score {schedina['score']:.2f})"
        
        print(f"{nome_display:<35} | {schedina['total_quota']:<8.2f} | {schedina['total_probability']:<6.1f}% | {schedina['expected_return_per_euro']:<6.2f} | {perc_str:<10} | {puntata_euro}€")

    print("-" * 100)
    print("Nota: La strategia 'Pesata' punta di più sulle schedine a basso rischio e alto EV.")
    print(f"{'='*100}\n")

# =======================
# BEST BETS CALCULATION (MODIFICATA)
# =======================
def calculate_best_bets(df_matches, odds_list, top_n_per_category=3):
    print("\n[SCHEDINE] Calcolo schedine ottimizzate per rischio/reward...", flush=True)
    print("-" * 80, flush=True)
    print(f"[INFO] Usando {len(odds_list)} set di quote", flush=True)
    
    match_options = []
    
    for i in range(len(df_matches)):
        if i >= len(odds_list):
            q = odds_list[-1]
        else:
            q = odds_list[i]
        
        row = df_matches.iloc[i]
        ph, p_d, pa = get_probs(row['prediction'])
        
        options = [
            {'type': '1', 'prob': ph, 'quota': q['1'], 'ev': ph * q['1']},
            {'type': 'X', 'prob': p_d, 'quota': q['X'], 'ev': p_d * q['X']},
            {'type': '2', 'prob': pa, 'quota': q['2'], 'ev': pa * q['2']},
            {'type': '1X', 'prob': ph + p_d, 'quota': q.get('1X', 1.70), 'ev': (ph + p_d) * q.get('1X', 1.70)},
            {'type': '2X', 'prob': pa + p_d, 'quota': q.get('2X', 2.30), 'ev': (pa + p_d) * q.get('2X', 2.30)},
            {'type': 'GG', 'prob': ph + pa, 'quota': q['GG'], 'ev': (ph + pa) * q['GG']},
            {'type': 'NG', 'prob': 1 - (ph + pa), 'quota': q['NG'], 'ev': (1 - (ph + pa)) * q['NG']},
        ]
        
        good_options = [o for o in options if o['ev'] > 1.0]
        
        if good_options:
            match_options.append({
                'match': row['home_team'] + ' vs ' + row['away_team'],
                'options': sorted(good_options, key=lambda x: x['ev'], reverse=True)
            })
    
    schedules_by_count = {2: [], 3: [], 4: [], 5: []}
    all_schedules_flat = []
    
    # Generazione combinazioni
    for n in range(2, 6):
        if n > len(match_options):
            continue
        
        # Limita le combinazioni per evitare esplosioni combinatorie se troppe opzioni
        iter_limit = 5000 
        count_iter = 0
        
        for combo in itertools.combinations(range(len(match_options)), n):
            count_iter += 1
            if count_iter > iter_limit: break
            
            matches_data = []
            for idx in combo:
                best_option = match_options[idx]['options'][0]
                matches_data.append({
                    'match': match_options[idx]['match'],
                    'type': best_option['type'],
                    'prob': best_option['prob'],
                    'quota': best_option['quota'],
                    'ev': best_option['ev']
                })
            
            total_quota = np.prod([m['quota'] for m in matches_data])
            total_prob = np.prod([m['prob'] for m in matches_data])
            total_ev = np.prod([m['ev'] for m in matches_data])
            
            if total_prob >= 0.15 and total_ev >= 1.0:
                sched = {
                    'matches': [m['match'] for m in matches_data],
                    'types': [m['type'] for m in matches_data],
                    'num_matches': n,
                    'total_quota': float(round(total_quota, 2)),
                    'total_probability': float(total_prob * 100),
                    'expected_return_per_euro': float(round(total_ev, 2)),
                    'roi_percentage': float(round((total_ev - 1) * 100, 2))
                }
                schedules_by_count[n].append(sched)
                all_schedules_flat.append(sched)
    
    all_top_schedules = []
    portfolio_best_picks = [] # Lista per il Money Management
    
    for num_partite in [2, 3, 4, 5]:
        scheds = schedules_by_count[num_partite]
        
        if not scheds:
            continue
        
        # CALCOLA SCORE
        scored_scheds = []
        for sched in scheds:
            prob_score = (sched['total_probability'] / 100) * 0.6 
            ev_score = min(sched['expected_return_per_euro'], 2.0) / 2.0 * 0.4
            combined_score = prob_score + ev_score
            
            scored_scheds.append({
                **sched,
                'score': combined_score
            })
        
        # ORDINA
        scored_scheds.sort(key=lambda x: x['score'], reverse=True)
        
        # Salva la migliore per il Money Management
        if len(scored_scheds) > 0:
            portfolio_best_picks.append(scored_scheds[0])
        
        print(f"\n{'=' * 100}", flush=True)
        print(f"[🏆 TOP {min(top_n_per_category, len(scored_scheds))} SCHEDINE CONSIGLIATE CON {num_partite} PARTITE]", flush=True)
        print(f"[Ordinate per Sicurezza (Prob + EV bilanciati)]", flush=True)
        print(f"{'=' * 100}\n", flush=True)
        
        for i, sched in enumerate(scored_scheds[:top_n_per_category]):
            print(f"{i+1}. SCORE: {sched['score']:.2f} | EV: {sched['expected_return_per_euro']:.2f}x | Prob: {sched['total_probability']:5.1f}% | ROI: {sched['roi_percentage']:6.1f}% | Quota: {sched['total_quota']:6.2f}", flush=True)
            for j, match in enumerate(sched['matches']):
                print(f"   {j+1}) {match} ({sched['types'][j]})", flush=True)
            print()
            all_top_schedules.append(sched)
    
    # -----------------------------------------------------------
    # CHIAMATA AL NUOVO MONEY MANAGEMENT
    # -----------------------------------------------------------
    if portfolio_best_picks:
        apply_money_management(portfolio_best_picks, budget_totale=BUDGET_TOTALE)
    # -----------------------------------------------------------

    print(f"\n{'=' * 100}", flush=True)
    print(f"[🔒 TOP 10 SCHEDINE PIÙ SICURE - Miglior equilibrio Probabilità/ROI]", flush=True)
    print(f"{'=' * 100}\n", flush=True)
    
    safe_schedules = []
    for sched in all_schedules_flat:
        prob_score = (sched['total_probability'] / 100) * 0.7
        roi_score = min(sched['roi_percentage'] / 50, 1.0) * 0.3
        safety_score = prob_score + roi_score
        
        safe_schedules.append({
            **sched,
            'safety_score': safety_score
        })
    
    safe_schedules.sort(key=lambda x: x['safety_score'], reverse=True)
    
    for i, sched in enumerate(safe_schedules[:10]):
        print(f"{i+1}. SICUREZZA: {sched['safety_score']:.2f} | ROI: {sched['roi_percentage']:6.1f}% | Prob: {sched['total_probability']:5.1f}% | Quota: {sched['total_quota']:6.2f}", flush=True)
        for j, match in enumerate(sched['matches']):
            print(f"   {j+1}) {match} ({sched['types'][j]})", flush=True)
        print()
    
    csv_file = f"schedine_migliori_giornata_{len(df_matches)}.csv"
    export_data = []
    for i, sched in enumerate(all_top_schedules):
        export_data.append({
            'posizione': i + 1,
            'num_partite': int(sched['num_matches']),
            'roi_percentage': float(sched['roi_percentage']),
            'probabilita_totale': float(sched['total_probability']),
            'quota_totale': float(sched['total_quota']),
            'partite': ' | '.join(sched['matches']),
            'tipi_scommessa': ' | '.join(sched['types']),
            'expected_return_per_euro': float(sched['expected_return_per_euro']),
            'score': float(sched.get('score', 0))
        })
    
    df_export = pd.DataFrame(export_data)
    df_export.to_csv(csv_file, index=False)
    print(f"\n[OK] Tutte le top schedine salvate in: {csv_file}\n", flush=True)

# =======================
# MAIN
# =======================
try:
    print("\n")
    
    print("[0] Rilevamento ultima giornata giocata...", flush=True)
    print("-" * 80, flush=True)
    
    current_season_matches = fetch_matches(PREDICT_SEASON)
    last_matchday = detect_last_matchday(current_season_matches)
    
    if DEBUG_MODE:
        if DEBUG_LAST_MATCHDAY is None:
            print(f"[DEBUG] Modalità DEBUG con INPUT MANUALE", flush=True)
            print(f"[INPUT] Quale giornata vuoi USARE per training? (1-37): ", end="", flush=True)
            try:
                user_input = input().strip()
                last_matchday = int(user_input)
                if last_matchday < 1 or last_matchday > 37:
                    print(f"[ERROR] Giornata non valida!", flush=True)
                    exit(1)
                print(f"[OK] Giornata {last_matchday} selezionata per training", flush=True)
            except ValueError:
                print(f"[ERROR] Inserire un numero valido!", flush=True)
                exit(1)
        else:
            print(f"[DEBUG] Modalità DEBUG attiva - Training fino giornata {DEBUG_LAST_MATCHDAY}", flush=True)
            last_matchday = DEBUG_LAST_MATCHDAY
    
    if last_matchday is None:
        print("[ERROR] Nessuna partita giocata trovata!")
        exit(1)
    
    next_matchday = last_matchday + 1
    print(f"[OK] Training con dati fino a: Giornata {last_matchday}", flush=True)
    print(f"[OK] Predicendo PROSSIMA giornata: {next_matchday}", flush=True)
    
    df_matches = build_match_df(SEASONS_TRAIN, SEASONS_CURRENT, last_matchday)
    
    if len(df_matches) == 0:
        print("[ERROR] Nessuna partita scaricata!")
        exit(1)
    
    print("\n[INFO] Distribuzione dati per stagione:")
    for s in sorted(df_matches['season'].unique()):
        count = len(df_matches[df_matches['season'] == s])
        print(f"    Stagione {s}-{s+1}: {count} partite")
    
    print("\n[2] Preparazione features...", flush=True)
    print("-" * 80, flush=True)
    X, y, seasons, df_matches = build_features_v2(df_matches)
    
    if len(X) == 0:
        print("[ERROR] Nessuna feature disponibile!")
        exit(1)
    
    model, scaler, best_alpha = train_model_v2(X, y, seasons)
    
    print(f"\n[4] Scaricamento matchday {next_matchday} ({PREDICT_SEASON}-{PREDICT_SEASON+1})...", flush=True)
    print("-" * 80, flush=True)
    
    future_matches = fetch_matches(PREDICT_SEASON)
    future_matches = [m for m in future_matches if m.get("matchday") == next_matchday]
    
    if len(future_matches) == 0:
        print(f"[INFO] Nessuna partita trovata con matchday {next_matchday}", flush=True)
        print(f"[INFO] Ricerca per data prossima...", flush=True)
        
        all_matches = fetch_matches(PREDICT_SEASON)
        unplayed = [m for m in all_matches if m["status"] in ["SCHEDULED", "TOBE_PLAYED"]]
        
        if unplayed:
            unplayed_sorted = sorted(unplayed, key=lambda x: x.get("utcDate", ""), reverse=False)
            
            if unplayed_sorted:
                first_date = unplayed_sorted[0].get("utcDate", "")
                future_matches = [m for m in unplayed_sorted if m.get("utcDate", "") == first_date][:10]
                
                print(f"[OK] Trovate {len(future_matches)} partite per la prossima data: {first_date}", flush=True)
    
    print(f"[OK] {len(future_matches)} partite trovate", flush=True)
    
    if len(future_matches) == 0:
        print(f"[WARNING] Nessuna partita trovata per giornata {next_matchday}!", flush=True)
        exit(1)
    
    df_next = pd.DataFrame([{
        "season": PREDICT_SEASON,
        "matchday": m.get("matchday", next_matchday),
        "home_team": m["homeTeam"]["name"],
        "away_team": m["awayTeam"]["name"],
        "home_goals": m["score"]["fullTime"]["home"] if m["status"] in ["FINISHED", "LIVE"] else None,
        "away_goals": m["score"]["fullTime"]["away"] if m["status"] in ["FINISHED", "LIVE"] else None,
        "status": m["status"]
    } for m in future_matches])
    
    print(f"[OK] DataFrame creato: {df_next.shape}", flush=True)
    
    print("\n[5] Generazione predizioni...", flush=True)
    print("-" * 80, flush=True)
    
    X_next_list = []
    for idx, row in df_next.iterrows():
        hs = compute_stats_v2(df_matches, row["home_team"], len(df_matches), PREDICT_SEASON)
        as_ = compute_stats_v2(df_matches, row["away_team"], len(df_matches), PREDICT_SEASON)
        
        if hs["matches"] < 1 or as_["matches"] < 1:
            hs = {"home_g_avg": 1.5, "away_g_avg": 1.2, "recent_form": 1.3, "matches": 0}
            as_ = {"home_g_avg": 1.5, "away_g_avg": 1.2, "recent_form": 1.3, "matches": 0}
        
        home_elo = 1500
        away_elo = 1500
        home_pts5 = away_pts5 = 0
        home_gf5 = home_ga5 = away_gf5 = away_ga5 = 0
        
        feats = [
            home_elo, away_elo,
            home_pts5, away_pts5,
            home_gf5, home_ga5,
            away_gf5, away_ga5,
            hs["home_g_avg"], as_["home_g_avg"],
            hs["away_g_avg"], as_["away_g_avg"],
            hs["recent_form"], as_["recent_form"],
            0, 0
        ]
        X_next_list.append(feats)
    
    X_next = np.array(X_next_list)
    X_next_scaled = scaler.transform(X_next)
    print(f"[OK] Features create: {X_next.shape}", flush=True)
    
    preds = model.predict(X_next_scaled)
    
    df_next["prediction"] = preds
    df_next["rounded"] = np.round(preds, 2)
    df_next["result"] = df_next["prediction"].apply(label)
    
    print("\n" + "=" * 100, flush=True)
    print(f"[MATCHDAY RESULTS - GIORNATA {next_matchday} STAGIONE 2025-26]", flush=True)
    print("=" * 100 + "\n", flush=True)
    
    for idx, row in df_next.iterrows():
        ph, p_d, pa = get_probs(row["prediction"])
        print(f"{idx+1}. {row['home_team']:15} vs {row['away_team']:15}", flush=True)
        if row['status'] == 'FINISHED':
            print(f"   Risultato: {int(row['home_goals'])}-{int(row['away_goals'])}", flush=True)
        print(f"   Pred: {row['rounded']:6.2f} -> {row['result']:3} | Casa {ph*100:5.1f}% Pareggio {p_d*100:5.1f}% Ospiti {pa*100:5.1f}%\n", flush=True)
    
    print("=" * 100, flush=True)
    
    print("\n[QUOTE] Caricamento quote...", flush=True)
    print("-" * 80, flush=True)
    
    odds_list = fetch_odds_per_matchday(df_next)
    print(f"[OK] {len(odds_list)} quote caricate\n", flush=True)
    
    print_odds_table(df_next, odds_list)
    export_odds_to_csv(df_next, odds_list, f"odds_giornata_{next_matchday}.csv")
    
    calculate_best_bets(df_next, odds_list, top_n_per_category=3)
    
    print("[6] Salvataggio risultati...", flush=True)
    print("-" * 80, flush=True)
    
    with open("model_seria_a_25_26_v6.pkl", "wb") as f:
        pickle.dump(model, f)
    print("[OK] Modello salvato: model_seria_a_25_26_v6.pkl", flush=True)
    
    with open("scaler_seria_a_25_26_v6.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("[OK] Scaler salvato: scaler_seria_a_25_26_v6.pkl", flush=True)
    
    df_next.to_csv(f"predictions_giornata_{next_matchday}.csv", index=False)
    print(f"[OK] Predizioni salvate: predictions_giornata_{next_matchday}.csv", flush=True)
    
    df_matches.to_csv("historical_data_25_26_v6.csv", index=False)
    print("[OK] Dati storici salvati: historical_data_25_26_v6.csv", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print("[SUCCESS] Esecuzione completata!", flush=True)
    print("=" * 80 + "\n", flush=True)

except Exception as e:
    print(f"\n[FATAL ERROR] {e}", flush=True)
    traceback.print_exc()
    print("\n", flush=True)