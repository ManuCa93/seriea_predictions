"""
EUROPEAN PREDICTOR 2025-26 - VERSIONE V21 GLOBAL WAR ROOM
FEATURES:
1. SCANSIONE TOTALE: Serie A + Premier League + La Liga (30+ Partite).
2. Schedine MISTE (Cross-League) per la massima diversificazione.
3. Gerarchie Globali e Modello Unificato.
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

print("=" * 100, flush=True)
print("[START] EUROPEAN PREDICTOR - V21 GLOBAL WAR ROOM", flush=True)
print("=" * 100, flush=True)

# =======================
# CONFIGURAZIONE
# =======================
API_KEY = "f65cdbbd6d67477883d3f468626a19cf"
SEASONS_TRAIN = [2024, 2023]
SEASONS_CURRENT = [2025]
PREDICT_SEASON = 2025
SEED = 42
np.random.seed(SEED)
BUDGET_TOTALE = 100.0 

DEBUG_MODE = True
# Definiamo l'ultima giornata giocata PER OGNI LEGA (approx per simulazione)
DEBUG_MATCHDAYS = {
    'SA': 13, # Serie A
    'PL': 14, # Premier
    'PD': 14  # Liga
}

# CONFIGURAZIONE LEGHE
LEAGUES_CONFIG = [
    {'code': 'SA', 'id': 2019, 'name': 'Serie A'},
    {'code': 'PL', 'id': 2021, 'name': 'Premier League'},
    {'code': 'PD', 'id': 2014, 'name': 'La Liga'}
]

# GERARCHIE GLOBALI (Unite)
TOP_TEAMS = [
    # Italia
    'Inter', 'Juventus', 'Milan', 'Napoli', 'Atalanta', 'Roma', 'Lazio',
    # Inghilterra
    'Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Tottenham', 'Man United', 'Newcastle',
    # Spagna
    'Real Madrid', 'Barcelona', 'Atleti', 'Girona', 'Athletic Club'
]

WEAK_ATTACKS = [
    # Italia
    'Lecce', 'Cagliari', 'Empoli', 'Monza', 'Venezia', 'Genoa', 'Verona', 'Udinese', 'Como',
    # Inghilterra
    'Southampton', 'Ipswich', 'Leicester', 'Everton', 'Wolves', 'Crystal Palace',
    # Spagna
    'Leganes', 'Valladolid', 'Espanyol', 'Getafe', 'Las Palmas', 'Valencia'
]

# =======================
# RATE LIMITING
# =======================
RATE_LIMIT_DELAY = 2.0 
API_CALL_COUNT = 0
LAST_API_CALL_TIME = None

def respect_rate_limit():
    global API_CALL_COUNT, LAST_API_CALL_TIME
    current_time = time.time()
    
    # Reset counter every minute
    if LAST_API_CALL_TIME is not None and (current_time - LAST_API_CALL_TIME) > 60:
        API_CALL_COUNT = 0
        
    # Free tier limit: ~10 calls / minute. We be safe with 8.
    if API_CALL_COUNT >= 8:
        wait_time = 61 - (current_time - LAST_API_CALL_TIME)
        if wait_time > 0:
            print(f"[WAIT] Rate Limit Protection... attendo {wait_time:.1f}s", flush=True)
            time.sleep(wait_time)
            API_CALL_COUNT = 0
            LAST_API_CALL_TIME = time.time()
            return

    if LAST_API_CALL_TIME is not None:
        elapsed = current_time - LAST_API_CALL_TIME
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
            
    LAST_API_CALL_TIME = time.time()
    API_CALL_COUNT += 1

# =======================
# FETCH DATA
# =======================
def fetch_matches(comp_id, season, league_name):
    respect_rate_limit()
    url = f"https://api.football-data.org/v4/competitions/{comp_id}/matches"
    headers = {"X-Auth-Token": API_KEY}
    params = {"season": season}
    print(f"[API] Scaricamento {league_name} ({season})...", flush=True)
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
        return matches
    except Exception as e:
        print(f"[ERROR] Fallito scaricamento {league_name}: {e}", flush=True)
        return []

# =======================
# ODDS
# =======================
def fetch_odds_global(df_matches):
    print("[QUOTE] Assegnazione quote ai match...", flush=True)
    odds_list = []
    default_odds = get_default_odds()
    for idx, row in df_matches.iterrows():
        # Usa l'indice per ruotare le quote simulate
        fallback_idx = idx % len(default_odds)
        odds_list.append(default_odds[fallback_idx])
    return odds_list

def get_default_odds():
    # Mix di quote verosimili
    return [
        {'1': 2.80, 'X': 3.20, '2': 2.55, '1X': 1.50, '2X': 1.40, 'GG': 1.77, 'NG': 1.97},
        {'1': 1.50, 'X': 4.40, '2': 6.00, '1X': 1.11, '2X': 2.50, 'GG': 1.80, 'NG': 1.90}, 
        {'1': 4.75, 'X': 3.80, '2': 1.70, '1X': 2.10, '2X': 1.17, 'GG': 1.80, 'NG': 1.90}, 
        {'1': 2.40, 'X': 3.00, '2': 3.15, '1X': 1.32, '2X': 1.50, 'GG': 1.97, 'NG': 1.75}, 
        {'1': 5.25, 'X': 3.60, '2': 1.67, '1X': 2.10, '2X': 1.13, 'GG': 2.00, 'NG': 1.70},
        {'1': 2.55, 'X': 3.00, '2': 2.85, '1X': 1.40, '2X': 1.47, 'GG': 2.00, 'NG': 1.73},
        {'1': 2.30, 'X': 2.95, '2': 3.40, '1X': 1.30, '2X': 1.57, 'GG': 1.95, 'NG': 1.77}, 
        {'1': 2.35, 'X': 2.95, '2': 3.35, '1X': 1.30, '2X': 1.55, 'GG': 1.97, 'NG': 1.77}, 
        {'1': 2.40, 'X': 3.00, '2': 3.15, '1X': 1.32, '2X': 1.50, 'GG': 1.97, 'NG': 1.75},
        {'1': 4.60, 'X': 3.55, '2': 1.75, '1X': 2.00, '2X': 1.17, 'GG': 1.82, 'NG': 1.90},


        {'1': 4.10, 'X': 3.40, '2': 1.87, '1X': 1.85, '2X': 1.20, 'GG': 1.83, 'NG': 1.88},
        {'1': 3.10, 'X': 3.65, '2': 2.10, '1X': 1.67, '2X': 1.33, 'GG': 1.50, 'NG': 2.40},
        {'1': 2.15, 'X': 3.15, '2': 3.55, '1X': 1.27, '2X': 1.65, 'GG': 1.80, 'NG': 1.90},
        {'1': 1.23, 'X': 6.00, '2': 11.00, '1X': 1.02, '2X': 3.80, 'GG': 1.97, 'NG': 1.75},
        {'1': 1.28, 'X': 5.50, '2': 9.50, '1X': 1.03, '2X': 3.40, 'GG': 2.00, 'NG': 1.73},
        {'1': 2.20, 'X': 3.50, '2': 3.05, '1X': 1.35, '2X': 1.60, 'GG': 1.60, 'NG': 2.20},
        {'1': 4.25, 'X': 4.10, '2': 1.70, '1X': 2.05, '2X': 1.19, 'GG': 1.55, 'NG': 2.30},
        {'1': 1.55, 'X': 4.10, '2': 5.25, '1X': 1.13, '2X': 2.35, 'GG': 1.63, 'NG': 2.15},
        {'1': 2.55, 'X': 3.25, '2': 2.70, '1X': 1.40, '2X': 1.47, 'GG': 1.73, 'NG': 2.00},
        {'1': 4.10, 'X': 3.85, '2': 1.75, '1X': 2.00, '2X': 1.20, 'GG': 1.60, 'NG': 2.20},

        
        {'1': 2.65, 'X': 3.00, '2': 2.80, '1X': 1.40, '2X': 1.43, 'GG': 1.95, 'NG': 1.77},
        {'1': 1.55, 'X': 3.85, '2': 6.00, '1X': 1.10, '2X': 2.35, 'GG': 2.05, 'NG': 1.70},
        {'1': 2.70, 'X': 2.95, '2': 2.75, '1X': 1.40, '2X': 1.40, 'GG': 1.97, 'NG': 1.75},
        {'1': 3.85, 'X': 4.25, '2': 1.75, '1X': 2.00, '2X': 1.23, 'GG': 1.35, 'NG': 2.90},
        {'1': 3.40, 'X': 3.25, '2': 2.15, '1X': 1.65, '2X': 1.30, 'GG': 1.87, 'NG': 1.85},
        {'1': 2.15, 'X': 3.40, '2': 3.30, '1X': 1.30, '2X': 1.65, 'GG': 1.67, 'NG': 2.10},
        {'1': 2.10, 'X': 3.40, '2': 3.40, '1X': 1.30, '2X': 1.67, 'GG': 1.80, 'NG': 1.90},
        {'1': 2.25, 'X': 3.35, '2': 3.10, '1X': 1.33, '2X': 1.60, 'GG': 1.90, 'NG': 1.80},
        {'1': 1.32, 'X': 5.50, '2': 8.00, '1X': 1.05, '2X': 3.15, 'GG': 1.60, 'NG': 2.20},
        {'1': 1.75, 'X': 3.65, '2': 4.50, '1X': 1.17, '2X': 2.00, 'GG': 1.77, 'NG': 1.93},
    ]

# =======================
# DATA PROCESSING
# =======================
def parse_match(m, s, league_code):
    return {
        "league": league_code,
        "season": s,
        "matchday": m.get("matchday", 0),
        "home_team": m["homeTeam"]["name"],
        "away_team": m["awayTeam"]["name"],
        "date": m["utcDate"],
        "home_goals": m["score"]["fullTime"]["home"],
        "away_goals": m["score"]["fullTime"]["away"],
    }

def build_global_dataset(leagues, seasons_train, seasons_curr, debug_mds):
    print("\n[1] COSTRUZIONE DATASET GLOBALE (IT/EN/ES)...")
    print("-" * 80)
    all_rows = []
    
    for league in leagues:
        l_code = league['code']
        l_id = league['id']
        current_md = debug_mds.get(l_code, 10) # Default 10 se manca
        
        # Training Seasons
        for s in seasons_train:
            matches = fetch_matches(l_id, s, league['name'])
            for m in matches:
                if m["status"] in ["FINISHED", "LIVE"]:
                    all_rows.append(parse_match(m, s, l_code))
        
        # Current Season
        for s in seasons_curr:
            matches = fetch_matches(l_id, s, league['name'])
            count_curr = 0
            for m in matches:
                if m["status"] in ["FINISHED", "LIVE"]:
                    if m.get("matchday", 0) <= current_md:
                        all_rows.append(parse_match(m, s, l_code))
                        count_curr += 1
            print(f" -> {league['name']}: {count_curr} partite correnti caricate.")

    df = pd.DataFrame(all_rows)
    if not df.empty: df["date"] = pd.to_datetime(df["date"])
    print(f"[OK] TOTALE GLOBALE: {len(df)} partite pronte per l'analisi.\n", flush=True)
    return df

def compute_elo(df, k=20):
    teams = list(set(df["home_team"]).union(set(df["away_team"])))
    elo = {t: 1500 for t in teams}
    home_elo_list, away_elo_list = [], []
    
    # Sort by date to ensure correct ELO progression
    df = df.sort_values('date')
    
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
# STATS & FEATURES
# =======================
def compute_stats_v21(df, team, idx):
    # Filtra solo partite precedenti a 'idx' (cronologia corretta)
    df_prev = df[df.index < idx]
    
    last5_h = df_prev[df_prev["home_team"] == team]
    last5_a = df_prev[df_prev["away_team"] == team]
    last_matches = pd.concat([last5_h, last5_a]).sort_values('date').tail(5)
    
    if len(last_matches) < 2: return 1.4, 1.3 # Default League Avg
    
    scored = 0
    conceded = 0
    for _, m in last_matches.iterrows():
        if m['home_team'] == team:
            scored += m['home_goals']
            conceded += m['away_goals']
        else:
            scored += m['away_goals']
            conceded += m['home_goals']
            
    return scored / len(last_matches), conceded / len(last_matches)

def build_features_v21(df):
    print("[2] CALCOLO FEATURES AVANZATE...")
    df = compute_elo(df) # Add ELO columns
    
    X, y = [], []
    df['stat_h_score'] = 0.0
    df['stat_h_conc'] = 0.0
    df['stat_a_score'] = 0.0
    df['stat_a_conc'] = 0.0
    
    for idx, row in df.iterrows():
        if pd.isna(row.get("home_goals")): continue
        
        h_s, h_c = compute_stats_v21(df, row["home_team"], idx)
        a_s, a_c = compute_stats_v21(df, row["away_team"], idx)
        
        # Save for Poisson
        df.at[idx, 'stat_h_score'] = h_s
        df.at[idx, 'stat_h_conc'] = h_c
        df.at[idx, 'stat_a_score'] = a_s
        df.at[idx, 'stat_a_conc'] = a_c
        
        # Features for RF
        feats = [
            row["elo_home"], row["elo_away"],
            h_s, h_c, a_s, a_c,
            row["elo_home"] - row["elo_away"]
        ]
        X.append(feats)
        
        # Target
        if row["home_goals"] > row["away_goals"]: y.append(2)
        elif row["home_goals"] < row["away_goals"]: y.append(0)
        else: y.append(1)
            
    print(f"[OK] Training Set Creato: {len(X)} campioni.")
    return np.array(X), np.array(y), df

def train_model(X, y):
    print("\n[3] TRAINING INTELLIGENZA ARTIFICIALE...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split cronologico 85/15
    split = int(len(X) * 0.85)
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = RandomForestClassifier(n_estimators=250, max_depth=6, random_state=SEED)
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[STATS] Precisione Modello (Test Set): {acc:.3f}", flush=True)
    return model, scaler

# =======================
# POISSON TUNED V21
# =======================
def is_top(t): return any(x in t for x in TOP_TEAMS)
def is_weak(t): return any(x in t for x in WEAK_ATTACKS)

def calc_poisson_v21(h_name, a_name, h_s, h_c, a_s, a_c):
    LEAGUE_FACTOR = 1.05
    lh = ((h_s * 0.6 + a_c * 0.4) + 0.1) * LEAGUE_FACTOR
    la = ((a_s * 0.6 + h_c * 0.4) - 0.1) * LEAGUE_FACTOR
    
    # Hierarchy Logic
    if is_top(h_name) and is_weak(a_name): la *= 0.50
    if is_top(a_name) and is_weak(h_name): lh *= 0.65
    
    # Caps
    lh = max(0.3, lh)
    la = max(0.2, la)
    
    ph = 1 - poisson.pmf(0, lh) # Prob goal home
    pa = 1 - poisson.pmf(0, la) # Prob goal away
    
    p_gg = ph * pa
    p_ng = 1 - p_gg
    return p_gg, p_ng, lh, la

# =======================
# PREDICTION ENGINE
# =======================
def predict_next_games(leagues, df_hist, model, scaler):
    print("\n[4] ANALISI PARTITE FUTURE (TUTTE LE LEGHE)...")
    future_rows = []
    
    for league in leagues:
        l_code = league['code']
        l_id = league['id']
        next_md = DEBUG_MATCHDAYS.get(l_code, 10) + 1
        
        matches = fetch_matches(l_id, PREDICT_SEASON, league['name'])
        # Filter for next matchday
        targets = [m for m in matches if m.get('matchday') == next_md]
        
        # Fallback if specific matchday is empty
        if not targets:
            targets = [m for m in matches if m['status'] == 'SCHEDULED'][:5]
            
        print(f" -> {league['name']}: {len(targets)} match trovati.")
        
        for m in targets:
            future_rows.append(parse_match(m, PREDICT_SEASON, l_code))
            
    df_next = pd.DataFrame(future_rows)
    X_next = []
    
    print("\n" + "="*100)
    print(f"{'LEGA':<5} | {'MATCH':<35} | {'PRED':<5} | {'1 %':<4} | {'X %':<4} | {'2 %':<4} | {'NG %':<4}")
    print("="*100)
    
    for i, row in df_next.iterrows():
        # Calc stats using history
        h_s, h_c = compute_stats_v21(df_hist, row['home_team'], len(df_hist)+1)
        a_s, a_c = compute_stats_v21(df_hist, row['away_team'], len(df_hist)+1)
        
        # Get ELO
        last_h = df_hist[df_hist['home_team']==row['home_team']].tail(1)
        elo_h = last_h['elo_home'].values[0] if not last_h.empty else 1500
        last_a = df_hist[df_hist['away_team']==row['away_team']].tail(1)
        elo_a = last_a['elo_away'].values[0] if not last_a.empty else 1500
        
        # Save for later
        df_next.at[i, 'stat_h_score'] = h_s
        df_next.at[i, 'stat_h_conc'] = h_c
        df_next.at[i, 'stat_a_score'] = a_s
        df_next.at[i, 'stat_a_conc'] = a_c
        
        feat = [elo_h, elo_a, h_s, h_c, a_s, a_c, elo_h - elo_a]
        X_next.append(feat)
        
    # RF Predict
    X_sc = scaler.transform(np.array(X_next))
    probs = model.predict_proba(X_sc)
    df_next['probs'] = list(probs)
    
    # Display Loop
    for i, row in df_next.iterrows():
        pr = row['probs']
        pa, px, ph = pr[0], pr[1], pr[2] # 0=2, 1=X, 2=1
        
        if ph > pa and ph > px: res = "1"
        elif pa > ph and pa > px: res = "2"
        else: res = "X"
        
        _, p_ng, _, _ = calc_poisson_v21(
            row['home_team'], row['away_team'],
            row['stat_h_score'], row['stat_h_conc'],
            row['stat_a_score'], row['stat_a_conc']
        )
        
        match_str = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
        print(f"{row['league']:<5} | {match_str:<35} | {res:<5} | {ph*100:<4.0f} | {px*100:<4.0f} | {pa*100:<4.0f} | {p_ng*100:<4.0f}")
        
    return df_next

# =======================
# BETTING ENGINE (CROSS-LEAGUE)
# =======================
def generate_combinations(options, n, min_p, min_e):
    if n > len(options): return []
    combos = itertools.combinations(options, n)
    combos = itertools.islice(combos, 10000) # Higher limit for cross-league
    valid = []
    
    for c in combos:
        data = []
        for idx in range(len(c)):
            opt = c[idx]['options'][0]
            data.append({
                'match': c[idx]['match'],
                'type': opt['type'],
                'prob': opt['prob'],
                'quota': opt['quota'],
                'ev': opt['ev']
            })
        
        tp = np.prod([x['prob'] for x in data])
        te = np.prod([x['ev'] for x in data])
        tq = np.prod([x['quota'] for x in data])
        
        if tp >= min_p and te > min_e:
            score = tp * (te**2)
            valid.append({
                'n': n,
                'matches': [x['match'] for x in data],
                'types': [x['type'] for x in data],
                'quota': tq,
                'prob': tp*100,
                'roi': (te-1)*100,
                'score': score
            })
    valid.sort(key=lambda x: x['score'], reverse=True)
    return valid

def calculate_best_bets(df_next, odds_list):
    print("\n[SCHEDINE] Generazione Schedine MISTE (Cross-League)...")
    
    opts_std = []
    opts_hard = []
    
    for i, row in df_next.iterrows():
        q = odds_list[i] if i < len(odds_list) else odds_list[-1]
        pr = row['probs']
        pa, px, ph = pr[0], pr[1], pr[2]
        
        p_gg, p_ng, _, _ = calc_poisson_v21(
            row['home_team'], row['away_team'],
            row['stat_h_score'], row['stat_h_conc'],
            row['stat_a_score'], row['stat_a_conc']
        )
        
        # Filtro Validità
        def valid(p, e): return (p > 0.65 and e > 0.85) or (p > 0.55 and e > 0.92) or (e > 1.02)
        
        raw = [
            {'type': '1', 'prob': ph, 'quota': q['1'], 'ev': ph*q['1']},
            {'type': 'X', 'prob': px, 'quota': q['X'], 'ev': px*q['X']},
            {'type': '2', 'prob': pa, 'quota': q['2'], 'ev': pa*q['2']},
            {'type': '1X', 'prob': ph+px, 'quota': q.get('1X', 1.05), 'ev': (ph+px)*q.get('1X', 1.05)},
            {'type': '2X', 'prob': pa+px, 'quota': q.get('2X', 1.05), 'ev': (pa+px)*q.get('2X', 1.05)},
            {'type': 'GG', 'prob': p_gg, 'quota': q['GG'], 'ev': p_gg*q['GG']},
            {'type': 'NG', 'prob': p_ng, 'quota': q['NG'], 'ev': p_ng*q['NG']},
        ]
        
        match_lbl = f"[{row['league']}] {row['home_team']} vs {row['away_team']}"
        
        good_std = [o for o in raw if valid(o['prob'], o['ev'])]
        if good_std:
            opts_std.append({'match': match_lbl, 'options': sorted(good_std, key=lambda x: x['prob'], reverse=True)})
            
        good_hard = [o for o in good_std if o['type'] not in ['1X', '2X', '12']]
        if good_hard:
            opts_hard.append({'match': match_lbl, 'options': sorted(good_hard, key=lambda x: x['prob'], reverse=True)})

    # Generate
    port_std = []
    port_hard = []
    
    print("\n>>> SCHEDINE STANDARD (Include Doppie)")
    for n in [2, 3, 4, 5]:
        res = generate_combinations(opts_std, n, 0.15, 0.95)
        if res:
            port_std.extend(res)
            print(f" [Top {n} Match] Quota: {res[0]['quota']:.2f} | Prob: {res[0]['prob']:.1f}% | ROI: {res[0]['roi']:.1f}%")

    print("\n>>> SCHEDINE HARDCORE (No Doppie)")
    for n in [2, 3, 4, 5]:
        res = generate_combinations(opts_hard, n, 0.08, 1.00)
        if res:
            port_hard.extend(res)
            print(f" [Top {n} Match] Quota: {res[0]['quota']:.2f} | Prob: {res[0]['prob']:.1f}% | ROI: {res[0]['roi']:.1f}%")
            
    print_final_strategy(port_std, port_hard, BUDGET_TOTALE)

def check_overlap(s1, s2):
    m1 = set([m.split(' vs ')[0] for m in s1['matches']])
    m2 = set([m.split(' vs ')[0] for m in s2['matches']])
    return len(m1.intersection(m2)) > 0 # ZERO overlap tollerato per la diversificazione vera

def print_final_strategy(all_std, all_hard, budget):
    print("\n\n")
    print("X" * 100)
    print(f"X   PORTAFOGLIO DIVERSIFICATO GLOBALE (BUDGET: {budget}€)   X")
    print("X" * 100)
    
    # 1. SAFE (Standard)
    all_std.sort(key=lambda x: x['score'], reverse=True)
    best_safe = all_std[0] if all_std else None
    
    # 2. VALUE (Hardcore) - NO OVERLAP
    all_hard.sort(key=lambda x: x['score'], reverse=True)
    best_value = None
    if best_safe:
        for s in all_hard:
            if not check_overlap(best_safe, s):
                best_value = s
                break
        if not best_value and all_hard: best_value = all_hard[0]
    else:
        best_value = all_hard[0] if all_hard else None
        
    # 3. BONUS (Longshot)
    best_bonus = None
    longshots = [s for s in all_hard if s['quota'] > 10.0]
    longshots.sort(key=lambda x: x['roi'], reverse=True)
    
    if longshots:
        for s in longshots:
            c1 = check_overlap(best_safe, s) if best_safe else False
            c2 = check_overlap(best_value, s) if best_value else False
            if not c1 and not c2:
                best_bonus = s
                break
        if not best_bonus: best_bonus = longshots[0]

    # Print
    allocs = [0.50, 0.35, 0.15]
    labels = ["CASSAFORTE 🛡️", "VALORE 💎", "COLPO GROSSO 🚀"]
    picks = [best_safe, best_value, best_bonus]
    
    for i, p in enumerate(picks):
        if p:
            amt = budget * allocs[i]
            print(f"\n{i+1}. [{labels[i]}] (Inv: {amt:.2f}€ -> Vincita: {amt*p['quota']:.2f}€)")
            print(f"   Quota: {p['quota']:.2f} | Prob: {p['prob']:.1f}% | ROI: {p['roi']:.1f}%")
            for j, m in enumerate(p['matches']):
                print(f"   - {m} -> {p['types'][j]}")
        else:
            print(f"\n{i+1}. [{labels[i]}] Nessuna giocata disponibile.")
            
    print("\n" + "="*100)

# =======================
# MAIN EXECUTION
# =======================
try:
    print("\n[0] INIZIO SCANSIONE EUROPA...")
    
    # 1. Build Global History
    df_hist = build_global_dataset(LEAGUES_CONFIG, SEASONS_TRAIN, SEASONS_CURRENT, DEBUG_MATCHDAYS)
    
    # 2. Train Global Model
    X, y, df_hist = build_features_v21(df_hist)
    model, scaler = train_model(X, y)
    
    # 3. Predict Future (All Leagues)
    df_next = predict_next_games(LEAGUES_CONFIG, df_hist, model, scaler)
    
    # 4. Odds
    odds = fetch_odds_global(df_next)
    
    # 5. Calculate
    calculate_best_bets(df_next, odds)
    
    print("\n[DONE] Analisi Completata.")
    
except Exception as e:
    print(f"\n[CRITICAL ERROR] {e}")
    traceback.print_exc()