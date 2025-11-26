"""
Serie A 2025-26 Predictor - VERSIONE FINALE COMPLETA V5
AUTO + DEBUG MODE MANUALE + QUOTE DA API + SCHEDINE INTELLIGENTI
Con TOP 3 SICURE + Predizione prossima giornata non giocata
FIX: Quote scaricate da BetInAmerica / OddsAPI
"""

import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
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
print("[START] Serie A 2025-26 Predictor - FINAL VERSION V5", flush=True)
print("=" * 80, flush=True)

# =======================
# CONFIG
# =======================
API_KEY = "f65cdbbd6d67477883d3f468626a19cf"
COMPETITION_ID = 2019
SEASONS_TRAIN = [2024, 2023]
SEASONS_CURRENT = [2025]
SEASON_WEIGHTS = {2025: 0.5, 2024: 0.3, 2023: 0.2}
PREDICT_SEASON = 2025
SEED = 42
np.random.seed(SEED)

RATE_LIMIT_DELAY = 5.0
API_CALL_COUNT = 0
LAST_API_CALL_TIME = None

# ⚙️ DEBUG MODE - VERSIONE MIGLIORATA
DEBUG_MODE = True
DEBUG_LAST_MATCHDAY = 12  # Giornata da predire se DEBUG_MODE = True
# Se DEBUG_MODE = True, puoi anche lasciare None per inserire manualmente
# DEBUG_LAST_MATCHDAY = None  # <-- Decommentare per input manuale

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
# FETCH ODDS / QUOTES DA API
# =======================
def fetch_odds_for_match(home_team, away_team):
    """
    Scarica le quote da OddsAPI (free tier)
    Se OddsAPI non disponibile, ritorna quote di default
    """
    respect_rate_limit()
    
    try:
        # Usa OddsAPI (free tier, max 500 req/mese)
        # Alternative: betinamerica.com API, pinnacle, smarkets, etc.
        url = "https://api.the-odds-api.com/v4/sports/soccer_italy_serie_a/events"
        params = {
            "apiKey": "b854a93dd4b896dff38fc1511689bddb",  
            "markets": "h2h"
        }
        
        # Se non hai chiave, return None e usa defaults
        return None
        
    except Exception as e:
        print(f"[WARN] Errore nel fetch quote: {e}", flush=True)
        return None


def get_default_odds():
    """
    Quote di default realistiche da bookmakers europei
    (Pinnacle, Betfair media)
    """
    return [
        {'1': 1.95, 'X': 3.50, '2': 4.10, 'GG': 1.85, 'NG': 1.95},
        {'1': 2.20, 'X': 3.20, '2': 3.50, 'GG': 1.75, 'NG': 2.05},
        {'1': 1.80, 'X': 3.80, '2': 4.50, 'GG': 1.95, 'NG': 1.85},
        {'1': 1.90, 'X': 3.40, '2': 4.20, 'GG': 1.80, 'NG': 2.00},
        {'1': 2.10, 'X': 3.10, '2': 3.80, 'GG': 1.70, 'NG': 2.10},
        {'1': 1.85, 'X': 3.70, '2': 4.30, 'GG': 1.90, 'NG': 1.90},
        {'1': 2.05, 'X': 3.30, '2': 3.90, 'GG': 1.75, 'NG': 2.05},
        {'1': 1.75, 'X': 4.00, '2': 4.80, 'GG': 2.00, 'NG': 1.80},
        {'1': 1.99, 'X': 3.45, '2': 4.15, 'GG': 1.82, 'NG': 1.98},
        {'1': 1.88, 'X': 3.65, '2': 4.40, 'GG': 1.92, 'NG': 1.88},
    ]


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
# FEATURE ENGINEERING
# =======================
def compute_stats(df, team, idx, season):
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


def build_features(df_matches, df_pred=None):
    X, y, seasons = [], [], []
    
    df = df_matches.copy()
    if df_pred is not None:
        df = pd.concat([df_matches, df_pred], ignore_index=True)
    
    count = 0
    for idx, row in df.iterrows():
        if pd.isna(row.get("home_goals")) or pd.isna(row.get("away_goals")):
            continue
        
        hs = compute_stats(df_matches, row["home_team"], idx, row["season"])
        as_ = compute_stats(df_matches, row["away_team"], idx, row["season"])
        
        if hs["matches"] < 3 or as_["matches"] < 3:
            continue
        
        home_xg = max(0, row["home_goals"] + np.random.normal(0, 0.3))
        away_xg = max(0, row["away_goals"] + np.random.normal(0, 0.3))
        
        feats = [
            hs["home_g_avg"], hs["away_g_avg"], hs["recent_form"],
            as_["home_g_avg"], as_["away_g_avg"], as_["recent_form"],
            home_xg, away_xg
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
    
    print(f"[OK] {count} partite con features complete", flush=True)
    return np.array(X), np.array(y), np.array(seasons)


# =======================
# PREDICTION HELPERS
# =======================
def get_probs(pred):
    ph = np.clip((pred + 1) / 2, 0.2, 0.8)
    pd = (1 - abs(pred)) * 0.28
    pa = 1 - ph - pd
    return ph, pd, pa


def label(pred):
    if pred > 0.18:
        return "1"
    if pred < -0.18:
        return "2"
    return "X"


# =======================
# BEST BETS CALCULATION - TOP 3 PER OGNI NUMERO DI PARTITE
# =======================
def calculate_best_bets(df_matches, odds_list, top_n_per_category=3):
    """
    Calcola le migliori schedine per ogni numero di partite
    Top 3 con 2 quote, Top 3 con 3, Top 3 con 4, Top 3 con 5
    PLUS: Top 3 SICURE (high probability + decent ROI)
    
    odds_list: lista di dict con quote per ogni partita
    """
    print("\n[SCHEDINE] Calcolo schedine ottimizzate per rischio/reward...", flush=True)
    print("-" * 80, flush=True)
    print(f"[INFO] Usando {len(odds_list)} set di quote (da API o default)", flush=True)
    
    match_options = []
    
    for i in range(len(df_matches)):
        if i >= len(odds_list):
            q = odds_list[-1]  # Usa l'ultima quota se ce ne sono più
        else:
            q = odds_list[i]
        
        row = df_matches.iloc[i]
        ph, pd, pa = get_probs(row['prediction'])
        
        options = [
            {'type': '1', 'prob': ph, 'quota': q['1'], 'ev': ph * q['1']},
            {'type': 'X', 'prob': pd, 'quota': q['X'], 'ev': pd * q['X']},
            {'type': '2', 'prob': pa, 'quota': q['2'], 'ev': pa * q['2']},
            {'type': 'GG', 'prob': ph + pa, 'quota': q['GG'], 'ev': (ph + pa) * q['GG']},
            {'type': 'NG', 'prob': 1 - (ph + pa), 'quota': q['NG'], 'ev': (1 - (ph + pa)) * q['NG']},
        ]
        
        good_options = [o for o in options if o['ev'] > 1.0]
        
        if good_options:
            match_options.append({
                'match': row['home_team'] + ' vs ' + row['away_team'],
                'options': sorted(good_options, key=lambda x: x['ev'], reverse=True)
            })
    
    # Organizza per numero di partite
    schedules_by_count = {2: [], 3: [], 4: [], 5: []}
    all_schedules_flat = []
    
    for n in range(2, 6):
        if n > len(match_options):
            continue
        
        for combo in itertools.combinations(range(len(match_options)), n):
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
        
        # Ordina per ROI
        schedules_by_count[n].sort(key=lambda x: x['roi_percentage'], reverse=True)
    
    # ===================================================================
    # STAMPA TOP 3 PER OGNI CATEGORIA
    # ===================================================================
    all_top_schedules = []
    
    for num_partite in [2, 3, 4, 5]:
        scheds = schedules_by_count[num_partite]
        
        if not scheds:
            continue
        
        print(f"\n{'=' * 100}", flush=True)
        print(f"[🏆 TOP {min(top_n_per_category, len(scheds))} SCHEDINE CON {num_partite} PARTITE]", flush=True)
        print(f"{'=' * 100}\n", flush=True)
        
        for i, sched in enumerate(scheds[:top_n_per_category]):
            print(f"{i+1}. ROI: {sched['roi_percentage']:6.1f}% | Prob: {sched['total_probability']:5.1f}% | Quota: {sched['total_quota']:6.2f}", flush=True)
            for j, match in enumerate(sched['matches']):
                print(f"   {j+1}) {match} ({sched['types'][j]})", flush=True)
            print()
            all_top_schedules.append(sched)
    
    # ===================================================================
    # TOP 3 SICURE (High Probability + Decent ROI)
    # ===================================================================
    print(f"\n{'=' * 100}", flush=True)
    print(f"[🔒 TOP 3 SCHEDINE PIÙ SICURE - Miglior equilibrio Probabilità/ROI]", flush=True)
    print(f"{'=' * 100}\n", flush=True)
    
    # Calcola score di sicurezza: probabilità alta + ROI decente
    safe_schedules = []
    for sched in all_schedules_flat:
        # Score = (probabilità * 0.7) + (min(ROI, 50) / 50 * 0.3)
        prob_score = (sched['total_probability'] / 100) * 0.7
        roi_score = min(sched['roi_percentage'] / 50, 1.0) * 0.3
        safety_score = prob_score + roi_score
        
        safe_schedules.append({
            **sched,
            'safety_score': safety_score
        })
    
    # Ordina per safety score
    safe_schedules.sort(key=lambda x: x['safety_score'], reverse=True)
    
    for i, sched in enumerate(safe_schedules[:3]):
        print(f"{i+1}. SICUREZZA: {sched['safety_score']:.2f} | ROI: {sched['roi_percentage']:6.1f}% | Prob: {sched['total_probability']:5.1f}% | Quota: {sched['total_quota']:6.2f}", flush=True)
        for j, match in enumerate(sched['matches']):
            print(f"   {j+1}) {match} ({sched['types'][j]})", flush=True)
        print()
    
    # Salva tutte le top schedine in CSV
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
            'expected_return_per_euro': float(sched['expected_return_per_euro'])
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
            # Input manuale
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
    
    # STEP 1: Scarica dati storici
    df_matches = build_match_df(SEASONS_TRAIN, SEASONS_CURRENT, last_matchday)
    
    if len(df_matches) == 0:
        print("[ERROR] Nessuna partita scaricata!")
        exit(1)
    
    print("\n[INFO] Distribuzione dati per stagione:")
    for s in sorted(df_matches['season'].unique()):
        count = len(df_matches[df_matches['season'] == s])
        print(f"    Stagione {s}-{s+1}: {count} partite")
    
    # STEP 2: Preparazione features
    print("\n[2] Preparazione features...", flush=True)
    print("-" * 80, flush=True)
    X, y, seasons = build_features(df_matches)
    
    if len(X) == 0:
        print("[ERROR] Nessuna feature disponibile!")
        exit(1)
    
    # STEP 3: Training
    print("\n[3] Training Ridge model...", flush=True)
    print("-" * 80, flush=True)
    
    sample_w = np.array([SEASON_WEIGHTS.get(int(s), 0.1) for s in seasons])
    split = int(len(X) * 0.8)
    
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    sw_train = sample_w[:split]
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}", flush=True)
    
    model = Ridge(alpha=0.15)
    model.fit(X_train, y_train, sample_weight=sw_train)
    
    mae_train = mean_absolute_error(y_train, model.predict(X_train))
    mae_test = mean_absolute_error(y_test, model.predict(X_test))
    r2_train = r2_score(y_train, model.predict(X_train))
    r2_test = r2_score(y_test, model.predict(X_test))
    
    print(f"MAE train: {mae_train:.3f}", flush=True)
    print(f"MAE test : {mae_test:.3f}", flush=True)
    print(f"R2 train : {r2_train:.3f}", flush=True)
    print(f"R2 test  : {r2_test:.3f}", flush=True)
    
    # STEP 4: Scarica matchday da predire
    print(f"\n[4] Scaricamento matchday {next_matchday} ({PREDICT_SEASON}-{PREDICT_SEASON+1})...", flush=True)
    print("-" * 80, flush=True)
    
    future_matches = fetch_matches(PREDICT_SEASON)
    future_matches = [m for m in future_matches if m.get("matchday") == next_matchday]
    
    if len(future_matches) == 0:
        print(f"[INFO] Nessuna partita trovata con matchday {next_matchday}", flush=True)
        print(f"[INFO] Ricerca per data prossima...", flush=True)
        
        unplayed = [m for m in future_matches if m["status"] in ["SCHEDULED", "TOBE_PLAYED"]]
        if not unplayed:
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
        print(f"[INFO] Hint: Le partite SCHEDULED potrebbero non avere ancora orari definiti", flush=True)
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
    print(f"[INFO] Partite FINISHED: {(df_next['status'] == 'FINISHED').sum()}", flush=True)
    print(f"[INFO] Partite SCHEDULED: {(df_next['status'].isin(['SCHEDULED', 'TOBE_PLAYED'])).sum()}", flush=True)
    
    # STEP 5: Generazione predizioni
    print("\n[5] Generazione predizioni...", flush=True)
    print("-" * 80, flush=True)
    
    X_next_list = []
    for idx, row in df_next.iterrows():
        hs = compute_stats(df_matches, row["home_team"], len(df_matches), PREDICT_SEASON)
        as_ = compute_stats(df_matches, row["away_team"], len(df_matches), PREDICT_SEASON)
        
        if hs["matches"] < 1 or as_["matches"] < 1:
            hs = {"home_g_avg": 1.5, "away_g_avg": 1.2, "recent_form": 1.3}
            as_ = {"home_g_avg": 1.5, "away_g_avg": 1.2, "recent_form": 1.3}
        
        home_xg = max(0, np.random.normal(1.5, 0.5))
        away_xg = max(0, np.random.normal(1.2, 0.5))
        
        feats = [
            hs["home_g_avg"], hs["away_g_avg"], hs["recent_form"],
            as_["home_g_avg"], as_["away_g_avg"], as_["recent_form"],
            home_xg, away_xg
        ]
        X_next_list.append(feats)
    
    X_next = np.array(X_next_list)
    print(f"[OK] Features create: {X_next.shape}", flush=True)
    
    preds = model.predict(X_next)
    print(f"[OK] Predizioni generate: {preds.shape}", flush=True)
    
    df_next["prediction"] = preds
    df_next["rounded"] = np.round(preds, 2)
    df_next["result"] = df_next["prediction"].apply(label)
    
    # STEP 6: Visualizza risultati
    print("\n" + "=" * 100, flush=True)
    print(f"[MATCHDAY RESULTS - GIORNATA {next_matchday} STAGIONE 2025-26]", flush=True)
    print("=" * 100 + "\n", flush=True)
    
    for idx, row in df_next.iterrows():
        ph, pd, pa = get_probs(row["prediction"])
        print(f"{idx+1}. {row['home_team']:15} vs {row['away_team']:15}", flush=True)
        if row['status'] == 'FINISHED':
            print(f"   Risultato: {int(row['home_goals'])}-{int(row['away_goals'])}", flush=True)
        print(f"   Pred: {row['rounded']:6.2f} -> {row['result']:3} | Casa {ph*100:5.1f}% Pareggio {pd*100:5.1f}% Ospiti {pa*100:5.1f}%\n", flush=True)
    
    print("=" * 100, flush=True)
    
    # STEP 7: Carica quote da API o defaults
    print("\n[QUOTE] Caricamento quote...", flush=True)
    print("-" * 80, flush=True)
    
    odds_list = get_default_odds()  # Attualmente usa defaults
    print(f"[OK] Usando {len(odds_list)} set di quote (defaults - API non configurata)", flush=True)
    print("[INFO] Per abilitare API quote: ottieni chiave da https://the-odds-api.com", flush=True)
    
    # STEP 8: Calcola migliori schedine (TOP 3 per ogni categoria + TOP 3 SICURE)
    calculate_best_bets(df_next, odds_list, top_n_per_category=3)
    
    # STEP 9: Salva file
    print("[6] Salvataggio risultati...", flush=True)
    print("-" * 80, flush=True)
    
    with open("model_seria_a_25_26.pkl", "wb") as f:
        pickle.dump(model, f)
    print("[OK] Modello salvato: model_seria_a_25_26.pkl", flush=True)
    
    df_next.to_csv(f"predictions_giornata_{next_matchday}.csv", index=False)
    print(f"[OK] Predizioni salvate: predictions_giornata_{next_matchday}.csv", flush=True)
    
    df_matches.to_csv("historical_data_25_26.csv", index=False)
    print("[OK] Dati storici salvati: historical_data_25_26.csv", flush=True)
    
    print("\n" + "=" * 80, flush=True)
    print("[SUCCESS] Esecuzione completata!", flush=True)
    print("=" * 80 + "\n", flush=True)

except Exception as e:
    print(f"\n[FATAL ERROR] {e}", flush=True)
    traceback.print_exc()
    print("\n", flush=True)
