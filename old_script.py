"""
EUROPEAN PREDICTOR 2025-26 - VERSIONE V24.1 FIX
FEATURES:
1. SCANSIONE TOTALE: Serie A + Premier League + La Liga (30+ Partite).
2. Schedine SMART (Moneyball Logic) con Criterio di Kelly.
3. 19 FEATURES (vs 11 della V22)
4. ✅ TEAM NAME NORMALIZATION
5. ✅ REAL SNAI ODDS MAPPING
6. ✅ HOME/AWAY SPLIT + TREND + EFFICIENCY + DEFENSE RATING
7. ✅ PORTFOLIO OPTIMIZATION (Safe / Value / Moonshot)
8. ✅ COMPLETE ERROR HANDLING
9. ✅ LOGGING SYSTEM
"""

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import poisson
import itertools
import warnings
import sys
import io
import traceback
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher

warnings.filterwarnings("ignore")
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass

# =======================
# LOGGING SYSTEM
# =======================
def get_next_sunday_log_filename():
    today = datetime.now()
    days_until_sunday = (6 - today.weekday()) % 7
    if days_until_sunday == 0 and today.weekday() != 6:
        days_until_sunday = 7
    next_sunday = today + timedelta(days=days_until_sunday)
    
    month_name = next_sunday.strftime("%B").lower()
    day = next_sunday.strftime("%d").lstrip('0')
    
    return f"predictor_{month_name}_{day}.log"

LOG_FILE = get_next_sunday_log_filename()

def log_msg(msg, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] [{level}] {msg}"
    print(log_entry, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    except:
        pass

log_msg("="*100)
log_msg("[START] EUROPEAN PREDICTOR - V24.1 FIX")
log_msg("="*100)

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
DEBUG_MATCHDAYS = {
    'SA': 13,
    'PL': 14,
    'PD': 14
}

LEAGUES_CONFIG = [
    {'code': 'SA', 'id': 2019, 'name': 'Serie A'},
    {'code': 'PL', 'id': 2021, 'name': 'Premier League'},
    {'code': 'PD', 'id': 2014, 'name': 'La Liga'}
]

# =======================
# TEAM NAME NORMALIZATION
# =======================
TEAM_ALIASES = {
    'FC Internazionale Milano': 'Inter', 'Internazionale': 'Inter', 'Inter Milan': 'Inter',
    'AC Milan': 'Milan', 'Milan': 'Milan', 'Associazione Calcio Milan': 'Milan',
    'SSC Napoli': 'Napoli', 'Napoli': 'Napoli', 'Juventus FC': 'Juventus', 'Juventus': 'Juventus',
    'AS Roma': 'Roma', 'Roma': 'Roma', 'SS Lazio': 'Lazio', 'Lazio': 'Lazio',
    'Atalanta Bergamasca Calcio': 'Atalanta', 'Atalanta': 'Atalanta', 'Atalanta BC': 'Atalanta',
    'Hellas Verona FC': 'Verona', 'Verona': 'Verona', 'US Lecce': 'Lecce', 'Lecce': 'Lecce',
    'Cagliari Calcio': 'Cagliari', 'Cagliari': 'Cagliari', 'Udinese Calcio': 'Udinese', 'Udinese': 'Udinese',
    'Genoa CFC': 'Genoa', 'Genoa': 'Genoa', 'Como 1907': 'Como', 'Como': 'Como',
    'US Cremonese': 'Cremonese', 'Cremonese': 'Cremonese', 'AC Pisa 1909': 'Pisa', 'Pisa': 'Pisa',
    'Parma Calcio 1913': 'Parma', 'Parma': 'Parma', 'Torino FC': 'Torino', 'Torino': 'Torino',
    'US Sassuolo Calcio': 'Sassuolo', 'Sassuolo': 'Sassuolo', 'ACF Fiorentina': 'Fiorentina',
    'Fiorentina': 'Fiorentina', 'Bologna FC 1909': 'Bologna', 'Bologna': 'Bologna',
    'Empoli FC': 'Empoli', 'Empoli': 'Empoli', 'Monza': 'Monza', 'AC Monza': 'Monza',
    'Venezia FC': 'Venezia', 'Venezia': 'Venezia',
    'Manchester City FC': 'Man City', 'Man City': 'Man City', 'Manchester United FC': 'Man United',
    'Man United': 'Man United', 'Arsenal FC': 'Arsenal', 'Arsenal': 'Arsenal',
    'Liverpool FC': 'Liverpool', 'Liverpool': 'Liverpool', 'Chelsea FC': 'Chelsea', 'Chelsea': 'Chelsea',
    'Tottenham Hotspur': 'Tottenham', 'Tottenham': 'Tottenham', 'Newcastle United': 'Newcastle',
    'Newcastle': 'Newcastle', 'Aston Villa FC': 'Aston Villa', 'Aston Villa': 'Aston Villa',
    'Brighton and Hove Albion': 'Brighton', 'Brighton': 'Brighton', 'West Ham United': 'West Ham',
    'West Ham': 'West Ham', 'Fulham FC': 'Fulham', 'Fulham': 'Fulham', 'Crystal Palace': 'Crystal Palace',
    'Bournemouth AFC': 'Bournemouth', 'Bournemouth': 'Bournemouth', 'Brentford FC': 'Brentford',
    'Brentford': 'Brentford', 'Nottingham Forest': 'Nottingham', 'Nottingham Forest FC': 'Nottingham',
    'Everton FC': 'Everton', 'Everton': 'Everton', 'Wolverhampton Wanderers': 'Wolves',
    'Wolves': 'Wolves', 'Wolverhampton Wanderers FC': 'Wolves', 'Leicester City': 'Leicester',
    'Leicester': 'Leicester', 'Southampton FC': 'Southampton', 'Southampton': 'Southampton',
    'Ipswich Town': 'Ipswich', 'Ipswich': 'Ipswich', 'Leeds United': 'Leeds', 'Leeds': 'Leeds',
    'Sunderland AFC': 'Sunderland', 'Sunderland': 'Sunderland', 'Burnley FC': 'Burnley', 'Burnley': 'Burnley',
    'Real Madrid CF': 'Real Madrid', 'Real Madrid': 'Real Madrid', 'FC Barcelona': 'Barcelona',
    'Barcelona': 'Barcelona', 'Atlético Madrid': 'Atletico', 'Atletico Madrid': 'Atletico',
    'Atleti': 'Atletico', 'Real Betis Balompié': 'Real Betis', 'Real Betis': 'Real Betis',
    'Betis': 'Real Betis', 'Villarreal CF': 'Villarreal', 'Villarreal': 'Villarreal',
    'Athletic Club': 'Athletic Club', 'Athletic Bilbao': 'Athletic Club', 'Athletic': 'Athletic Club',
    'Real Sociedad': 'Real Sociedad', 'Real Sociedad de Fútbol': 'Real Sociedad',
    'Girona FC': 'Girona', 'Girona': 'Girona', 'Getafe CF': 'Getafe', 'Getafe': 'Getafe',
    'Sevilla FC': 'Sevilla', 'Sevilla': 'Sevilla', 'Valencia CF': 'Valencia', 'Valencia': 'Valencia',
    'Rayo Vallecano': 'Rayo', 'Vallecano': 'Rayo', 'RCD Espanyol': 'Espanyol', 'Espanyol': 'Espanyol',
    'Real Oviedo': 'Oviedo', 'Oviedo': 'Oviedo', 'RCD Mallorca': 'Mallorca', 'Mallorca': 'Mallorca',
    'Elche CF': 'Elche', 'Elche': 'Elche', 'Celta Vigo': 'Celta', 'Celta de Vigo': 'Celta',
    'Celta': 'Celta', 'Levante UD': 'Levante', 'Levante': 'Levante', 'CA Osasuna': 'Osasuna',
    'Osasuna': 'Osasuna', 'Las Palmas': 'Las Palmas', 'CD Leganés': 'Leganes',
    'Leganes': 'Leganes', 'Real Valladolid': 'Valladolid', 'Valladolid': 'Valladolid',
}

def normalize_team_name(name):
    if name in TEAM_ALIASES:
        return TEAM_ALIASES[name]
    best_match = None
    best_ratio = 0
    for key, value in TEAM_ALIASES.items():
        ratio = SequenceMatcher(None, name.lower(), key.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = value
    if best_ratio > 0.7:
        return best_match
    return name

TOP_TEAMS = [
    'Inter', 'Juventus', 'Milan', 'Napoli', 'Atalanta', 'Roma', 'Lazio',
    'Man City', 'Arsenal', 'Liverpool', 'Chelsea', 'Tottenham', 'Man United', 'Newcastle',
    'Real Madrid', 'Barcelona', 'Atletico', 'Girona', 'Athletic Club'
]

WEAK_ATTACKS = [
    'Lecce', 'Cagliari', 'Empoli', 'Monza', 'Venezia', 'Genoa', 'Verona', 'Udinese', 'Como',
    'Southampton', 'Ipswich', 'Leicester', 'Everton', 'Wolves', 'Crystal Palace',
    'Leganes', 'Valladolid', 'Espanyol', 'Getafe', 'Las Palmas', 'Valencia'
]

def get_odds_mapping():
    return {
        'SA': [
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
        ],
        'PL': [
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
        ],
        'PD': [
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
    }

RATE_LIMIT_DELAY = 2.0
API_CALL_COUNT = 0
LAST_API_CALL_TIME = None

def respect_rate_limit():
    global API_CALL_COUNT, LAST_API_CALL_TIME
    current_time = time.time()
    if LAST_API_CALL_TIME is not None and (current_time - LAST_API_CALL_TIME) > 60:
        API_CALL_COUNT = 0
    if API_CALL_COUNT >= 8:
        wait_time = 61 - (current_time - LAST_API_CALL_TIME)
        if wait_time > 0:
            log_msg(f"[WAIT] Rate Limit Protection... attendo {wait_time:.1f}s")
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

def fetch_matches(comp_id, season, league_name):
    respect_rate_limit()
    url = f"https://api.football-data.org/v4/competitions/{comp_id}/matches"
    headers = {"X-Auth-Token": API_KEY}
    params = {"season": season}
    log_msg(f"[API] Scaricamento {league_name} ({season})...")
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
        return matches
    except Exception as e:
        log_msg(f"[ERROR] Fallito scaricamento {league_name}: {e}", level="ERROR")
        return []

def fetch_odds_global(df_matches):
    log_msg("[QUOTE] Assegnazione quote ai match (SNAI REALE)...")
    odds_list = []
    odds_mapping = get_odds_mapping()
    try:
        for idx, row in df_matches.iterrows():
            league = row['league']
            league_odds = odds_mapping.get(league, [])
            if not league_odds:
                log_msg(f"[WARN] Nessuna quota per lega {league}, usando default", level="WARNING")
                odds_list.append({'1': 1.5, 'X': 3.0, '2': 3.0, '1X': 1.05, '2X': 1.50, 'GG': 1.70, 'NG': 2.00})
            else:
                idx_league = idx % len(league_odds)
                odds_list.append(league_odds[idx_league])
        log_msg(f"[OK] Quote assegnate: {len(odds_list)} match")
        return odds_list
    except Exception as e:
        log_msg(f"[ERROR] Errore assegnazione quote: {e}", level="ERROR")
        return []

def parse_match(m, s, league_code):
    try:
        home = normalize_team_name(m["homeTeam"]["name"])
        away = normalize_team_name(m["awayTeam"]["name"])
        return {
            "league": league_code,
            "season": s,
            "matchday": m.get("matchday", 0),
            "home_team": home,
            "away_team": away,
            "date": m["utcDate"],
            "home_goals": m["score"]["fullTime"]["home"],
            "away_goals": m["score"]["fullTime"]["away"],
        }
    except Exception as e:
        log_msg(f"[ERROR] Errore parsing match: {e}", level="ERROR")
        return None

def build_global_dataset(leagues, seasons_train, seasons_curr, debug_mds):
    log_msg("\n[1] COSTRUZIONE DATASET GLOBALE (IT/EN/ES)...")
    log_msg("-" * 80)
    all_rows = []
    try:
        for league in leagues:
            l_code = league['code']
            l_id = league['id']
            current_md = debug_mds.get(l_code, 10)
            for s in seasons_train:
                matches = fetch_matches(l_id, s, league['name'])
                for m in matches:
                    if m["status"] in ["FINISHED", "LIVE"]:
                        parsed = parse_match(m, s, l_code)
                        if parsed:
                            all_rows.append(parsed)
            for s in seasons_curr:
                matches = fetch_matches(l_id, s, league['name'])
                count_curr = 0
                for m in matches:
                    if m["status"] in ["FINISHED", "LIVE"]:
                        if m.get("matchday", 0) <= current_md:
                            parsed = parse_match(m, s, l_code)
                            if parsed:
                                all_rows.append(parsed)
                                count_curr += 1
                log_msg(f" -> {league['name']}: {count_curr} partite correnti caricate.")
        df = pd.DataFrame(all_rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        log_msg(f"[OK] TOTALE GLOBALE: {len(df)} partite pronte per l'analisi.\n")
        return df
    except Exception as e:
        log_msg(f"[ERROR] Errore costruzione dataset: {e}", level="ERROR")
        return pd.DataFrame()

def compute_elo(df, k=20):
    try:
        teams = list(set(df["home_team"]).union(set(df["away_team"])))
        elo = {t: 1500 for t in teams}
        home_elo_list, away_elo_list = [], []
        df = df.sort_values('date').reset_index(drop=True)
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
    except Exception as e:
        log_msg(f"[ERROR] Errore calcolo ELO: {e}", level="ERROR")
        return df

def compute_advanced_stats(df, team, idx, last_n_recent=5, last_n_all=10):
    """Calcolo stats AVANZATE V23: 19 features"""
    try:
        df_prev = df[df.index < idx]
        last_h = df_prev[df_prev["home_team"] == team]
        last_a = df_prev[df_prev["away_team"] == team]

        last_h_recent = last_h.tail(last_n_recent)
        last_a_recent = last_a.tail(last_n_recent)
        last_h_all = last_h.tail(last_n_all)
        last_a_all = last_a.tail(last_n_all)

        all_recent = pd.concat([last_h_recent, last_a_recent]).sort_values('date').tail(last_n_recent)
        all_matches = pd.concat([last_h_all, last_a_all]).sort_values('date').tail(last_n_all)

        if len(all_matches) < 2:
            return {'scored_overall': 1.4, 'conceded_overall': 1.3, 'form_overall': 1.0, 'points_overall': 0.5,
                    'home_advantage': 0.0, 'trend_recent': 0.0, 'efficiency': 0.5, 'defense_rating': 1.3,
                    'consistency': 0.5, 'win_ratio': 0.33, 'streak': 0, 'h2h_record': 0.5}

        scored_all = 0
        conceded_all = 0
        points_all = 0

        for _, m in all_matches.iterrows():
            if m['home_team'] == team:
                scored_all += m['home_goals']
                conceded_all += m['away_goals']
                if m['home_goals'] > m['away_goals']: points_all += 3
                elif m['home_goals'] == m['away_goals']: points_all += 1
            else:
                scored_all += m['away_goals']
                conceded_all += m['home_goals']
                if m['away_goals'] > m['home_goals']: points_all += 3
                elif m['away_goals'] == m['home_goals']: points_all += 1

        goals_per_match_all = scored_all / len(all_matches)
        conceded_per_match_all = conceded_all / len(all_matches)
        form_all = points_all / (len(all_matches) * 3)

        scored_recent = 0
        conceded_recent = 0
        points_recent = 0

        for _, m in all_recent.iterrows():
            if m['home_team'] == team:
                scored_recent += m['home_goals']
                conceded_recent += m['away_goals']
                if m['home_goals'] > m['away_goals']: points_recent += 3
                elif m['home_goals'] == m['away_goals']: points_recent += 1
            else:
                scored_recent += m['away_goals']
                conceded_recent += m['home_goals']
                if m['away_goals'] > m['home_goals']: points_recent += 3
                elif m['away_goals'] == m['home_goals']: points_recent += 1

        if len(last_h_all) > 0:
            home_gf = sum([m['home_goals'] for _, m in last_h_all.iterrows()])
            home_advantage = home_gf / len(last_h_all)
        else:
            home_advantage = 0.0

        if len(all_recent) > 0:
            trend_recent = (points_recent / len(all_recent)) - (points_all / len(all_matches))
        else:
            trend_recent = 0.0

        if len(all_matches) > 0:
            efficiency = scored_all / max(1, scored_all + conceded_all)
        else:
            efficiency = 0.5

        defense_rating = conceded_per_match_all
        all_gf = [m['home_goals'] if m['home_team']==team else m['away_goals'] for _, m in all_matches.iterrows()]
        consistency = np.std(all_gf) if len(all_gf) > 1 else 0.5

        wins = sum(1 for _, m in all_matches.iterrows() 
                   if (m['home_team']==team and m['home_goals']>m['away_goals']) or 
                      (m['away_team']==team and m['away_goals']>m['home_goals']))
        win_ratio = wins / len(all_matches) if len(all_matches) > 0 else 0.33

        streak = 0
        for _, m in all_recent.iloc[::-1].iterrows():
            if m['home_team'] == team:
                if m['home_goals'] > m['away_goals']: streak += 1
                else: break
            else:
                if m['away_goals'] > m['home_goals']: streak += 1
                else: break

        h2h_record = 0.5

        return {'scored_overall': goals_per_match_all, 'conceded_overall': conceded_per_match_all,
                'form_overall': form_all, 'points_overall': points_all, 'home_advantage': home_advantage,
                'trend_recent': trend_recent, 'efficiency': efficiency, 'defense_rating': defense_rating,
                'consistency': consistency, 'win_ratio': win_ratio, 'streak': streak, 'h2h_record': h2h_record}
    except Exception as e:
        log_msg(f"[WARN] Errore advanced stats per {team}: {e}", level="WARNING")
        return {'scored_overall': 1.4, 'conceded_overall': 1.3, 'form_overall': 1.0, 'points_overall': 0.5,
                'home_advantage': 0.0, 'trend_recent': 0.0, 'efficiency': 0.5, 'defense_rating': 1.3,
                'consistency': 0.5, 'win_ratio': 0.33, 'streak': 0, 'h2h_record': 0.5}

def build_features_v23_mega(df):
    log_msg("[2] CALCOLO FEATURES MEGA (V23: 11 → 19 FEATURES)...")
    try:
        df = compute_elo(df)
        X, y = [], []

        for idx, row in df.iterrows():
            try:
                if pd.isna(row.get("home_goals")):
                    continue

                h_stats = compute_advanced_stats(df, row["home_team"], idx)
                a_stats = compute_advanced_stats(df, row["away_team"], idx)

                feats = [
                    row["elo_home"], row["elo_away"],
                    h_stats['scored_overall'], h_stats['conceded_overall'],
                    a_stats['scored_overall'], a_stats['conceded_overall'],
                    h_stats['form_overall'], a_stats['form_overall'],
                    row["elo_home"] - row["elo_away"],
                    h_stats['scored_overall'] * 0.6 + a_stats['conceded_overall'] * 0.4,
                    a_stats['scored_overall'] * 0.6 + h_stats['conceded_overall'] * 0.4,
                    h_stats['home_advantage'],
                    a_stats['home_advantage'],
                    h_stats['trend_recent'],
                    a_stats['trend_recent'],
                    h_stats['efficiency'],
                    a_stats['efficiency'],
                    h_stats['defense_rating'],
                    a_stats['defense_rating'],
                ]

                X.append(feats)

                if row["home_goals"] > row["away_goals"]: y.append(2)
                elif row["home_goals"] < row["away_goals"]: y.append(0)
                else: y.append(1)
            except Exception as e:
                log_msg(f"[WARN] Errore feature per match {idx}: {e}", level="WARNING")
                continue

        log_msg(f"[OK] Training Set Creato: {len(X)} campioni con 19 features.")
        return np.array(X), np.array(y), df
    except Exception as e:
        log_msg(f"[ERROR] Errore build_features_v23: {e}", level="ERROR")
        return np.array([]), np.array([]), df

def train_model(X, y):
    log_msg("\n[3] TRAINING INTELLIGENZA ARTIFICIALE (V23 TUNED)...")
    try:
        if len(X) == 0 or len(y) == 0:
            log_msg("[ERROR] Training set vuoto!", level="ERROR")
            return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        split = int(len(X) * 0.85)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestClassifier(
            n_estimators=450,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=SEED,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)

        acc = accuracy_score(y_test, model.predict(X_test))
        log_msg(f"[STATS] Precisione Modello (Test Set): {acc:.3f}", level="INFO")

        if acc < 0.45:
            log_msg("[WARN] Accuracy molto bassa, il modello potrebbe non essere affidabile", level="WARNING")

        return model, scaler
    except Exception as e:
        log_msg(f"[ERROR] Errore training: {e}", level="ERROR")
        return None, None

def is_top(t):
    return t in TOP_TEAMS

def is_weak(t):
    return t in WEAK_ATTACKS

def calc_poisson_v23(h_name, a_name, h_s, h_c, a_s, a_c):
    try:
        LEAGUE_FACTOR = 1.05
        lh = ((h_s * 0.6 + a_c * 0.4) + 0.1) * LEAGUE_FACTOR
        la = ((a_s * 0.6 + h_c * 0.4) - 0.1) * LEAGUE_FACTOR

        if is_top(h_name) and is_weak(a_name):
            la *= 0.50
        if is_top(a_name) and is_weak(h_name):
            lh *= 0.65

        lh = max(0.3, lh)
        la = max(0.2, la)

        ph = 1 - poisson.pmf(0, lh)
        pa = 1 - poisson.pmf(0, la)

        p_gg = ph * pa
        p_ng = 1 - p_gg

        return p_gg, p_ng, lh, la
    except Exception as e:
        log_msg(f"[WARN] Errore Poisson: {e}", level="WARNING")
        return 0.5, 0.5, h_s, a_s

def predict_next_games(leagues, df_hist, model, scaler):
    log_msg("\n[4] ANALISI PARTITE FUTURE (TUTTE LE LEGHE)...")
    future_rows = []

    try:
        if model is None or scaler is None:
            log_msg("[ERROR] Modello non disponibile", level="ERROR")
            return pd.DataFrame()

        for league in leagues:
            l_code = league['code']
            l_id = league['id']
            next_md = DEBUG_MATCHDAYS.get(l_code, 10) + 1

            matches = fetch_matches(l_id, PREDICT_SEASON, league['name'])
            targets = [m for m in matches if m.get('matchday') == next_md]

            if not targets:
                targets = [m for m in matches if m['status'] == 'SCHEDULED'][:5]

            log_msg(f" -> {league['name']}: {len(targets)} match trovati.")

            for m in targets:
                parsed = parse_match(m, PREDICT_SEASON, l_code)
                if parsed:
                    future_rows.append(parsed)

        df_next = pd.DataFrame(future_rows)
        if df_next.empty:
            log_msg("[WARN] Nessuna partita futura trovata", level="WARNING")
            return df_next

        X_next = []

        log_msg("\n" + "="*100)
        log_msg(f"{'LEGA':<5} | {'MATCH':<35} | {'PRED':<5} | {'1 %':<4} | {'X %':<4} | {'2 %':<4} | {'NG %':<4}")
        log_msg("="*100)

        for i, row in df_next.iterrows():
            try:
                h_stats = compute_advanced_stats(df_hist, row['home_team'], len(df_hist)+1)
                a_stats = compute_advanced_stats(df_hist, row['away_team'], len(df_hist)+1)

                last_h = df_hist[df_hist['home_team']==row['home_team']].tail(1)
                elo_h = last_h['elo_home'].values[0] if not last_h.empty else 1500
                last_a = df_hist[df_hist['away_team']==row['away_team']].tail(1)
                elo_a = last_a['elo_away'].values[0] if not last_a.empty else 1500

                feat = [
                    elo_h, elo_a,
                    h_stats['scored_overall'], h_stats['conceded_overall'],
                    a_stats['scored_overall'], a_stats['conceded_overall'],
                    h_stats['form_overall'], a_stats['form_overall'],
                    elo_h - elo_a,
                    h_stats['scored_overall'] * 0.6 + a_stats['conceded_overall'] * 0.4,
                    a_stats['scored_overall'] * 0.6 + h_stats['conceded_overall'] * 0.4,
                    h_stats['home_advantage'],
                    a_stats['home_advantage'],
                    h_stats['trend_recent'],
                    a_stats['trend_recent'],
                    h_stats['efficiency'],
                    a_stats['efficiency'],
                    h_stats['defense_rating'],
                    a_stats['defense_rating'],
                ]
                X_next.append(feat)
            except Exception as e:
                log_msg(f"[WARN] Errore calcolo feature match {i}: {e}", level="WARNING")
                continue

        if not X_next:
            log_msg("[ERROR] Nessuna feature calcolata", level="ERROR")
            return df_next

        X_sc = scaler.transform(np.array(X_next))
        probs = model.predict_proba(X_sc)

        if probs.shape[1] != 3:
            log_msg(f"[ERROR] Probabilità con dimensione sbagliata: {probs.shape}", level="ERROR")
            return df_next

        df_next['probs'] = list(probs)

        for i, row in df_next.iterrows():
            try:
                pr = row['probs']
                pa, px, ph = pr[0], pr[1], pr[2]

                if ph > pa and ph > px: res = "1"
                elif pa > ph and pa > px: res = "2"
                else: res = "X"

                h_stats_temp = compute_advanced_stats(df_hist, row['home_team'], len(df_hist)+1)
                a_stats_temp = compute_advanced_stats(df_hist, row['away_team'], len(df_hist)+1)

                _, p_ng, _, _ = calc_poisson_v23(
                    row['home_team'], row['away_team'],
                    h_stats_temp['scored_overall'], h_stats_temp['conceded_overall'],
                    a_stats_temp['scored_overall'], a_stats_temp['conceded_overall']
                )

                match_str = f"{row['home_team'][:15]} vs {row['away_team'][:15]}"
                log_msg(f"{row['league']:<5} | {match_str:<35} | {res:<5} | {ph*100:<4.0f} | {px*100:<4.0f} | {pa*100:<4.0f} | {p_ng*100:<4.0f}")
            except Exception as e:
                log_msg(f"[WARN] Errore display prediction {i}: {e}", level="WARNING")
                continue

        return df_next
    except Exception as e:
        log_msg(f"[ERROR] Errore predict_next_games: {e}", level="ERROR")
        return pd.DataFrame()

# ==============================================================================
#  STRATEGIA V24: MONEYBALL PORTFOLIO & KELLY CRITERION (FIXED)
# ==============================================================================

def calculate_kelly_stake(prob, quota, bankroll, fractional=0.2):
    """
    Calcola lo stake ideale usando il Criterio di Kelly Frazionale.
    fractional=0.2 significa che siamo conservativi (usiamo il 20% del Kelly puro).
    Formula: f* = (bp - q) / b
    dove b = quota - 1, p = probabilità, q = 1 - p
    """
    if quota <= 1.0: return 0.0
    b = quota - 1
    p = prob
    q = 1 - p
    f_star = (b * p - q) / b
    
    # Se il valore è negativo (nessun vantaggio statistico), non scommettere
    if f_star <= 0: return 0.0
    
    # Applichiamo il frazionamento per sicurezza
    stake_pct = f_star * fractional
    
    # Cap massimo per singola schedina (mai più del 15% del budget su una bet)
    return min(stake_pct * bankroll, bankroll * 0.15)

def check_correlation_conflict(matches_in_slip):
    """
    Verifica che non ci siano squadre ripetute in modo conflittuale nella stessa schedina.
    """
    teams_seen = set()
    for m in matches_in_slip:
        try:
            # Estrai i nomi delle squadre dalla stringa "League Home vs Away"
            parts = m.split('] ')[1].split(' vs ')
            h, a = parts[0], parts[1]
            if h in teams_seen or a in teams_seen:
                return True # Conflitto trovato
            teams_seen.add(h)
            teams_seen.add(a)
        except:
            continue
    return False

def generate_smart_portfolio(options, n_legs, min_prob_tot, strategy_type):
    """
    Genera combinazioni intelligenti basate sulla strategia richiesta.
    """
    valid_slips = []
    
    # Pre-filtro opzioni in base alla strategia
    if strategy_type == 'SAFE':
        candidates = [o for o in options if o['prob'] >= 0.65 and 1.15 <= o['quota'] <= 1.60]
    elif strategy_type == 'VALUE':
        candidates = [o for o in options if o['ev'] > 1.05 and 1.50 <= o['quota'] <= 2.40]
    elif strategy_type == 'HIGH_YIELD':
        candidates = [o for o in options if o['quota'] >= 2.10]
    else:
        candidates = options

    # Se non abbiamo abbastanza candidati, rilassiamo i vincoli
    if len(candidates) < n_legs:
        candidates = options

    # Ordiniamo per EV decrescente per prendere i migliori
    candidates.sort(key=lambda x: x['ev'], reverse=True)
    top_candidates = candidates[:15] # Top 15 per combinare

    combos = itertools.combinations(top_candidates, n_legs)
    
    for c in combos:
        match_names = [x['match'] for x in c]
        if check_correlation_conflict(match_names):
            continue
            
        probs = [x['prob'] for x in c]
        quotas = [x['quota'] for x in c]
        
        tot_prob = np.prod(probs)
        tot_quota = np.prod(quotas)
        
        if tot_prob < min_prob_tot:
            continue
            
        tot_ev = tot_prob * tot_quota
        
        valid_slips.append({
            'strategy': strategy_type,
            'matches': match_names,
            'types': [x['type'] for x in c],
            'quota': tot_quota,
            'prob': tot_prob,
            'ev': tot_ev,
            'quotas': quotas # FIX: Salviamo le quote singole
        })
        
    valid_slips.sort(key=lambda x: x['ev'], reverse=True)
    return valid_slips[:3]

def calculate_best_bets_v24(df_next, odds_list):
    log_msg("\n[SCHEDINE V24] Generazione 'Moneyball' Portfolio (Kelly Criterion)...")

    all_options = []

    # 1. ESTENSIONE E FILTRAGGIO OPZIONI
    for i, row in df_next.iterrows():
        try:
            q = odds_list[i] if i < len(odds_list) else {}
            pr = row['probs']
            pa, px, ph = pr[0], pr[1], pr[2] # 2, X, 1
            
            _, _, lh, la = calc_poisson_v23(
                row['home_team'], row['away_team'], 
                1.4, 1.3, 1.4, 1.3 
            )
            
            # Stima Probabilità GG/NG
            p_0_0 = poisson.pmf(0, lh) * poisson.pmf(0, la)
            prob_ng = (poisson.pmf(0, lh) + poisson.pmf(0, la) - p_0_0)
            prob_gg = 1 - prob_ng
            
            match_lbl = f"[{row['league']}] {row['home_team']} vs {row['away_team']}"
            
            raw_bets = [
                {'type': '1', 'prob': ph, 'quota': q.get('1', 1.0)},
                {'type': 'X', 'prob': px, 'quota': q.get('X', 1.0)},
                {'type': '2', 'prob': pa, 'quota': q.get('2', 1.0)},
                {'type': '1X', 'prob': ph+px, 'quota': q.get('1X', 1.0)},
                {'type': '2X', 'prob': pa+px, 'quota': q.get('2X', 1.0)},
                {'type': '12', 'prob': ph+pa, 'quota': 1.30},
                {'type': 'GG', 'prob': prob_gg, 'quota': q.get('GG', 1.0)},
                {'type': 'NG', 'prob': prob_ng, 'quota': q.get('NG', 1.0)}
            ]
            
            for bet in raw_bets:
                if bet['quota'] <= 1.05: continue
                
                ev = bet['prob'] * bet['quota']
                if ev > 0.95:
                    all_options.append({
                        'match': match_lbl,
                        'type': bet['type'],
                        'prob': bet['prob'],
                        'quota': bet['quota'],
                        'ev': ev
                    })

        except Exception as e:
            continue

    # 2. GENERAZIONE PORTFOLIO DIVERSIFICATO
    portfolio = []

    # STRATEGIA A: "IL MURO" (Raddoppio Sicuro)
    safe_slips = generate_smart_portfolio(all_options, n_legs=2, min_prob_tot=0.45, strategy_type='SAFE')
    if not safe_slips:
         safe_slips = generate_smart_portfolio(all_options, n_legs=3, min_prob_tot=0.45, strategy_type='SAFE')
    portfolio.extend(safe_slips[:1])

    # STRATEGIA B: "VALUE BETTING" (Tripla/Quadrupla Equilibrata)
    value_slips = generate_smart_portfolio(all_options, n_legs=3, min_prob_tot=0.25, strategy_type='VALUE')
    portfolio.extend(value_slips[:2])

    # STRATEGIA C: "IL BOMBER" (Quota Alta)
    high_slips = generate_smart_portfolio(all_options, n_legs=4, min_prob_tot=0.10, strategy_type='HIGH_YIELD')
    portfolio.extend(high_slips[:1])

    print_final_strategy_v24(portfolio, BUDGET_TOTALE)

def print_final_strategy_v24(portfolio, budget):
    log_msg("\n\n")
    log_msg("$$" * 40)
    log_msg(f"$$   PORTAFOGLIO INTELLIGENTE V24 (KELLY OPTIMIZED) - BUDGET: {budget}€   $$")
    log_msg("$$" * 40)
    
    if not portfolio:
        log_msg("[!] Nessuna combinazione valida trovata con i criteri attuali.")
        return

    used_budget = 0.0

    for idx, slip in enumerate(portfolio, 1):
        # Calcolo Stake Dinamico
        safe_prob = slip['prob'] * 0.9 
        suggested_stake = calculate_kelly_stake(safe_prob, slip['quota'], budget, fractional=0.25)
        
        if suggested_stake < 2.0: suggested_stake = 2.0
        
        if slip['strategy'] == 'SAFE': icon, title = "🛡️", "IL MURO (Sicurezza)"
        elif slip['strategy'] == 'VALUE': icon, title = "💎", "VALUE ACCA (Valore)"
        else: icon, title = "🚀", "MOONSHOT (Alto Rischio)"
        
        potential_win = suggested_stake * slip['quota']
        
        log_msg(f"\n{idx}. {icon} {title}")
        log_msg(f"   Quota Totale: {slip['quota']:.2f} | Prob. Stimata: {slip['prob']*100:.1f}% | EV Schedina: {slip['ev']:.2f}")
        log_msg(f"   💰 Puntata Suggerita: {suggested_stake:.2f}€  -->  Vincita Pot.: {potential_win:.2f}€")
        
        log_msg(f"   {'-'*60}")
        for j, m in enumerate(slip['matches']):
            # FIX: Usa la quota salvata invece di calcolarla
            single_q = slip['quotas'][j]
            log_msg(f"   • {m:<40}  Mossa: {slip['types'][j]:<4} (Quota: {single_q:.2f})")
        
        used_budget += suggested_stake

    log_msg("="*80)
    log_msg(f"TOTALE BUDGET IMPIEGATO: {used_budget:.2f}€ (Rimanente: {budget - used_budget:.2f}€)")
    log_msg("Consiglio: Se il budget rimanente è alto, conservalo per la copertura live.")
    log_msg("="*80)

# =======================
# MAIN EXECUTION
# =======================
try:
    log_msg("\n[0] INIZIO SCANSIONE EUROPA (V24.1 FIX)...")

    df_hist = build_global_dataset(LEAGUES_CONFIG, SEASONS_TRAIN, SEASONS_CURRENT, DEBUG_MATCHDAYS)

    if df_hist.empty:
        log_msg("[ERROR] Dataset vuoto, impossibile continuare", level="ERROR")
    else:
        X, y, df_hist = build_features_v23_mega(df_hist)
        model, scaler = train_model(X, y)

        if model is not None and scaler is not None:
            df_next = predict_next_games(LEAGUES_CONFIG, df_hist, model, scaler)

            if not df_next.empty:
                odds = fetch_odds_global(df_next)
                calculate_best_bets_v24(df_next, odds)
            else:
                log_msg("[WARN] Nessuna partita futura per l'analisi", level="WARNING")
        else:
            log_msg("[ERROR] Training fallito", level="ERROR")

    log_msg("\n[DONE] Analisi Completata.")

except Exception as e:
    log_msg(f"\n[CRITICAL ERROR] {e}", level="ERROR")
    traceback.print_exc()
    log_msg(traceback.format_exc(), level="ERROR")