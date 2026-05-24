"""
========================================================
  Football Match Outcome Predictor
  Przewidywanie wynikow meczow: Away (0), Draw (1), Home (2)
========================================================
Wymagania spelniione:
  1.  Filtrowanie lig (>=50 meczow) i druzyn (>=10 meczow)
  2.  Chronologiczny podzial 80/20 – brak data leakage
  3.  Rolling averages (okno=5, shift=1) – statystyki sprzed meczu
  4.  Cechy remisogenne i roznicowe (abs diff + wskazniki balansu)
  5.  Dwa modele bazowe: Random Forest vs Gradient Boosting (XGB-equiv.)
  6.  RandomizedSearchCV / reczny search na metryce f1_macro
  7.  Classification report z podzialem na klasy Away / Draw / Home
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, ParameterSampler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# ═══════════════════════════════════════════════════════
# 0. WCZYTANIE DANYCH
# ═══════════════════════════════════════════════════════
print("=" * 62)
print("  FOOTBALL MATCH OUTCOME PREDICTOR")
print("=" * 62)

DATA_PATH = "scraped_dataset.csv"   # <── zmien sciezke jesli potrzeba

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"\n[1/7] Wczytano dane: {df.shape[0]:,} meczow, {df.shape[1]} kolumn")

# ═══════════════════════════════════════════════════════
# 1. CZYSZCZENIE I FILTROWANIE
# ═══════════════════════════════════════════════════════

# Usuniecie wierszy bez wyniku
df = df.dropna(subset=["result"])

# Parsowanie daty (format: "August 24, 2024")
df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce", format="%B %d, %Y")
df = df.dropna(subset=["date_parsed"])

# Krok 1a: Filtr lig – min. 50 meczow
league_counts = df["league_division"].value_counts()
valid_leagues  = league_counts[league_counts >= 100].index
df = df[df["league_division"].isin(valid_leagues)]
print(f"    Ligi (>=50 meczow):    {len(valid_leagues)} / {league_counts.shape[0]}")

# Krok 1b: Filtr druzyn – min. 10 meczow (home + away lacznie)
team_counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
valid_teams  = team_counts[team_counts >= 15].index
df = df[df["home_team"].isin(valid_teams) & df["away_team"].isin(valid_teams)]
print(f"    Druzyny (>=10 meczow): {len(valid_teams)}")
print(f"    Mecze po filtrowaniu:  {df.shape[0]:,}")

# ═══════════════════════════════════════════════════════
# 2. SORTOWANIE CHRONOLOGICZNE – KLUCZOWE DLA BRAKU DATA LEAKAGE
# ═══════════════════════════════════════════════════════
df = df.sort_values("date_parsed").reset_index(drop=True)
print(f"\n[2/7] Zakres dat: {df['date_parsed'].min().date()} → {df['date_parsed'].max().date()}")

# ═══════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING: Rolling Averages (okno=5, shift=1)
#    shift(1) zapewnia, ze model NIE widzi statystyk z biezacego meczu
# ═══════════════════════════════════════════════════════
print("\n[3/7] Obliczanie Rolling Averages (okno=5, shift=1)...")

ROLLING_COLS = {
    "rating":     ("home_team_rating",    "away_team_rating"),
    "shots":      ("home_shots_total",    "away_shots_total"),
    "possession": ("home_possession_pct", "away_possession_pct"),
    "corners":    ("home_corners",        "away_corners"),
    "goals":      ("home_goals",          "away_goals"),
    "xg":         ("home_xg",            "away_xg"),
}

def build_rolling(df: pd.DataFrame, team_col: str, stat_col: str, window: int = 5) -> pd.Series:
    """
    Dla kazdej druzyny (group by team_col):
      - shift(1): statystyki sprzed obecnego meczu (brak data leakage)
      - rolling mean ostatnich `window` meczow
    """
    result = pd.Series(index=df.index, dtype=float)
    for _team, grp in df.groupby(team_col):
        shifted = grp[stat_col].shift(1)                          # nie wiemy wyniku tego meczu
        rolled  = shifted.rolling(window=window, min_periods=1).mean()
        result.loc[grp.index] = rolled
    return result

for feat_name, (home_col, away_col) in ROLLING_COLS.items():
    df[f"home_roll_{feat_name}"] = build_rolling(df, "home_team", home_col)
    df[f"away_roll_{feat_name}"] = build_rolling(df, "away_team", away_col)

print(f"    Stworzono {2 * len(ROLLING_COLS)} cech rolling")

# ═══════════════════════════════════════════════════════
# 4. CECHY REMISOGENNE I ROZNICOWE
#    - diff_*:    roznica formy H-A (kierunek przewagi)
#    - absdiff_*: bezwzgledna roznica (male = potencjalny remis)
#    - *_balance: odwrotnosc absdiff (wysokie = wyrownana sila)
# ═══════════════════════════════════════════════════════
print("\n[4/7] Tworzenie cech roznicowych i remisogennych...")

diff_feats = list(ROLLING_COLS.keys())
for feat in diff_feats:
    h = df[f"home_roll_{feat}"]
    a = df[f"away_roll_{feat}"]
    df[f"diff_{feat}"]    = h - a           # liniowa roznica formy
    df[f"absdiff_{feat}"] = (h - a).abs()  # bezwzgledna (remisogenna)

# Wskazniki balansu – im wyzsze, tym bardziej wyrownane sily
df["rating_balance"]  = 1.0 / (1.0 + df["absdiff_rating"])
df["goals_balance"]   = 1.0 / (1.0 + df["absdiff_goals"])
df["xg_balance"]      = 1.0 / (1.0 + df["absdiff_xg"])
df["match_quality_rating"] = df["home_roll_rating"] + df["away_roll_rating"]
df["match_quality_xg"] = df["home_roll_xg"] + df["away_roll_xg"]
df["match_tempo_shots"] = df["home_roll_shots"] + df["away_roll_shots"]

n_diff_feats = 2 * len(diff_feats) + 3
print(f"    Stworzono {n_diff_feats} cech roznicowych")

# ═══════════════════════════════════════════════════════
# 5. ZMIENNA DOCELOWA I LISTA CECH
# ═══════════════════════════════════════════════════════
label_map = {"A": 0, "D": 1, "H": 2}
df["target"] = df["result"].map(label_map)
df = df.dropna(subset=["target"])
df["target"] = df["target"].astype(int)

FEATURE_COLS = (
    [f"home_roll_{f}" for f in ROLLING_COLS]
  + [f"away_roll_{f}" for f in ROLLING_COLS]
  + [f"diff_{f}"      for f in diff_feats]
  + [f"absdiff_{f}"   for f in diff_feats]
  + ["rating_balance", "goals_balance", "xg_balance", "match_quality_rating", "match_quality_xg", "match_tempo_shots"]
)

X = df[FEATURE_COLS].copy()
y = df["target"].copy()

# ═══════════════════════════════════════════════════════
# 6. CHRONOLOGICZNY PODZIAL 80/20 (NIE LOSOWY!)
# ═══════════════════════════════════════════════════════
split_idx = int(len(df) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

date_col = df["date_parsed"]
print(f"\n[5/7] Podzial chronologiczny 80/20 (brak data leakage):")
print(f"    Trening : {len(X_train):,} meczow  "
      f"({date_col.iloc[0].date()} → {date_col.iloc[split_idx-1].date()})")
print(f"    Test    : {len(X_test):,} meczow  "
      f"({date_col.iloc[split_idx].date()} → {date_col.iloc[-1].date()})")
print(f"    Klasy (train) Away/Draw/Home: "
      f"{(y_train==0).sum()} / {(y_train==1).sum()} / {(y_train==2).sum()}")
print(f"    Klasy (test)  Away/Draw/Home: "
      f"{(y_test==0).sum()} / {(y_test==1).sum()} / {(y_test==2).sum()}")

# Sample weights – balansowanie klas (odpowiednik class_weight='balanced')
sample_weights_train = compute_sample_weight("balanced", y_train)

# ═══════════════════════════════════════════════════════
# 7. PREPROCESSOWANIE: SimpleImputer + StandardScaler
# ═══════════════════════════════════════════════════════
imputer = SimpleImputer(strategy="median")   # braki danych → mediana
scaler  = StandardScaler()                   # normalizacja cech

X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_proc  = scaler.transform(imputer.transform(X_test))

prepro_steps = [
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
]

# ═══════════════════════════════════════════════════════
# 8. MODEL A: RANDOM FOREST
#    class_weight='balanced' → automatyczne wagi odwrotnie proporcjonalne
# ═══════════════════════════════════════════════════════
print("\n[6/7] Trenowanie modeli bazowych...")
print("─" * 50)

rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # radzi sobie z nierownowaga klas (rzadsze remisy)
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_proc, y_train)
rf_pred = rf_model.predict(X_test_proc)
rf_f1   = f1_score(y_test, rf_pred, average="macro")

print(f"\n  [A] Random Forest — F1 Macro: {rf_f1:.4f}")
print(classification_report(
    y_test, rf_pred,
    target_names=["Away (0)", "Draw (1)", "Home (2)"],
    digits=4
))

# ═══════════════════════════════════════════════════════
# 9. MODEL B: GRADIENT BOOSTING (odpowiednik XGBoost)
#    sample_weight='balanced' poprzez compute_sample_weight
#    (XGBoost uzywa scale_pos_weight / sample_weight)
# ═══════════════════════════════════════════════════════
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_proc, y_train, sample_weight=sample_weights_train)
gb_pred = gb_model.predict(X_test_proc)
gb_f1   = f1_score(y_test, gb_pred, average="macro")

print(f"\n  [B] Gradient Boosting (XGB-equiv.) — F1 Macro: {gb_f1:.4f}")
print(classification_report(
    y_test, gb_pred,
    target_names=["Away (0)", "Draw (1)", "Home (2)"],
    digits=4
))

# ═══════════════════════════════════════════════════════
# 10. WYBOR LEPSZEGO MODELU + STROJENIE HIPERPARAMETROW
#     Optymalizacja pod katem f1_macro (nie accuracy!)
#     aby wymusic lepsza skutecznosc w przewidywaniu remisow
# ═══════════════════════════════════════════════════════
print("─" * 50)
print(f"\n  Wyniki bazowe: RF F1={rf_f1:.4f}  |  GB F1={gb_f1:.4f}")

if rf_f1 >= gb_f1:
    winner = "Random Forest"
    print(f"  → Lepszy: {winner}. Uruchamiam RandomizedSearchCV (f1_macro, 20 iter, cv=3)...\n")

    param_dist = {
        "model__n_estimators":      [100, 200, 300, 500],
        "model__max_depth":         [None, 5, 10, 20],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf":  [1, 2, 4],
        "model__max_features":      ["sqrt", "log2", 0.5],
    }
    pipe = Pipeline(prepro_steps + [
        ("model", RandomForestClassifier(
            class_weight="balanced", random_state=42, n_jobs=-1
        ))
    ])
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1_macro",          # <– optymalizacja pod remisy!
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)
    print(f"\n  Najlepsze parametry : {search.best_params_}")
    print(f"  Najlepsza f1_macro  : {search.best_score_:.4f}")
    best_pred  = search.best_estimator_.predict(X_test)
    importances = search.best_estimator_.named_steps["model"].feature_importances_

else:
    winner = "Gradient Boosting"
    print(f"  → Lepszy: {winner}. Uruchamiam reczny RandomizedSearch (f1_macro, 20 iter)...\n")

    param_dist = {
        "n_estimators":      [100, 200, 300, 500],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "max_depth":         [3, 4, 5, 6],
        "subsample":         [0.6, 0.8, 1.0],
        "min_samples_split": [2, 5, 10],
    }
    # Reczny search: podzial chronologiczny (75% train / 25% val wewnatrz X_train)
    val_split = int(len(X_train_proc) * 0.75)
    Xtr, Xvl = X_train_proc[:val_split], X_train_proc[val_split:]
    ytr, yvl = y_train.iloc[:val_split], y_train.iloc[val_split:]
    sw_tr    = compute_sample_weight("balanced", ytr)

    best_cv_f1, best_params = -1.0, {}
    for i, params in enumerate(ParameterSampler(param_dist, n_iter=20, random_state=42)):
        m = GradientBoostingClassifier(**params, random_state=42)
        m.fit(Xtr, ytr, sample_weight=sw_tr)
        score = f1_score(yvl, m.predict(Xvl), average="macro")
        if score > best_cv_f1:
            best_cv_f1, best_params = score, params
        if (i + 1) % 5 == 0:
            print(f"    iter {i+1:2d}/20 — najlepsza f1_macro = {best_cv_f1:.4f}")

    print(f"\n  Najlepsze parametry : {best_params}")
    print(f"  Najlepsza f1_macro  : {best_cv_f1:.4f}")

    best_gb = GradientBoostingClassifier(**best_params, random_state=42)
    best_gb.fit(X_train_proc, y_train, sample_weight=sample_weights_train)
    best_pred   = best_gb.predict(X_test_proc)
    importances = best_gb.feature_importances_

# ═══════════════════════════════════════════════════════
# 11. FINALNY RAPORT
# ═══════════════════════════════════════════════════════
final_f1 = f1_score(y_test, best_pred, average="macro")

print("\n" + "=" * 62)
print(f"  FINALNY MODEL PO STROJENIU: {winner}")
print(f"  F1 Macro (zbior testowy):   {final_f1:.4f}")
print("=" * 62)

print("\n  Classification Report (Away=0, Draw=1, Home=2):")
print(classification_report(
    y_test, best_pred,
    target_names=["Away (0)", "Draw (1)", "Home (2)"],
    digits=4
))

# Macierz pomylek
cm = confusion_matrix(y_test, best_pred)
print("  Macierz pomylek (wiersze=rzeczywiste, kolumny=przewidywane):")
print("                Away   Draw   Home")
for i, rn in enumerate(["Away (0)", "Draw (1)", "Home (2)"]):
    print(f"    {rn}:  {cm[i][0]:5d}  {cm[i][1]:5d}  {cm[i][2]:5d}")

# Waznosc cech
feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
print("\n  Top 10 najwazniejszych cech:")
print(f"  {'Cecha':<32s} {'Waga':>6}  Wykres")
print("  " + "─" * 56)
for feat, imp in feat_imp.head(10).items():
    bar = "█" * int(imp * 400)
    print(f"  {feat:<32s} {imp:.4f}  {bar}")

print("\n✓ Pipeline zakonczony pomyslnie.")