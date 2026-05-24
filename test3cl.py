"""
===================================================================
  PREDYKTOR WYNIKÓW MECZÓW PIŁKARSKICH — Zaawansowany Model ML
  Klasy: Away (0) | Draw (1) | Home (2)
===================================================================

WYNIKI NA scraped_dataset.csv:
  - Po filtracji (ligi ≥200, drużyny ≥10): ~13 800 meczów
  - Podział chronologiczny: Train 80% | Test 20%

  ┌───────────────────────────────────────────────────┬──────────┬──────────┐
  │ Model                                             │ Accuracy │ F1 Macro │
  ├───────────────────────────────────────────────────┼──────────┼──────────┤
  │ Logistic Regression (baseline)                    │  0.4492  │  0.4183  │
  │ Random Forest (baseline)                          │  0.4585  │  0.3464  │
  │ LR + dual-window(5+10) + 8 statystyk (FINALNY)   │  0.4562  │  0.4373  │
  └───────────────────────────────────────────────────┴──────────┴──────────┘

DLACZEGO LOGISTIC REGRESSION > RANDOM FOREST:
  Przy ograniczonej liczbie meczów i dużej losowości wyników piłkarskich,
  dane mają charakter bardziej liniowy niż hierarchiczny. RF przeuczy się
  do klasy dominującej (Home), ignorując remisy. LR z class_weight='balanced'
  i regularyzacją C=0.3 zachowuje lepszą równowagę między klasami.

KLUCZOWE ULEPSZENIA vs. bazowy kod:
  1. Dual-window rolling: okno 5 (forma krótkoterminowa) + okno 10 (trend)
  2. 8 statystyk zamiast 5: +xg, +shots_total, +goalkeeper_saves
  3. Cechy draw-detection: draw_signal dla okna 5 i 10
  4. ID drużyn + ID ligi jako cechy kontekstowe
  5. RandomizedSearchCV optymalizujący f1_macro
===================================================================
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from scipy.stats import loguniform

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────
DATA_PATH        = "scraped_dataset.csv"
MIN_LEAGUE_GAMES = 200   # wyższy próg = czystsze dane lig
MIN_TEAM_GAMES   = 10
TRAIN_SPLIT      = 0.80
WINDOW_SHORT     = 5     # forma krótkoterminowa
WINDOW_LONG      = 10    # trend długoterminowy
RANDOM_STATE     = 42

# Statystyki do rolling (nazwa: (kolumna_home, kolumna_away))
STATS = {
    "rating":     ("home_team_rating",       "away_team_rating"),
    "shots_ot":   ("home_shots_on_target",   "away_shots_on_target"),
    "shots_tot":  ("home_shots_total",       "away_shots_total"),
    "poss":       ("home_possession_pct",    "away_possession_pct"),
    "corners":    ("home_corners",           "away_corners"),
    "goals":      ("home_goals",             "away_goals"),
    "xg":         ("home_xg",               "away_xg"),
    "gk_saves":   ("home_goalkeeper_saves",  "away_goalkeeper_saves"),
}


# ─────────────────────────────────────────────────────────────────
# KROK 1: WCZYTANIE I FILTROWANIE
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  KROK 1: Wczytywanie i filtrowanie")
print("=" * 65)

df = pd.read_csv(DATA_PATH, low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "result"])
df = df.sort_values("date").reset_index(drop=True)
print(f"  Wczytano: {len(df):,} meczów")

# Filtrowanie lig
lc = df["league_division"].value_counts()
df = df[df["league_division"].isin(lc[lc >= MIN_LEAGUE_GAMES].index)]
print(f"  Ligi z ≥{MIN_LEAGUE_GAMES} meczami: {(lc >= MIN_LEAGUE_GAMES).sum()} / {len(lc)}")

# Filtrowanie drużyn
tc = pd.concat([df["home_team"], df["away_team"]]).value_counts()
df = df[
    df["home_team"].isin(tc[tc >= MIN_TEAM_GAMES].index) &
    df["away_team"].isin(tc[tc >= MIN_TEAM_GAMES].index)
]
print(f"  Po filtracji: {len(df):,} meczów")

# Kodowanie targetu (LabelEncoder gwarantuje A=0, D=1, H=2)
df["target"] = LabelEncoder().fit_transform(df["result"])
print(f"\n  Rozkład klas:")
for name, code in [("Away (A)", 0), ("Draw (D)", 1), ("Home (H)", 2)]:
    n = (df["target"] == code).sum()
    print(f"    {name}: {n:>5,}  ({n/len(df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────
# KROK 2: ROLLING AVERAGES — DUAL WINDOW (5 i 10)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  KROK 2: Rolling Averages — dual window (5 krótka + 10 długa forma)")
print("  shift(1): statystyki bieżącego meczu NIEZNANE modelowi → brak wycieku")
print("=" * 65)


def build_rolling(data: pd.DataFrame, window: int) -> pd.DataFrame:
    """Buduje rolling features dla wszystkich drużyn i statystyk."""
    frames = []
    for stat, (hcol, acol) in STATS.items():
        home = data[["date", "home_team", hcol]].copy()
        home.columns = ["date", "team", "val"]
        away = data[["date", "away_team", acol]].copy()
        away.columns = ["date", "team", "val"]
        combined = pd.concat([home, away]).sort_values(["team", "date"])
        col_name = f"{stat}_r{window}"
        combined[col_name] = combined.groupby("team")["val"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        frames.append(
            combined[["date", "team", col_name]].drop_duplicates(["date", "team"])
        )

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on=["date", "team"], how="left")
    return result


roll5  = build_rolling(df, WINDOW_SHORT)
roll10 = build_rolling(df, WINDOW_LONG)

for side, team_col in [("h", "home_team"), ("a", "away_team")]:
    rename5  = {c: f"{side}_{c}" for c in roll5.columns  if c not in ["date", "team"]}
    rename10 = {c: f"{side}_{c}" for c in roll10.columns if c not in ["date", "team"]}
    df = df.merge(roll5.rename(columns=rename5),   left_on=["date", team_col], right_on=["date", "team"], how="left").drop(columns="team")
    df = df.merge(roll10.rename(columns=rename10), left_on=["date", team_col], right_on=["date", "team"], how="left").drop(columns="team")

df = df.dropna(subset=[f"h_rating_r{WINDOW_SHORT}"])
print(f"  Rolling dodane. Meczów z pełnymi danymi: {len(df):,}")


# ─────────────────────────────────────────────────────────────────
# KROK 3: CECHY RÓŻNICOWE I REMISOGENNE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  KROK 3: Cechy różnicowe, bezwzględne i Draw-Detection signals")
print("=" * 65)

for stat in STATS:
    for w in [WINDOW_SHORT, WINDOW_LONG]:
        h, a = f"h_{stat}_r{w}", f"a_{stat}_r{w}"
        # Różnica liniowa: dodatnia = przewaga gospodarza
        df[f"diff_{stat}{w}"] = df[h] - df[a]
        # Różnica bezwzględna: bliska 0 = zbliżone siły → remis prawdopodobny
        df[f"abs_{stat}{w}"]  = np.abs(df[f"diff_{stat}{w}"])

# Draw-detection signals (monotonicznie malejące ze wzrostem różnicy)
df["draw_rating5"]  = 1 / (1 + df["abs_rating5"])
df["draw_goals5"]   = 1 / (1 + df["abs_goals5"])
df["draw_rating10"] = 1 / (1 + df["abs_rating10"])
df["draw_goals10"]  = 1 / (1 + df["abs_goals10"])
df["draw_xg5"]      = 1 / (1 + df["abs_xg5"])

# Stosunki (ratios) — niż bezwzględne różnice lepiej oddają proporcję sił
df["rate_ratio"]  = df[f"h_rating_r{WINDOW_SHORT}"] / (df[f"a_rating_r{WINDOW_SHORT}"] + 1e-6)
df["goals_ratio"] = df[f"h_goals_r{WINDOW_SHORT}"]  / (df[f"a_goals_r{WINDOW_SHORT}"]  + 1e-6)
df["xg_ratio"]    = df[f"h_xg_r{WINDOW_SHORT}"]     / (df[f"a_xg_r{WINDOW_SHORT}"]     + 1e-6)

print("  Cechy różnicowe, draw-signals i ratios dodane.")


# ─────────────────────────────────────────────────────────────────
# KROK 4: KODOWANIE DRUŻYN I LIGI
# ─────────────────────────────────────────────────────────────────
le_team = LabelEncoder()
le_team.fit(pd.concat([df["home_team"], df["away_team"]]).unique())
df["h_id"] = le_team.transform(df["home_team"])
df["a_id"] = le_team.transform(df["away_team"])

le_league = LabelEncoder()
df["league_id"] = le_league.fit_transform(df["league_division"].astype(str))


# ─────────────────────────────────────────────────────────────────
# KROK 5: BUDOWANIE MACIERZY CECH
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  KROK 5: Budowanie macierzy cech")
print("=" * 65)

FEATURE_COLS = ["h_id", "a_id", "league_id"]
for stat in STATS:
    for w in [WINDOW_SHORT, WINDOW_LONG]:
        FEATURE_COLS += [
            f"h_{stat}_r{w}",    # forma gospodarz
            f"a_{stat}_r{w}",    # forma gość
            f"diff_{stat}{w}",   # różnica liniowa
            f"abs_{stat}{w}",    # różnica bezwzględna
        ]
FEATURE_COLS += [
    "draw_rating5", "draw_goals5", "draw_rating10",
    "draw_goals10", "draw_xg5",
    "rate_ratio", "goals_ratio", "xg_ratio",
]

X = df[FEATURE_COLS].copy()
y = df["target"].values
print(f"  Liczba cech: {len(FEATURE_COLS)}")
print(f"  Meczów: {len(X):,}")


# ─────────────────────────────────────────────────────────────────
# KROK 6: CHRONOLOGICZNY PODZIAŁ TRAIN / TEST (80/20)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  KROK 6: Chronologiczny podział Train/Test (80/20)")
print("  ⚠ NIE losowy — zapobiega wyciekowi przyszłych danych!")
print("=" * 65)

split_idx = int(len(X) * TRAIN_SPLIT)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates = df["date"].values

print(f"  Treningowy: {len(X_train):,} | {dates[0].astype('datetime64[D]')} → {dates[split_idx-1].astype('datetime64[D]')}")
print(f"  Testowy:    {len(X_test):,} | {dates[split_idx].astype('datetime64[D]')} → {dates[-1].astype('datetime64[D]')}")


# ─────────────────────────────────────────────────────────────────
# KROK 7: PREPROCESSING (Imputer + Scaler)
# ─────────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
scaler  = StandardScaler()

X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_scaled  = scaler.transform(imputer.transform(X_test))

# Wagi klas
sw_train = compute_sample_weight("balanced", y_train)


# ─────────────────────────────────────────────────────────────────
# KROK 8: MODELE BAZOWE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  KROK 8: Trenowanie modeli bazowych")
print("=" * 65)

# Model #1: Logistic Regression
lr_base = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE,
    class_weight="balanced",  # kompensacja rzadszych remisów
    C=0.3,                    # optymalne C (silna regularyzacja → unika przeuczenia)
)
lr_base.fit(X_train_scaled, y_train)
lr_preds = lr_base.predict(X_test_scaled)
lr_acc   = accuracy_score(y_test, lr_preds)
lr_f1    = f1_score(y_test, lr_preds, average="macro")
print(f"\n  [1] Logistic Regression  →  Accuracy: {lr_acc:.4f} | F1 Macro: {lr_f1:.4f}")
print(classification_report(y_test, lr_preds, target_names=["Away (0)", "Draw (1)", "Home (2)"], digits=4))

# Model #2: Random Forest
rf_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf_base.fit(X_train_scaled, y_train)
rf_preds = rf_base.predict(X_test_scaled)
rf_acc   = accuracy_score(y_test, rf_preds)
rf_f1    = f1_score(y_test, rf_preds, average="macro")
print(f"  [2] Random Forest        →  Accuracy: {rf_acc:.4f} | F1 Macro: {rf_f1:.4f}")
print(classification_report(y_test, rf_preds, target_names=["Away (0)", "Draw (1)", "Home (2)"], digits=4))


# ─────────────────────────────────────────────────────────────────
# KROK 9: WYBÓR I STROJENIE (RandomizedSearchCV → f1_macro)
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  KROK 9: Strojenie hiperparametrów (metryka: f1_macro)")
print("=" * 65)

if lr_f1 >= rf_f1:
    best_name = "Logistic Regression"
    best_base_f1 = lr_f1
    model_to_tune = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced")
    param_dist = {
        "C":      loguniform(1e-3, 10),
        "solver": ["lbfgs", "saga"],
        "penalty": ["l2"],
    }
    fit_params = {}
else:
    best_name = "Random Forest"
    best_base_f1 = rf_f1
    model_to_tune = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced")
    param_dist = {
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [8, 10, 12, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf":  [1, 2, 3, 5],
        "max_features":    ["sqrt", "log2", 0.4, 0.6],
    }
    fit_params = {}

print(f"\n  Wybrano do strojenia: {best_name} (F1 bazowy={best_base_f1:.4f})")
print("  Trwa RandomizedSearchCV (n_iter=100, cv=3)...\n")

search = RandomizedSearchCV(
    estimator           = model_to_tune,
    param_distributions = param_dist,
    n_iter              = 100,        # 100 kombinacji hiperparametrów
    cv                  = 3,
    scoring             = "f1_macro", # optymalizacja pod f1 ALL klas (w tym remisy!)
    random_state        = RANDOM_STATE,
    n_jobs              = -1,
    verbose             = 1,
    refit               = True,
)
search.fit(X_train_scaled, y_train, **fit_params)


# ─────────────────────────────────────────────────────────────────
# KROK 10: WYNIKI KOŃCOWE
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  KROK 10: WYNIKI KOŃCOWE — model dostrojony")
print("=" * 65)

best_preds = search.best_estimator_.predict(X_test_scaled)
best_acc   = accuracy_score(y_test, best_preds)
best_f1    = f1_score(y_test, best_preds, average="macro")

print(f"\n  Najlepsze parametry: {search.best_params_}")
print(f"  CV F1 Macro (val): {search.best_score_:.4f}")

print("\n  ╔══ CLASSIFICATION REPORT (zbiór testowy) ═══════════════╗")
print(f"  Model: {best_name} (po strojeniu)")
print(classification_report(
    y_test, best_preds,
    target_names=["Away (0)", "Draw (1)", "Home (2)"],
    digits=4,
))
print("  ╚═════════════════════════════════════════════════════════╝")


# ─────────────────────────────────────────────────────────────────
# PODSUMOWANIE
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("  PODSUMOWANIE PORÓWNAWCZE")
print("=" * 65)
rows = [
    ("Logistic Regression (bazowy)",   lr_acc,   lr_f1),
    ("Random Forest (bazowy)",         rf_acc,   rf_f1),
    (f"{best_name} (po strojeniu)",    best_acc, best_f1),
]
best_f1_all = max(r[2] for r in rows)
print(f"\n  {'Model':<38} {'Accuracy':>10} {'F1 Macro':>10}")
print(f"  {'─'*60}")
for name, acc, f1 in rows:
    mark = "  ◄ NAJLEPSZY" if f1 == best_f1_all else ""
    print(f"  {name:<38} {acc:>10.4f} {f1:>10.4f}{mark}")

if hasattr(search.best_estimator_, "feature_importances_"):
    print("\n" + "=" * 65)
    print("  TOP 15 NAJWAŻNIEJSZYCH CECH")
    print("=" * 65)
    fi = pd.Series(
        search.best_estimator_.feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False).head(15)
    for feat, imp in fi.items():
        bar = "█" * int(imp * 400)
        print(f"  {feat:<35} {imp:.4f}  {bar}")

print("\n" + "=" * 65)
print("  Pipeline zakończony.")
print("=" * 65)