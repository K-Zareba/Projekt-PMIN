import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Wczytanie danych z kodowaniem 'latin-1' (często rozwiązuje problem bajtów np. ze znakami specjalnymi/stopniami)

# Używamy sep=None i engine='python', aby pandas sam wykrył separator (przecinek, średnik, tabulacja)
df = pd.read_csv('scraped_dataset.csv', encoding='latin-1', sep=None, engine='python', on_bad_lines='warn')


# Sprawdzenie braków i usunięcie niepotrzebnych kolumn
missing_ratios = df.isnull().mean()
# Usuwamy kolumny, w których brakuje ponad 50% danych
cols_to_drop_missing = missing_ratios[missing_ratios > 0.5].index.tolist()
df_cleaned = df.drop(columns=cols_to_drop_missing)

# Usuwamy ręcznie kolumny identyfikacyjne/tekstowe, które nie przydadzą się łatwo do modelu ML
cols_to_drop_manual = [
    'match_id', 'league_division', 'round', 'date', 'kickoff_time',
    'stadium', 'city', 'home_formation', 'away_formation', 'attendance'
]
df_cleaned = df_cleaned.drop(columns=[c for c in cols_to_drop_manual if c in df_cleaned.columns])

print(f"Liczba kolumn po czyszczeniu: {len(df_cleaned.columns)}")

# Definiowanie zmiennej docelowej (jeśli są bramki)
def get_result(row):
    if pd.isna(row.get('home_goals')) or pd.isna(row.get('away_goals')):
        return np.nan
    if row['home_goals'] > row['away_goals']: return 'Home (1)'
    elif row['home_goals'] == row['away_goals']: return 'Draw (X)'
    else: return 'Away (2)'

df_cleaned['match_outcome'] = df_cleaned.apply(get_result, axis=1)

# Usunięcie wierszy z brakiem wyniku
df_cleaned = df_cleaned.dropna(subset=['match_outcome'])

# Wykres 1: Balans klas
plt.figure(figsize=(8, 5))
sns.countplot(x='match_outcome', data=df_cleaned, palette='viridis', order=['Home (1)', 'Draw (X)', 'Away (2)'])
plt.title('Rozkład wyników meczów (Balans klas)')
plt.xlabel('Wynik')
plt.ylabel('Liczba meczów')
plt.savefig('class_balance.png', bbox_inches='tight')
plt.close()

# Wykres 2: Macierz korelacji wybranych statystyk
num_cols = df_cleaned.select_dtypes(include=[np.number]).columns
cols_for_corr = [c for c in [
    'home_goals', 'away_goals', 'home_shots_total', 'away_shots_total',
    'home_possession_pct', 'away_possession_pct', 'home_corners', 'away_corners',
    'home_team_rating', 'away_team_rating'
] if c in num_cols]

plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned[cols_for_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Macierz korelacji kluczowych statystyk')
plt.savefig('correlation_matrix.png', bbox_inches='tight')
plt.close()

# Zapisanie wyczyszczonego pliku (dla użytkownika)
df_cleaned.to_csv('cleaned_dataset.csv', index=False)
print("Wykresy i plik 'cleaned_dataset.csv' zostały wygenerowane.")