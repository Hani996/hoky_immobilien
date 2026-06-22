# %% [markdown]
# # Immobilienpreis-Projekt: Statistische Analyse & Visualisierung
# Agiles Programmierprojekt – intoCODE / InterGeeks
#
# Dieses Skript deckt folgende Aufgabenbereiche ab:
# 1. Datenbereinigung (Vorbereitung für Statistik)
# 2. Deskriptive Statistik und Datenverständnis
# 3. Einflussanalyse
# 4. Zusammenhangsanalyse
# 5. Hypothesentests und Inferenzstatistik
#
# Hinweis: Modellierung (Linear Regression / Random Forest / XGBoost)
# läuft in separaten Skripten – hier nur die statistische Vorbereitung
# und der finale Modellvergleich (Abschnitt 6).

# %%
# ----------------------------------------------------------------------
# 0. IMPORTS & SETUP
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Plot-Stil global festlegen
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.titleweight"] = "bold"

# Output-Ordner für gespeicherte Grafiken (für PowerPoint-Export)
import os
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(fig, name):
    """Speichert jede Grafik zusätzlich als PNG für die Präsentation."""
    fig.savefig(f"{OUTPUT_DIR}/{name}.png", dpi=150, bbox_inches="tight")


# %%
# ----------------------------------------------------------------------
# 1. DATEN LADEN
# ----------------------------------------------------------------------
df_raw = pd.read_csv("../data/price_clean.csv")  # Pfad ggf. anpassen
print("Rohdaten-Shape:", df_raw.shape)
df_raw.head()


# %%
# ----------------------------------------------------------------------
# 2. DATENBEREINIGUNG (Vorbereitung)
# ----------------------------------------------------------------------
# Schritt 2.1: Kopie erstellen, damit Rohdaten erhalten bleiben
df = df_raw.copy()

# Schritt 2.2: obj_regio1 ist konstant ("Niedersachsen") -> keine Information, entfernen
print("Eindeutige Werte obj_regio1:", df["obj_regio1"].unique())
df = df.drop(columns=["obj_regio1"])

# Schritt 2.3: Zielleck (Data Leakage) vermeiden
# obj_purchasePrice_per_qm ist direkt aus der Zielvariable berechnet
# (purchasePrice / livingSpace) -> darf NICHT als Feature in ML verwendet werden.
# Wir behalten sie hier nur für die deskriptive Analyse, markieren sie aber klar.
LEAKAGE_COL = "obj_purchasePrice_per_qm"

# Schritt 2.4: Unplausible / fehlerhafte Zeilen entfernen
# a) Wohnfläche = 0 -> führt zu inf bei price_per_qm, technisch unmöglich
n_before = len(df)
df = df[df["obj_livingSpace"] > 0]
print(f"Entfernt wegen livingSpace <= 0: {n_before - len(df)} Zeilen")

# b) Kaufpreis unrealistisch niedrig (< 5000 EUR) -> vermutlich Dateneingabefehler
#    oder Sonderfälle (z.B. Erbpacht, symbolischer Preis)
n_before = len(df)
df = df[df["obj_purchasePrice"] >= 5000]
print(f"Entfernt wegen purchasePrice < 5000 EUR: {n_before - len(df)} Zeilen")

# c) Nach der Bereinigung: keine inf/NaN mehr in der Zielvariable prüfen
assert not np.isinf(df[LEAKAGE_COL]).any(), "Es gibt noch inf-Werte!"
print("Bereinigte Daten-Shape:", df.shape)


# %%
# Schritt 2.5: Fehlende Werte dokumentieren (NICHT blind auffüllen,
# sondern bewusst je Spalte entscheiden)
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(1)
missing_table = pd.DataFrame({"fehlend": missing, "prozent": missing_pct})
missing_table = missing_table[missing_table["fehlend"] > 0]
print(missing_table)

# Visualisierung: fehlende Werte
fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(x=missing_table["prozent"], y=missing_table.index, hue=missing_table.index,
            ax=ax, palette="rocket", legend=False)
ax.set_title("Anteil fehlender Werte je Spalte")
ax.set_xlabel("Fehlend (%)")
ax.set_ylabel("")
plt.tight_layout()
save_fig(fig, "01_fehlende_werte")
plt.show()

# %% [markdown]
# **Entscheidungen zu fehlenden Werten:**
# - `obj_houseNumber`, `obj_street`: über 90% fehlend, keine Modellierungs-relevanz
#   -> Spalten werden für die Analyse verworfen (Adressdetails nicht nötig,
#   `geo_plz` / `geo_krs` / `obj_regio3` liefern bereits Lageinformation).
# - `obj_firingTypes`: fehlende Werte -> Kategorie "unknown" (kein Heizungstyp angegeben)
# - `obj_condition`: fehlende Werte -> Kategorie "unknown"
# - `obj_telekomUploadSpeed` / `obj_telekomDownloadSpeed`: fehlende Werte ->
#   vermutlich kein Internetangebot verfügbar -> mit 0 auffüllen + Flag-Spalte

# %%
df = df.drop(columns=["obj_houseNumber", "obj_street"])

df["obj_firingTypes"] = df["obj_firingTypes"].fillna("unknown")
df["obj_condition"] = df["obj_condition"].fillna("unknown")

df["obj_hasInternetData"] = df["obj_telekomUploadSpeed"].notna().astype(int)
df["obj_telekomUploadSpeed"] = df["obj_telekomUploadSpeed"].fillna(0)
df["obj_telekomDownloadSpeed"] = df["obj_telekomDownloadSpeed"].fillna(0)

# obj_telekomInternetProductAvailable ist boolean-artig mit fehlenden Werten
df["obj_telekomInternetProductAvailable"] = (
    df["obj_telekomInternetProductAvailable"].fillna("unknown").astype(str)
)

print("Verbleibende fehlende Werte:\n", df.isnull().sum()[df.isnull().sum() > 0])


# %%
# Schritt 2.6: Heizungstypen gruppieren (60 verschiedene Ausprägungen -> zu granular)
def group_firing_type(value):
    value = str(value).lower()
    if value == "unknown":
        return "unknown"
    if "gas" in value:
        return "gas"
    if "oil" in value:
        return "oil"
    if "district_heating" in value or "local_heating" in value:
        return "fernwaerme"
    if "solar" in value or "geothermal" in value or "environmental" in value or "pellet" in value or "wood" in value:
        return "erneuerbar"
    if "electricity" in value:
        return "strom"
    return "sonstige"

df["firingType_grouped"] = df["obj_firingTypes"].apply(group_firing_type)
print(df["firingType_grouped"].value_counts())


# %%
# Finaler bereinigter Datensatz - Übersicht
print("Finale Shape:", df.shape)
print("\nSpalten:", list(df.columns))
df.to_csv("price_cleaned_for_analysis.csv", index=False)
print("\nGespeichert als price_cleaned_for_analysis.csv")


# %% [markdown]
# ----------------------------------------------------------------------
# # 3. DESKRIPTIVE STATISTIK UND DATENVERSTÄNDNIS
# ----------------------------------------------------------------------
# Lageparameter, Streuungsmaße, Verteilungen, Ausreißer, Datenqualität

# %%
# 3.1 Lage- und Streuungsmaße für die wichtigsten numerischen Variablen
numeric_cols = [
    "obj_purchasePrice",
    "obj_purchasePrice_per_qm",
    "obj_livingSpace",
    "obj_noRooms",
    "obj_yearConstructed",
]

desc_table = df[numeric_cols].describe().T
desc_table["median"] = df[numeric_cols].median()
desc_table["variance"] = df[numeric_cols].var()
desc_table["IQR"] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
desc_table["skew"] = df[numeric_cols].skew()  # Schiefe der Verteilung
desc_table = desc_table.round(2)
print(desc_table)


# %%
# 3.2 Verteilung der Zielvariable: Kaufpreis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df["obj_purchasePrice"], bins=50, kde=True, ax=axes[0], color="teal")
axes[0].set_title("Verteilung: Kaufpreis (EUR)")
axes[0].set_xlabel("Kaufpreis (EUR)")

# Log-Skala, da Immobilienpreise meist rechtsschief sind
sns.histplot(np.log1p(df["obj_purchasePrice"]), bins=50, kde=True, ax=axes[1], color="darkorange")
axes[1].set_title("Verteilung: log(Kaufpreis)")
axes[1].set_xlabel("log(1 + Kaufpreis)")

plt.tight_layout()
save_fig(fig, "02_verteilung_kaufpreis")
plt.show()

print(f"Schiefe (Skewness) Kaufpreis: {df['obj_purchasePrice'].skew():.2f}")
print(f"Schiefe (Skewness) log(Kaufpreis): {np.log1p(df['obj_purchasePrice']).skew():.2f}")


# %%
# 3.3 Verteilung weiterer zentraler Variablen
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df["obj_livingSpace"], bins=50, kde=True, ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("Wohnfläche (m²)")

sns.histplot(df["obj_noRooms"], bins=20, kde=False, ax=axes[0, 1], color="indianred")
axes[0, 1].set_title("Anzahl Zimmer")

sns.histplot(df["obj_yearConstructed"], bins=50, kde=True, ax=axes[1, 0], color="seagreen")
axes[1, 0].set_title("Baujahr")

sns.histplot(df["obj_purchasePrice_per_qm"], bins=50, kde=True, ax=axes[1, 1], color="purple")
axes[1, 1].set_title("Kaufpreis pro m²")

plt.tight_layout()
save_fig(fig, "03_verteilungen_uebersicht")
plt.show()


# %%
# 3.4 Ausreißer-Identifikation mittels IQR-Methode (Boxplots)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(y=df["obj_purchasePrice"], ax=axes[0], color="teal")
axes[0].set_title("Kaufpreis – Boxplot")

sns.boxplot(y=df["obj_livingSpace"], ax=axes[1], color="steelblue")
axes[1].set_title("Wohnfläche – Boxplot")

sns.boxplot(y=df["obj_purchasePrice_per_qm"], ax=axes[2], color="purple")
axes[2].set_title("Preis/m² – Boxplot")

plt.tight_layout()
save_fig(fig, "04_boxplots_ausreisser")
plt.show()


def count_iqr_outliers(series):
    """Zählt Ausreißer nach der 1.5*IQR-Regel."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

for col in ["obj_purchasePrice", "obj_livingSpace", "obj_purchasePrice_per_qm", "obj_noRooms"]:
    n_out = count_iqr_outliers(df[col])
    print(f"{col}: {n_out} Ausreißer ({n_out/len(df)*100:.1f}%) nach 1.5*IQR-Regel")


# %%
# 3.5 Kategoriale Variablen: Häufigkeitsverteilungen
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

df["obj_condition"].value_counts().plot(kind="bar", ax=axes[0, 0], color="darkcyan")
axes[0, 0].set_title("Objektzustand (obj_condition)")
axes[0, 0].tick_params(axis="x", rotation=60)

df["firingType_grouped"].value_counts().plot(kind="bar", ax=axes[0, 1], color="darkorange")
axes[0, 1].set_title("Heizungstyp (gruppiert)")
axes[0, 1].tick_params(axis="x", rotation=45)

df["obj_newlyConst"].value_counts().plot(kind="bar", ax=axes[1, 0], color="seagreen")
axes[1, 0].set_title("Neubau (obj_newlyConst)")
axes[1, 0].tick_params(axis="x", rotation=0)

df["obj_cellar"].value_counts().plot(kind="bar", ax=axes[1, 1], color="indianred")
axes[1, 1].set_title("Keller vorhanden (obj_cellar)")
axes[1, 1].tick_params(axis="x", rotation=0)

plt.tight_layout()
save_fig(fig, "05_kategoriale_verteilungen")
plt.show()


# %%
# 3.6 Top-Regionen nach Anzahl Inseraten (geo_krs)
fig, ax = plt.subplots(figsize=(10, 8))
top_krs = df["geo_krs"].value_counts().head(15)
sns.barplot(x=top_krs.values, y=top_krs.index, hue=top_krs.index,
            ax=ax, palette="mako", legend=False)
ax.set_title("Top 15 Landkreise nach Anzahl Inserate")
ax.set_xlabel("Anzahl Inserate")
plt.tight_layout()
save_fig(fig, "06_top_landkreise")
plt.show()


# %% [markdown]
# **Zusammenfassung Datenqualität (für die Präsentation):**
# - Ursprünglich 5055 Zeilen, nach Bereinigung: siehe `df.shape`
# - Zielvariable `obj_purchasePrice` ist stark rechtsschief -> log-Transformation
#   für Regressionsmodelle empfehlenswert
# - Ausreißer vorhanden, aber plausibel (teure/große Immobilien) -> nicht
#   pauschal entfernen, sondern bei Modellierung robuste Verfahren nutzen
# - `obj_condition` und `obj_firingTypes` hatten viele fehlende Werte ->
#   als "unknown"-Kategorie behandelt, um Informationsverlust zu vermeiden


# %% [markdown]
# ----------------------------------------------------------------------
# # 4. EINFLUSSANALYSE
# ----------------------------------------------------------------------
# Einfluss einzelner Variablen auf die Zielvariable (Kaufpreis)

# %%
# 4.1 Einfluss des Objektzustands auf den Preis/m²
fig, ax = plt.subplots(figsize=(11, 6))
order = df.groupby("obj_condition")["obj_purchasePrice_per_qm"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="obj_condition", y="obj_purchasePrice_per_qm", order=order,
            hue="obj_condition", legend=False, ax=ax, palette="viridis")
ax.set_title("Einfluss des Objektzustands auf Preis/m²")
ax.set_xlabel("Objektzustand")
ax.set_ylabel("Kaufpreis pro m² (EUR)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
save_fig(fig, "07_einfluss_zustand")
plt.show()


# %%
# 4.2 Einfluss von Neubau (obj_newlyConst) auf den Preis/m²
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(data=df, x="obj_newlyConst", y="obj_purchasePrice_per_qm",
            hue="obj_newlyConst", legend=False, ax=ax, palette="Set2")
ax.set_title("Einfluss Neubau-Status auf Preis/m²")
ax.set_xlabel("Neubau (n=nein, y=ja)")
ax.set_ylabel("Kaufpreis pro m² (EUR)")
plt.tight_layout()
save_fig(fig, "08_einfluss_neubau")
plt.show()

print(df.groupby("obj_newlyConst")["obj_purchasePrice_per_qm"].agg(["mean", "median", "std"]))


# %%
# 4.3 Einfluss des Baujahrs auf den Preis (Streudiagramm + Trend)
fig, ax = plt.subplots(figsize=(10, 6))
sns.regplot(
    data=df, x="obj_yearConstructed", y="obj_purchasePrice_per_qm",
    scatter_kws={"alpha": 0.3, "s": 15}, line_kws={"color": "red"}, ax=ax
)
ax.set_title("Zusammenhang Baujahr und Preis/m²")
ax.set_xlabel("Baujahr")
ax.set_ylabel("Kaufpreis pro m² (EUR)")
plt.tight_layout()
save_fig(fig, "09_einfluss_baujahr")
plt.show()


# %%
# 4.4 Einfluss der Wohnfläche auf den Gesamtpreis
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df, x="obj_livingSpace", y="obj_purchasePrice",
    hue="obj_condition", alpha=0.5, ax=ax, legend=False
)
ax.set_title("Wohnfläche vs. Kaufpreis (eingefärbt nach Zustand)")
ax.set_xlabel("Wohnfläche (m²)")
ax.set_ylabel("Kaufpreis (EUR)")
plt.tight_layout()
save_fig(fig, "10_einfluss_wohnflaeche")
plt.show()


# %%
# 4.5 Regionaler Einfluss: Top 10 teuerste vs. günstigste Landkreise (Preis/m²)
krs_median = df.groupby("geo_krs")["obj_purchasePrice_per_qm"].median().sort_values()
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

krs_median.head(10).plot(kind="barh", ax=axes[0], color="steelblue")
axes[0].set_title("10 günstigste Landkreise (Median Preis/m²)")
axes[0].set_xlabel("EUR/m²")

krs_median.tail(10).plot(kind="barh", ax=axes[1], color="indianred")
axes[1].set_title("10 teuerste Landkreise (Median Preis/m²)")
axes[1].set_xlabel("EUR/m²")

plt.tight_layout()
save_fig(fig, "11_regionaler_einfluss")
plt.show()


# %% [markdown]
# ----------------------------------------------------------------------
# # 5. ZUSAMMENHANGSANALYSE
# ----------------------------------------------------------------------
# Stärke und Richtung der Zusammenhänge zwischen Variablen

# %%
# 5.1 Korrelationsmatrix (Pearson) für numerische Variablen
corr_cols = [
    "obj_purchasePrice", "obj_livingSpace", "obj_noRooms",
    "obj_yearConstructed", "obj_telekomUploadSpeed", "obj_telekomDownloadSpeed",
]
corr_matrix = df[corr_cols].corr(method="pearson")

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Korrelationsmatrix (Pearson) numerischer Variablen")
plt.tight_layout()
save_fig(fig, "12_korrelationsmatrix")
plt.show()


# %%
# 5.2 Zusammenhang kategorial -> numerisch: Eta-Quadrat (Effektstärke aus ANOVA)
def eta_squared_anova(df, cat_col, num_col):
    """Berechnet Eta² als Maß für den Zusammenhang zwischen einer
    kategorialen und einer numerischen Variable (Anteil erklärter Varianz)."""
    groups = [g[num_col].dropna().values for _, g in df.groupby(cat_col)]
    groups = [g for g in groups if len(g) > 1]
    f_stat, p_value = stats.f_oneway(*groups)
    grand_mean = df[num_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum((df[num_col] - grand_mean) ** 2)
    eta_sq = ss_between / ss_total
    return eta_sq, f_stat, p_value

for cat_col in ["obj_condition", "firingType_grouped", "obj_newlyConst", "obj_cellar"]:
    eta_sq, f_stat, p_val = eta_squared_anova(df, cat_col, "obj_purchasePrice_per_qm")
    print(f"{cat_col:25s} | Eta² = {eta_sq:.3f} | F = {f_stat:8.2f} | p = {p_val:.4f}")


# %%
# 5.3 Zusammenhang zwischen zwei kategorialen Variablen: Cramér's V
def cramers_v(x, y):
    """Cramér's V: Zusammenhangsmaß für zwei kategoriale Variablen (0=kein, 1=stark)."""
    contingency = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    phi2 = chi2 / n
    r, k = contingency.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

v = cramers_v(df["obj_condition"], df["firingType_grouped"])
print(f"Cramér's V (obj_condition vs. firingType_grouped): {v:.3f}")


# %% [markdown]
# ----------------------------------------------------------------------
# # 6. HYPOTHESENTESTS UND INFERENZSTATISTIK
# ----------------------------------------------------------------------

# %%
# 6.1 Hypothese: Neubauten (obj_newlyConst = 'y') haben einen anderen
# Preis/m² als Bestandsimmobilien.
# H0: Kein Unterschied im Mittelwert. H1: Es gibt einen Unterschied.
group_new = df.loc[df["obj_newlyConst"] == "y", "obj_purchasePrice_per_qm"]
group_old = df.loc[df["obj_newlyConst"] == "n", "obj_purchasePrice_per_qm"]

# Voraussetzung prüfen: Normalverteilung (Shapiro, auf Stichprobe wegen n)
shapiro_new = stats.shapiro(group_new.sample(min(500, len(group_new)), random_state=42))
shapiro_old = stats.shapiro(group_old.sample(min(500, len(group_old)), random_state=42))
print(f"Shapiro-Wilk Neubau: p = {shapiro_new.pvalue:.4f}")
print(f"Shapiro-Wilk Bestand: p = {shapiro_old.pvalue:.4f}")
# -> p < 0.05 in der Regel -> keine Normalverteilung -> nicht-parametrischer Test

# Daher: Mann-Whitney-U-Test (statt t-Test)
u_stat, p_value = stats.mannwhitneyu(group_new, group_old, alternative="two-sided")
print(f"\nMann-Whitney-U-Test: U = {u_stat:.1f}, p = {p_value:.6f}")
if p_value < 0.05:
    print("-> Signifikanter Unterschied zwischen Neubau und Bestand (p < 0.05)")
else:
    print("-> Kein signifikanter Unterschied (p >= 0.05)")


# %%
# 6.2 Hypothese: Immobilien mit Keller (obj_cellar = 'y') sind teurer pro m²
group_cellar = df.loc[df["obj_cellar"] == "y", "obj_purchasePrice_per_qm"]
group_no_cellar = df.loc[df["obj_cellar"] == "n", "obj_purchasePrice_per_qm"]

u_stat2, p_value2 = stats.mannwhitneyu(group_cellar, group_no_cellar, alternative="two-sided")
print(f"Mann-Whitney-U-Test (Keller): U = {u_stat2:.1f}, p = {p_value2:.6f}")
print(f"Median mit Keller: {group_cellar.median():.0f} EUR/m²")
print(f"Median ohne Keller: {group_no_cellar.median():.0f} EUR/m²")


# %%
# 6.3 Hypothese: Der Objektzustand hat einen signifikanten Einfluss auf den
# Preis/m² (mehr als 2 Gruppen -> Kruskal-Wallis statt ANOVA, da nicht normalverteilt)
groups_condition = [g["obj_purchasePrice_per_qm"].values for _, g in df.groupby("obj_condition")]
h_stat, p_value3 = stats.kruskal(*groups_condition)
print(f"Kruskal-Wallis-Test (obj_condition): H = {h_stat:.2f}, p = {p_value3:.6f}")
if p_value3 < 0.05:
    print("-> Der Objektzustand hat einen signifikanten Einfluss auf den Preis/m²")


# %% [markdown]
# ----------------------------------------------------------------------
# # 7. MODELLVERGLEICH (Linear Regression -> Random Forest -> XGBoost)
# ----------------------------------------------------------------------
# Hinweis: Diese Sektion ist ein Template für die Zusammenführung der
# Ergebnisse aus deinen separaten Modellierungs-Skripten. Trage die
# tatsächlichen Metriken aus deinen vier LR-Phasen, zwei RF-Phasen
# und dem finalen XGBoost-Modell hier ein.

# %%
model_results = pd.DataFrame({
    "Modell": [
        "Linear Regression (Phase 1)",
        "Linear Regression (Phase 2)",
        "Linear Regression (Phase 3)",
        "Linear Regression (Phase 4)",
        "Random Forest (Phase 1)",
        "Random Forest (Phase 2)",
        "XGBoost (Final)",
    ],
    # TODO: Werte aus deinen Notebooks/Skripten hier eintragen
    "RMSE": [None, None, None, None, None, None, None],
    "MAE":  [None, None, None, None, None, None, None],
    "R2":   [None, None, None, None, None, None, None],
})
print(model_results)

# Sobald die Werte eingetragen sind, folgendes Diagramm nutzen:
# fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# for ax, metric in zip(axes, ["RMSE", "MAE", "R2"]):
#     sns.barplot(data=model_results, x="Modell", y=metric, ax=ax, palette="viridis")
#     ax.set_title(f"Modellvergleich: {metric}")
#     ax.tick_params(axis="x", rotation=75)
# plt.tight_layout()
# save_fig(fig, "13_modellvergleich")
# plt.show()

print("\nAlle Grafiken wurden im Ordner 'plots/' gespeichert (PNG, 150 dpi) –")
print("bereit zum Einfügen in die PowerPoint-Präsentation.")
