import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

raw_data = pd.read_csv("recipeData.csv", encoding = "latin-1", na_values=["N/A","NA",""])

print("Wymiary zbioru",raw_data.shape)
print(raw_data.info())

pd.set_option("display.max_rows", None)


print("\nLiczba unikalnych stylów:", raw_data["Style"].nunique())
print("\nLiczebność stylów:")
print(raw_data["Style"].value_counts())

print("\nTop 12:")
print(raw_data["Style"].value_counts().head(12))

TOP_N = 12
style_counts = raw_data["Style"].value_counts()
top_styles = style_counts.head(TOP_N).index
data = raw_data[raw_data["Style"].isin(top_styles)].copy()
tabela = (
    data["Style"]
    .value_counts()
    .rename_axis("Styl")
    .reset_index(name="Liczba receptur")
)
tabela["Udział (%)"] = (
    tabela["Liczba receptur"] / tabela["Liczba receptur"].sum() * 100
).round(2)
suma = pd.DataFrame({
    "Styl": ["Razem"],
    "Liczba receptur": [tabela["Liczba receptur"].sum()],
    "Udział (%)": [100.0],
})
tabela = pd.concat([tabela, suma], ignore_index=True)
print(tabela.to_string(index=False))

counts = data["Style"].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=counts.values, y=counts.index, color="#2c5f2d")
plt.xlabel("Liczba receptur")
plt.ylabel("Styl piwa")
plt.title(f"Liczebność Top {TOP_N} stylów piwa")
for i, v in enumerate(counts.values):
    plt.text(v + max(counts.values) * 0.01, i, f"{v:,}", va="center", fontsize=9)
plt.tight_layout()
# plt.show()


# Braki danych w zbiorze Top 12
braki = (
    data.isna()
    .sum()
    .sort_values(ascending=False)
    .to_frame(name="Liczba braków")
)
braki["Odsetek (%)"] = (braki["Liczba braków"] / len(data) * 100).round(2)

print("\nBraki danych (od największej liczby NA):")
print(braki[braki["Liczba braków"] > 0].to_string())
do_usuniecia = braki[braki["Liczba braków"] > 0].index.tolist()
do_usuniecia.extend(['BeerID', 'URL', 'StyleID'])
data_clean = data.drop(columns=do_usuniecia)
print("Kolumny po usunięciu:", data_clean.columns.tolist())

y = data_clean["Style"].copy()          
X = data_clean.drop(columns=["Style"])

opis = X.select_dtypes(include="number").describe(percentiles=[0.25, 0.5, 0.75]).T

opis = opis.rename(columns={
    "50%": "mediana",
    "25%": "Q1",
    "75%": "Q3",
    "mean": "średnia",
    "std": "odch. std.",
    "min": "min",
    "max": "max",
})

opis = opis[[ "średnia", "mediana", "odch. std.", "min", "Q1", "Q3", "max"]].round(4)
print(opis.to_string())

cols = ["OG", "FG", "ABV", "IBU", "Color", "BoilTime", "BoilSize", "Size(L)", "Efficiency"]

for i in range(0, len(cols), 3):
    batch = cols[i:i+3]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, batch):
        sns.histplot(X[col], bins=40, kde=False, ax=ax, color="#3a6b35")
        ax.set_title(f"{col} (skośność={X[col].skew():.2f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Liczba")
    
    fig.tight_layout()
# plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, col in zip(axes, ["BrewMethod", "SugarScale"]):
    pct = data[col].value_counts(normalize=True).mul(100).round(2)
    sns.barplot(x=pct.values, y=pct.index, ax=ax, color="#2c5f2d")
    ax.set_xlabel("Udział (%)")
    ax.set_ylabel(col)
    ax.set_title(f"{col} — udział kategorii")

    for i, v in enumerate(pct.values):
        ax.text(v + max(pct.values) * 0.01, i, f"{v:.2f}%", va="center", fontsize=9)

fig.tight_layout()
# plt.show()

print(data["SugarScale"].value_counts(dropna=False))

def plato_to_sg(p):
    p = np.asarray(p, dtype=float)
    return 1 + (p / (258.6 - ((p / 258.2) * 227.1)))

mask_plato = data_clean["SugarScale"].astype(str).str.contains("Plato", case=False, na=False)

print("Przed konwersją:")
print(data_clean["SugarScale"].value_counts(dropna=False))
print("Liczba rekordów Plato:", int(mask_plato.sum()))

for col in ["OG", "FG", "BoilGravity"]:
    if col in data_clean.columns:
        data_clean.loc[mask_plato, col] = plato_to_sg(data_clean.loc[mask_plato, col])
data_clean.loc[mask_plato, "SugarScale"] = "Specific Gravity"

print("\nPo konwersji:")
print(data_clean["SugarScale"].value_counts(dropna=False))

data_clean = data_clean.drop(columns=["SugarScale"])
# print(data_clean.info())

print(data.nsmallest(32, "OG")[["Name", "Efficiency", "IBU", "OG", "ABV"]])
print("\n===Czyszczenie i naprawa danych===\n")
n_before = len(data_clean)
data_clean = data_clean[data_clean["OG"] > 1.000].copy()
print("Usunięto:", n_before - len(data_clean),"rekordów ze względu na OG = 1")

# print(data_clean.nsmallest(32, "ABV")[["Style", "Efficiency", "IBU", "OG", "ABV"]])

cols = ["Color", "Efficiency","IBU"]
data_clean[cols] = data_clean[cols].replace(0, np.nan)
print("Zmienianiona liczba rekordów:\n", data_clean[cols].isna().sum())
data_clean[cols] = data_clean.groupby("Style")[cols].transform(
    lambda s: s.fillna(s.median())
)

# print(data_clean.nlargest(200, "IBU")[["Style", "BoilTime", "IBU", "OG", "ABV"]])
n_before = len(data_clean)
data_clean = data_clean[data_clean["FG"] < data_clean["OG"]].copy()
print("Usunięto FG >= OG:", n_before - len(data_clean))
# print(data.nlargest(200, "IBU")[["Name", "BoilTime", "IBU", "OG", "ABV"]])

n_before = len(data_clean)
data_clean = data_clean[data_clean["IBU"] <= 250].copy()
print("Usunięto IBU > 250:", n_before - len(data_clean))

n_before = len(data_clean)
data_clean = data_clean[data_clean["ABV"] <= 20].copy()
print("Usunięto ABV > 20:", n_before - len(data_clean))

n_before = len(data_clean)
data_clean = data_clean[data_clean["Color"] <= 50].copy()
print("Usunięto Color > 50:", n_before - len(data_clean))

print("Liczba pozostałych rekordów:",len(data_clean))

plt.close("all")
y = data_clean["Style"].copy()          
X = data_clean.drop(columns=["Style"])

opis = X.select_dtypes(include="number").describe(percentiles=[0.25, 0.5, 0.75]).T

opis = opis.rename(columns={
    "50%": "mediana",
    "25%": "Q1",
    "75%": "Q3",
    "mean": "średnia",
    "std": "odch. std.",
    "min": "min",
    "max": "max",
})
opis = opis[[ "średnia", "mediana", "odch. std.", "min", "Q1", "Q3", "max"]].round(4)
print(opis.to_string())

cols = ["OG", "FG", "ABV", "IBU", "Color", "BoilTime", "BoilSize", "Size(L)", "Efficiency"]

for i in range(0, len(cols), 3):
    batch = cols[i:i+3]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, batch):
        sns.histplot(X[col], bins=40, kde=False, ax=ax, color="#3a6b35")
        ax.set_title(f"{col} (skośność={X[col].skew():.2f})")
        ax.set_xlabel(col)
        ax.set_ylabel("Liczba")
    
    fig.tight_layout()
# plt.show()

pct = X["BrewMethod"].value_counts(normalize=True).mul(100).round(2)
plt.figure(figsize=(8, 4.5))
sns.barplot(x=pct.values, y=pct.index, color="#2c5f2d")
plt.xlabel("Udział (%)")
plt.ylabel("BrewMethod")
plt.title("BrewMethod — udział kategorii")
for i, v in enumerate(pct.values):
    plt.text(v + max(pct.values) * 0.01, i, f"{v:.2f}%", va="center", fontsize=9)
plt.tight_layout()

counts = data_clean["Style"].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=counts.values, y=counts.index, color="#2c5f2d")
plt.xlabel("Liczba receptur")
plt.ylabel("Styl piwa")
plt.title(f"Liczebność Top {TOP_N} stylów piwa")
for i, v in enumerate(counts.values):
    plt.text(v + max(counts.values) * 0.01, i, f"{v:,}", va="center", fontsize=9)
plt.tight_layout()

plt.show()

cols = ["IBU", "Color", "ABV", "OG", "FG", "Efficiency"]
order = data_clean["Style"].value_counts().index
for col in cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=data_clean, x="Style", y=col, order=order, color="#a3b18a")
    plt.title(f"Wykres pudełkowy: {col} w podziale na style")
    plt.xlabel("Styl piwa")
    plt.ylabel(col)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    # plt.show()

data_clean = data_clean.drop(columns=["Size(L)", "BoilSize"], errors="ignore")
print(data_clean.columns.tolist())

data_clean.to_csv("data_clean.csv", index=False, encoding="utf-8-sig")
print("Zapisano:", len(data_clean), "wierszy,", data_clean.shape[1], "kolumn")