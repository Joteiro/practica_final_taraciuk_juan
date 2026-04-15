import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(path):
    """
    Carga el dataset desde la ruta especificada.

    Parameters:
    path (str): Ruta al archivo CSV

    Returns:
    pd.DataFrame: DataFrame con los datos cargados
    """
    return pd.read_csv(path)

def preprocess_df(df):
    """
    Limpieza y feature engineering del dataset.
    """

    df = df.copy()

    # ---------------------------
    # Feature engineering
    # ---------------------------
    df["age"] = df["year_of_mission"] - df["year_of_birth"]
    df["years_since_selection"] = df["year_of_mission"] - df["year_of_selection"]

    # ---------------------------
    # Drop columnas irrelevantes
    # ---------------------------
    cols_to_drop = [
        "id",
        "name",
        "original_name",
        "number",
        "nationwide_number",
        "field21",
        "year_of_birth"
    ]

    df = df.drop(columns=cols_to_drop)

    # ---------------------------
    # Limpieza occupation
    # --------------------------
    df["occupation"] = df["occupation"].str.strip().str.lower()

    df["occupation"] = df["occupation"].replace({
        "other (space tourist)": "space tourist"
    })

    return df

def structural_summary(df):
    """
    Genera un resumen estructural del dataset.

    Parameters:
    df (pd.DataFrame): Dataset

    Returns:
    dict: Diccionario con métricas estructurales
    """
    summary = {}

    # Dimensiones
    summary["n_rows"] = df.shape[0]
    summary["n_cols"] = df.shape[1]

    # Memoria
    summary["memory_usage_mb"] = df.memory_usage(deep=True).sum() / (1024 ** 2)

    # Tipos de datos
    summary["dtypes"] = df.dtypes

    # Nulos
    nulls = df.isnull().sum()
    null_percentage = (nulls / len(df)) * 100

    summary["nulls"] = pd.DataFrame({
        "null_count": nulls,
        "null_percentage": null_percentage
    })

    return summary

def compute_iqr(data):
    """
    Calcula IQR y límites de outliers usando método de Tukey.

    Parameters:
    data (array-like): datos numéricos

    Returns:
    dict: Q1, Q3, IQR, lim_inf, lim_sup
    """
    data = np.array(data)

    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR

    return {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lim_inf": lim_inf,
        "lim_sup": lim_sup
    }

def numerical_summary(df):
    """
    Genera estadísticos descriptivos completos para variables numéricas,
    incluyendo variables derivadas y detección de IQR para hours_mission.

    Parameters:
    df (pd.DataFrame): Dataset original

    Returns:
    pd.DataFrame: tabla de estadísticos descriptivos
    dict: información de outliers para hours_mission
    """

    num_df = df.select_dtypes(include=[np.number])

    # ---------------------------
    # Estadísticos base
    # ---------------------------
    desc = num_df.describe().T

    desc["median"] = num_df.median()
    desc["mode"] = num_df.mode().iloc[0]
    desc["var"] = num_df.var()
    desc["skewness"] = num_df.skew()
    desc["kurtosis"] = num_df.kurtosis()

    # ---------------------------
    # IQR SOLO target
    # ---------------------------
    desc["IQR"] = np.nan
    outlier_info = {}

    if "hours_mission" in num_df.columns:
        stats_iqr = compute_iqr(num_df["hours_mission"].dropna())

        desc.loc["hours_mission", "IQR"] = stats_iqr["IQR"]

        outlier_info = stats_iqr

    # ---------------------------
    # Orden final de columnas
    # ---------------------------
    desc = desc[
        [
            "mean", "median", "mode", "std", "var",
            "min", "25%", "50%", "75%", "max",
            "skewness", "kurtosis", "IQR"
        ]
    ]

    return desc, outlier_info

def plot_histograms(df, output_path):
    """
    Genera histogramas para todas las variables numéricas.
    """
    num_df = df.select_dtypes(include=[np.number])

    num_df.hist(figsize=(16, 12), bins=30)
    plt.suptitle("Histogramas de variables numéricas")

    plt.savefig(output_path)
    plt.close()

def plot_boxplots(df, target, cat_cols, output_path):
    """
    Boxplots del target segmentado por variables categóricas.
    """
    plt.figure(figsize=(18, 12))

    n = len(cat_cols)

    for i, col in enumerate(cat_cols):
        plt.subplot((n // 2) + 1, 2, i + 1)
        sns.boxplot(data=df, x=col, y=target)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.title(f"{target} vs {col}")

    plt.savefig(output_path)
    plt.close()

def detect_outliers_iqr(df, col):
    """
    Detecta outliers usando función reutilizable de IQR.
    """
    stats = compute_iqr(df[col].dropna())

    outliers = df[
        (df[col] < stats["lim_inf"]) |
        (df[col] > stats["lim_sup"])
    ]

    stats["n_outliers"] = len(outliers)

    return stats

def summarize_outliers(df):
    """
    Calcula número de outliers por variable numérica usando IQR.

    Returns:
    dict: {columna: nº de outliers}
    """
    num_df = df.select_dtypes(include=[np.number])

    results = {}

    for col in num_df.columns:
        stats = detect_outliers_iqr(num_df, col)
        results[col] = stats["n_outliers"]

    return results

def categorical_analysis(df, output_path, top_n=10):
    """
    Genera análisis de variables categóricas:
    - Frecuencia absoluta y relativa
    - Gráficos de barras (top N categorías)

    Parameters:
    df (pd.DataFrame): Dataset ya preprocesado
    output_path (str): Ruta de guardado
    top_n (int): número de categorías a mostrar
    """

    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    n = len(cat_cols)

    plt.figure(figsize=(18, 5 * n))

    results = {}

    for i, col in enumerate(cat_cols):

        # ---------------------------
        # Frecuencias
        # ---------------------------
        freq_abs = df[col].value_counts()
        freq_rel = df[col].value_counts(normalize=True) * 100

        summary = pd.DataFrame({
            "count": freq_abs,
            "percentage": freq_rel
        })

        results[col] = summary

        # ---------------------------
        # Top N para graficar
        # ---------------------------
        top_data = summary.head(top_n)

        # ---------------------------
        # Plot
        # ---------------------------
        plt.subplot(n, 1, i + 1)

        labels = [
            f"{cat} ({pct:.1f}%)"
            for cat, pct in zip(top_data.index, top_data["percentage"])
        ]

        sns.barplot(
            x=labels,
            y=top_data["count"]
        )

        plt.title(f"{col}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return results

def plot_correlation_heatmap(df, output_path):
    """
    Genera heatmap de correlaciones (Pearson) para variables numéricas.
    """

    num_df = df.select_dtypes(include=[np.number])

    corr = num_df.corr(method="pearson")

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )

    plt.title("Matriz de correlación (Pearson)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return corr

def get_top_correlations(corr_matrix, target, n=3):
    """
    Obtiene las variables más correlacionadas con el target.
    """

    target_corr = corr_matrix[target].drop(target)

    top_corr = target_corr.abs().sort_values(ascending=False).head(n)

    return top_corr

def detect_multicollinearity(corr_matrix, threshold=0.9):
    """
    Detecta pares de variables altamente correlacionadas.
    """

    pairs = []

    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]

            if abs(val) > threshold:
                pairs.append((cols[i], cols[j], val))

    return pairs

if __name__ == "__main__":
    
    # 1. Cargar datos
    df = load_data("data/astronauts.csv")

    summary = structural_summary(df)

    print("Filas:", summary["n_rows"])
    print("Columnas:", summary["n_cols"])
    print("Memoria (MB):", summary["memory_usage_mb"])
    print("\nTipos de datos:\n", summary["dtypes"])
    print("\nValores nulos:\n", summary["nulls"])

    # 2. Preprocesar UNA vez
    df_clean = preprocess_df(df)

    # 3. Estadísticos
    desc, target_metrics = numerical_summary(df)

    desc.to_csv("output/ej1_descriptivo.csv", float_format="%.3f")

    # 4. Histogramas
    plot_histograms(df_clean, "output/ej1_histogramas.png")

    # 5. Boxplots
    target = "hours_mission"

    categorical_cols = df_clean.select_dtypes(include=["object", "string"]).columns.tolist()

    categorical_cols = [
        col for col in categorical_cols
        if df_clean[col].nunique() < 50
    ]

    plot_boxplots(df_clean, target, categorical_cols, "output/ej1_boxplots.png")

    # 6. Outliers
    outliers_summary = summarize_outliers(df_clean)

    with open("output/ej1_outliers.txt", "w") as f:
        f.write("Método utilizado: IQR (1.5 * IQR)\n\n")

        total = len(df_clean)

        for col, n in outliers_summary.items():
            pct = (n / total) * 100
            f.write(f"{col}: {n} outliers ({pct:.2f}%)\n")

    # 7. Variables categóricas
    cat_results = categorical_analysis(
        df_clean,
        output_path="output/ej1_categoricas.png"
    )

# 8. Correlaciones

corr_matrix = plot_correlation_heatmap(
    df_clean,
    output_path="output/ej1_heatmap_correlacion.png"
)

# Top 3 variables correlacionadas con el target
top_corr = get_top_correlations(corr_matrix, target="hours_mission")

print("\nTop correlaciones con hours_mission:\n", top_corr)

# Multicolinealidad
high_corr_pairs = detect_multicollinearity(corr_matrix)

print("\nPares con alta correlación (>0.9):")
for pair in high_corr_pairs:
    print(pair)