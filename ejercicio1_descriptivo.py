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

    df = df.copy()  # 🔒 evita modificar el original

    # ---------------------------
    # Feature engineering
    # ---------------------------
    df["age"] = df["year_of_mission"] - df["year_of_birth"]
    df["years_since_selection"] = df["year_of_mission"] - df["year_of_selection"]

    # ---------------------------
    # Columnas a eliminar
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
    # Variables numéricas
    # ---------------------------
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
        plt.xticks(rotation=45)
        plt.title(f"{target} vs {col}")

    plt.tight_layout()
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


if __name__ == "__main__":
    df = load_data("data/astronauts.csv")

    summary = structural_summary(df)

    print("Filas:", summary["n_rows"])
    print("Columnas:", summary["n_cols"])
    print("Memoria (MB):", summary["memory_usage_mb"])
    print("\nTipos de datos:\n", summary["dtypes"])
    print("\nValores nulos:\n", summary["nulls"])

    df = pd.read_csv("data/astronauts.csv")

    desc, target_metrics = numerical_summary(df)

    # Guardar output requerido
    desc.to_csv("output/ej1_descriptivo.csv", float_format="%.3f")

    # ---------------------------
    # Histogramas
    # ---------------------------
    plot_histograms(
        df,
        output_path="output/ej1_histogramas.png"
    )

    # ---------------------------
    # Boxplots (target vs categóricas)
    # ---------------------------

    target = "hours_mission"

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # opcional: limpiar categóricas irrelevantes
    categorical_cols = [
        col for col in categorical_cols
        if df[col].nunique() < 50  # evita variables tipo nombre
    ]

    plot_boxplots(
        df,
        target=target,
        cat_cols=categorical_cols,
        output_path="output/ej1_boxplots.png"
    )