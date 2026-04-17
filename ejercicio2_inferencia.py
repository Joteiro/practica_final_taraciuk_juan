import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import seaborn as sns


def preprocess_for_model(df):
    """
    Limpieza + feature engineering + selección de variables
    """

    df = df.copy()

    # Features derivadas
    df["age"] = df["year_of_mission"] - df["year_of_birth"]
    df["years_since_selection"] = df["year_of_mission"] - df["year_of_selection"]

    # Drop columnas irrelevantes
    df = df.drop(columns=[
        "id", "name", "original_name",
        "number", "nationwide_number",
        "selection", "year_of_selection",
        "field21", "year_of_birth",
        "total_number_of_missions", "mission_title",
        "ascend_shuttle", "in_orbit", "descend_shuttle",
        "total_hrs_sum", "total_eva_hrs"  # casi lo mismo que "eva_hrs_mission"
    ])

    # Target
    y = df["hours_mission"]

    # Features
    X = df.drop(columns=["hours_mission"])

    return X, y


def prepare_features(X):
    """
    One-hot encoding + scaling
    """

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Guardar columnas
    cols = X.columns

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, cols


def split_data(X, y):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# ---------------------------
# Modelo lineal
# ---------------------------

def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# ---------------------------
# Métricas
# ---------------------------

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2, y_pred

# ---------------------------
# Coeficientes (Top 10)
# ---------------------------

def plot_top_coefficients(model, feature_names, output_path):
    coefs = pd.Series(model.coef_, index=feature_names)

    top_coefs = coefs.abs().sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    top_coefs.sort_values().plot(kind="barh")

    plt.title("Top 10 coeficientes (valor absoluto)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ---------------------------
# Gráfico de residuos
# ---------------------------

def plot_residuals(y_test, y_pred, output_path):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)

    plt.axhline(0, linestyle="--")

    plt.xlabel("Valores predichos")
    plt.ylabel("Residuos")
    plt.title("Gráfico de residuos")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# ---------------------------
# Guardar métricas
# ---------------------------

def save_metrics(mae, rmse, r2, output_path):
    with open(output_path, "w") as f:
        f.write(f"MAE: {mae:.3f}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R2: {r2:.3f}\n")







if __name__ == "__main__":

    # ---------------------------
    # 1. Cargar datos
    # ---------------------------
    df = pd.read_csv("data/astronauts.csv")

    # ---------------------------
    # 2. Preprocesamiento
    # ---------------------------
    X, y = preprocess_for_model(df)

    X_processed, feature_names = prepare_features(X)

    # ---------------------------
    # 3. Split
    # ---------------------------
    X_train, X_test, y_train, y_test = split_data(X_processed, y)

    # ---------------------------
    # 4. Modelo
    # ---------------------------
    model = train_linear_model(X_train, y_train)

    # ---------------------------
    # 5. Evaluación
    # ---------------------------
    mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)

    save_metrics(mae, rmse, r2, "output/ej2_metricas_regresion.txt")

    # ---------------------------
    # 6. Gráficos
    # ---------------------------
    plot_top_coefficients(
        model,
        feature_names,
        "output/ej2_coeficientes.png"
    )

    plot_residuals(
        y_test,
        y_pred,
        "output/ej2_residuos.png"
    )