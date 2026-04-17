import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# PREPROCESSING COMÚN
# =========================================================

def preprocess_for_model(df):
    """
    Limpieza + feature engineering + selección de variables
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
    df = df.drop(columns=[
        "id", "name", "original_name",
        "number", "nationwide_number",
        "selection", "year_of_selection",
        "field21", "year_of_birth",
        "total_number_of_missions",
        "mission_title",
        "ascend_shuttle", "in_orbit", "descend_shuttle",
        "total_hrs_sum",
        "total_eva_hrs"
    ])

    return df


# =========================================================
# FEATURES
# =========================================================

def prepare_features(X):
    """
    One-hot encoding + scaling
    """

    X = pd.get_dummies(X, drop_first=True)

    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_names, scaler


# =========================================================
# SPLIT
# =========================================================

def split_data(X, y, classification=False):
    """
    Split train/test con opcional estratificación
    """

    if classification:
        return train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


# =========================================================
# MODELO A — REGRESIÓN LINEAL
# =========================================================

def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2, y_pred


def save_regression_metrics(mae, rmse, r2, output_path):
    with open(output_path, "w") as f:
        f.write(f"MAE: {mae:.3f}\n")
        f.write(f"RMSE: {rmse:.3f}\n")
        f.write(f"R2: {r2:.3f}\n")


def plot_top_coefficients(model, feature_names, output_path):
    coefs = pd.Series(model.coef_, index=feature_names)

    top = coefs.abs().sort_values(ascending=False).head(10)

    plt.figure(figsize=(8, 6))
    top.sort_values().plot(kind="barh")

    plt.title("Top 10 coeficientes (valor absoluto)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_residuals(y_test, y_pred, output_path):
    residuals = y_test - y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicciones")
    plt.ylabel("Residuos")
    plt.title("Gráfico de residuos")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# =========================================================
# MODELO B — REGRESIÓN LOGÍSTICA
# =========================================================

def create_target_categories(y):
    """
    Discretización en 3 clases (qcut)
    """

    y_cat, bins = pd.qcut(
        y,
        q=3,
        labels=["bajo", "medio", "alto"],
        retbins=True
    )

    print("\nIntervalos de discretización:")
    for i in range(len(bins) - 1):
        print(f"{y_cat.cat.categories[i]}: {bins[i]:.2f} – {bins[i+1]:.2f} horas")

    return y_cat


def train_logistic_model(X_train, y_train):
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    )

    model.fit(X_train, y_train)
    return model


def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted")
    }

    return metrics, y_pred


def save_classification_metrics(metrics, output_path):
    with open(output_path, "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")


def plot_confusion_matrix(y_test, y_pred, output_path):
    labels = ["bajo", "medio", "alto"]

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title("Matriz de confusión - Regresión logística")
    plt.xlabel("Predicción")
    plt.ylabel("Real")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def artemis_prediction(
    lin_model,
    log_model,
    scaler_reg,
    scaler_clf,
    feature_names_reg,
    feature_names_clf,
    output_path
):
    """
    Predicción Artemis II con scaling correcto
    """

    artemis_df = pd.DataFrame([
        {
            "year_of_mission": 2026,
            "mission_number": 2,
            "occupation": "commander",
            "nationality": "U.S.",
            "military_civilian": "military",
            "eva_hrs_mission": 0,
            "age": 50,
            "years_since_selection": 2026 - 2009,
            "sex": "male",
            "astronaut": "Reid Wiseman"
        },
        {
            "year_of_mission": 2026,
            "mission_number": 2,
            "occupation": "pilot",
            "nationality": "U.S.",
            "military_civilian": "military",
            "eva_hrs_mission": 0,
            "age": 50,
            "years_since_selection": 2026 - 2013,
            "sex": "male",
            "astronaut": "Victor Glover"
        },
        {
            "year_of_mission": 2026,
            "mission_number": 2,
            "occupation": "MSP",
            "nationality": "U.S.",
            "military_civilian": "civilian",
            "eva_hrs_mission": 0,
            "age": 47,
            "years_since_selection": 2026 - 2013,
            "sex": "female",
            "astronaut": "Christina Koch"
        },
        {
            "year_of_mission": 2026,
            "mission_number": 1,
            "occupation": "MSP",
            "nationality": "Canada",
            "military_civilian": "military",
            "eva_hrs_mission": 0,
            "age": 50,
            "years_since_selection": 2026 - 2009,
            "sex": "male",
            "astronaut": "Jeremy Hansen"
        }
    ])

    names = artemis_df["astronaut"]
    X_base = artemis_df.drop(columns=["astronaut"])

    # =========================================================
    # 🔵 REGRESIÓN
    # =========================================================
    X_reg = pd.get_dummies(X_base, drop_first=True)

    for col in feature_names_reg:
        if col not in X_reg:
            X_reg[col] = 0

    X_reg = X_reg[feature_names_reg]

    X_reg_scaled = scaler_reg.transform(X_reg)

    pred_hours = lin_model.predict(X_reg_scaled)

    # =========================================================
    # 🟢 LOGÍSTICA
    # =========================================================
    X_clf = pd.get_dummies(X_base, drop_first=True)

    for col in feature_names_clf:
        if col not in X_clf:
            X_clf[col] = 0

    X_clf = X_clf[feature_names_clf]

    X_clf_scaled = scaler_clf.transform(X_clf)

    pred_class = log_model.predict(X_clf_scaled)

    # =========================================================
    # RESULTADOS
    # =========================================================
    results = pd.DataFrame({
        "astronaut": names,
        "predicted_hours": pred_hours,
        "predicted_category": pred_class
    })

    # Guardar
    with open(output_path, "w") as f:
        f.write("Predicciones Artemis II\n")
        f.write("========================\n\n")

        for _, row in results.iterrows():
            f.write(f"Astronauta: {row['astronaut']}\n")
            f.write(f"  Horas estimadas: {row['predicted_hours']:.2f}\n")
            f.write(f"  Categoría: {row['predicted_category']}\n\n")

    return results

def rebuild_scaler(X_raw, feature_names):
    """
    Reconstruye el scaler original a partir de los datos de entrenamiento
    """

    X = pd.get_dummies(X_raw, drop_first=True)

    # Alinear columnas
    for col in feature_names:
        if col not in X:
            X[col] = 0

    X = X[feature_names]

    scaler = StandardScaler()
    scaler.fit(X)

    return scaler


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    # ---------------------------
    # Load
    # ---------------------------
    df = pd.read_csv("data/astronauts.csv")

    df = preprocess_for_model(df)

    # =====================================================
    # MODELO A — REGRESIÓN LINEAL
    # =====================================================

    X_reg = df.drop(columns=["hours_mission"])
    y_reg = df["hours_mission"]

    X_reg, feat_names_reg, scaler_reg = prepare_features(X_reg)

    X_train, X_test, y_train, y_test = split_data(X_reg, y_reg)

    lin_model = train_linear_model(X_train, y_train)

    mae, rmse, r2, y_pred = evaluate_regression(lin_model, X_test, y_test)

    save_regression_metrics(mae, rmse, r2, "output/ej2_metricas_regresion.txt")
    
    plot_top_coefficients(lin_model, feat_names_reg, "output/ej2_coeficientes.png")

    plot_residuals(y_test, y_pred, "output/ej2_residuos.png")

    print("\nREGRESIÓN LINEAL:")
    print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # =====================================================
    # MODELO B — REGRESIÓN LOGÍSTICA
    # =====================================================

    y_class = create_target_categories(df["hours_mission"])

    X_clf = df.drop(columns=["hours_mission"])
    X_clf, feat_names_clf, scaler_clf = prepare_features(X_clf)

    X_train, X_test, y_train, y_test = split_data(
        X_clf,
        y_class,
        classification=True
    )

    log_model = train_logistic_model(X_train, y_train)

    metrics, y_pred_clf = evaluate_classification(log_model, X_test, y_test)

    save_classification_metrics(metrics, "output/ej2_metricas_logistica.txt")

    plot_confusion_matrix(
        y_test,
        y_pred_clf,
        "output/ej2_matriz_confusion.png"
    )

    print("\nREGRESIÓN LOGÍSTICA:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    
# ---------------------------
# Predicción Artemis II
# ---------------------------
artemis_results = artemis_prediction(
    lin_model,
    log_model,
    scaler_reg,
    scaler_clf,
    feat_names_reg,
    feat_names_clf,
    "output/ej2_artemis_predictions.txt"
)

print("\nPredicciones Artemis II:")
print(artemis_results)