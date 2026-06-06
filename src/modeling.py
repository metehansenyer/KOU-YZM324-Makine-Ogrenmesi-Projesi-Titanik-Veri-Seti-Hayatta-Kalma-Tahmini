import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.config import FIG_DIR, RANDOM_STATE, TEST_SIZE


def split_data(df_enc: pd.DataFrame):
    X = df_enc.drop(columns=["Survived"])
    y = df_enc["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, y_train, y_test


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_STATE),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = get_models()
    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fitted_models[name] = model

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})

        _save_confusion_matrix(name, confusion_matrix(y_test, y_pred))
        print(f"{name}: Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")

    rf = fitted_models["Random Forest"]
    _save_feature_importance(rf, X_train.columns)

    results_df = pd.DataFrame(results)
    return results_df, fitted_models


def _save_confusion_matrix(model_name: str, cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Hayatta Kalmadı", "Hayatta Kaldı"],
        yticklabels=["Hayatta Kalmadı", "Hayatta Kaldı"],
        title=f"Karışıklık Matrisi — {model_name}",
        ylabel="Gerçek",
        xlabel="Tahmin",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    fig.savefig(FIG_DIR / f"cm_{safe_name}.png", dpi=150)
    plt.close(fig)


def _save_feature_importance(rf, feature_names) -> None:
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(indices)), importances[indices], color="#3498db")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Random Forest — En Önemli 10 Özellik")
    ax.set_ylabel("Önem Skoru")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_feature_importance.png", dpi=150)
    plt.close(fig)
