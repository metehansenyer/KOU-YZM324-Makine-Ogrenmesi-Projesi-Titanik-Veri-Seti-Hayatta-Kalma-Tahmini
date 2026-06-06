import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import FIG_DIR


def run_eda(df: pd.DataFrame) -> None:
    _fig1_survival_distribution(df)
    _fig2_survival_by_sex_class(df)
    _fig3_age_distribution(df)
    print("EDA grafikleri kaydedildi.")


def _fig1_survival_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Survived"].value_counts().sort_index()
    ax.bar(["Hayatta Kalmadı (0)", "Hayatta Kaldı (1)"], counts.values, color=["#e74c3c", "#2ecc71"])
    ax.set_title("Hayatta Kalma Dağılımı")
    ax.set_ylabel("Yolcu Sayısı")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_survival_distribution.png", dpi=150)
    plt.close(fig)


def _fig2_survival_by_sex_class(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sex_rate = df.groupby("Sex")["Survived"].mean()
    axes[0].bar(sex_rate.index, sex_rate.values, color=["#3498db", "#e91e63"])
    axes[0].set_title("Cinsiyete Göre Hayatta Kalma Oranı")
    axes[0].set_ylabel("Hayatta Kalma Oranı")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(sex_rate.values):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    class_rate = df.groupby("Pclass")["Survived"].mean()
    axes[1].bar([f"Sınıf {c}" for c in class_rate.index], class_rate.values,
                color=["#f39c12", "#27ae60", "#8e44ad"])
    axes[1].set_title("Yolcu Sınıfına Göre Hayatta Kalma Oranı")
    axes[1].set_ylabel("Hayatta Kalma Oranı")
    axes[1].set_ylim(0, 1)
    for i, v in enumerate(class_rate.values):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_survival_by_sex_class.png", dpi=150)
    plt.close(fig)


def _fig3_age_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    survived = df[df["Survived"] == 1]["Age"].dropna()
    not_survived = df[df["Survived"] == 0]["Age"].dropna()
    ax.hist([not_survived, survived], bins=30, stacked=True,
            label=["Hayatta Kalmadı", "Hayatta Kaldı"],
            color=["#e74c3c", "#2ecc71"], alpha=0.8)
    ax.set_title("Yaşa Göre Hayatta Kalma Dağılımı")
    ax.set_xlabel("Yaş")
    ax.set_ylabel("Yolcu Sayısı")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_age_distribution.png", dpi=150)
    plt.close(fig)
