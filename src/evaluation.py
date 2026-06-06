import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import FIG_DIR, RESULTS_CSV


def compare_models(results_df: pd.DataFrame) -> pd.DataFrame:
    ranked = results_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    _save_comparison_chart(ranked)

    ranked.to_csv(RESULTS_CSV, index=False, encoding="utf-8-sig")
    print("\n=== Model Karşılaştırma Sonuçları ===")
    print(ranked.to_string(index=False))
    best = ranked.iloc[0]["Model"]
    print(f"\nEn iyi model: {best} (Accuracy: {ranked.iloc[0]['Accuracy']:.4f})")

    return ranked


def _save_comparison_chart(ranked: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    models = ranked["Model"].tolist()
    x = np.arange(len(models))
    width = 0.2
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        bars = ax.bar(x + i * width, ranked[metric], width, label=metric, color=color, alpha=0.85)
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Performans Karşılaştırması")
    ax.set_ylabel("Skor")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_model_comparison.png", dpi=150)
    plt.close(fig)
