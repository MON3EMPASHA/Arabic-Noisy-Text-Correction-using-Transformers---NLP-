"""Generate presentation chart assets matching IMSA template colors."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

BLUE = "#006499"
BLUE_LIGHT = "#4A9BC7"
BLUE_DARK = "#004466"
ACCENT = "#E8A838"
GRAY = "#5A6A7A"
GREEN = "#2E8B57"
ORANGE = "#D68910"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def training_curve() -> None:
    epochs = np.arange(1, 11)
    acc = [38.38, 62.5, 78.2, 85.4, 89.8, 91.69, 91.2, 90.5, 89.9, 89.36]
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    ax.plot(epochs, acc, color=BLUE, marker="o", linewidth=2.5, markersize=8, label="Token Accuracy")
    ax.axvline(6, color=ACCENT, linestyle="--", linewidth=1.5, label="Peak (Epoch 6)")
    ax.fill_between(epochs, acc, alpha=0.12, color=BLUE)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Token-level Accuracy (%)", fontsize=12)
    ax.set_title("Custom Transformer Training Dynamics", fontsize=14, fontweight="bold", color=BLUE_DARK)
    ax.set_xticks(epochs)
    ax.set_ylim(30, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    for x, y in [(1, 38.38), (6, 91.69), (10, 89.36)]:
        ax.annotate(
            f"{y:.2f}%",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color=BLUE_DARK,
        )
    save(fig, "training_curve.png")


def error_pie() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
    sizes = [65, 17.5, 17.5]
    labels = ["Substitutions\n~60-70%", "Insertions\n~15-20%", "Deletions\n~15-20%"]
    colors = [BLUE, BLUE_LIGHT, ACCENT]
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        explode=(0.04, 0.02, 0.02),
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
    )
    ax.set_title(
        "Error Type Distribution\n(Custom Transformer Analysis)",
        fontsize=13,
        fontweight="bold",
        color=BLUE_DARK,
    )
    centre = plt.Circle((0, 0), 0.28, fc="white")
    ax.add_artist(centre)
    ax.text(0, 0, "CER\nFocus", ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE_DARK)
    save(fig, "error_pie.png")


def baseline_bars() -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=160)
    methods = ["Rule-based\n(~30-50%)", "BiLSTM-style\n(~80-85%)", "Custom Transformer\n(89.36%)"]
    vals = [40, 82.5, 89.36]
    bars = ax.bar(methods, vals, color=[GRAY, BLUE_LIGHT, BLUE], width=0.55, edgecolor="white")
    ax.errorbar(
        [0, 1],
        [40, 82.5],
        yerr=[[10, 2.5], [10, 2.5]],
        fmt="none",
        ecolor=BLUE_DARK,
        capsize=6,
        linewidth=1.5,
    )
    ax.set_ylabel("Character Accuracy (%)", fontsize=12)
    ax.set_title("Baseline Context on Synthetic Split", fontsize=14, fontweight="bold", color=BLUE_DARK)
    ax.set_ylim(0, 105)
    ax.grid(True, axis="y", alpha=0.3)
    for b, label in zip(bars, ["~40%", "~82.5%", "89.36%"]):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 2,
            label,
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=BLUE_DARK,
        )
    save(fig, "baseline_bars.png")


def cer_comparison() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=160)
    x = np.arange(2)
    cer = [0.0364, 0.0950]
    bars = ax.bar(x, cer, 0.45, color=[BLUE, BLUE_LIGHT], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Custom Transformer\n(In-domain Synthetic)", "AraBART Service\n(Zero-shot Multi-modal)"]
    )
    ax.set_ylabel("Character Error Rate (CER)", fontsize=12)
    ax.set_title("CER Across Evaluation Settings\n(Lower is Better)", fontsize=14, fontweight="bold", color=BLUE_DARK)
    ax.set_ylim(0, 0.14)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, cer):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.004,
            f"{v:.4f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
            color=BLUE_DARK,
        )
    save(fig, "cer_comparison.png")


def footprint() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    names = ["Custom Transformer", "AraBART Service"]
    sizes_mb = [50, 1500]
    bars = ax.barh(names, sizes_mb, color=[GREEN, ORANGE], height=0.5)
    ax.set_xlabel("Model Footprint (MB, log scale)", fontsize=12)
    ax.set_title("Deployment Footprint Comparison", fontsize=14, fontweight="bold", color=BLUE_DARK)
    ax.set_xscale("log")
    ax.grid(True, axis="x", alpha=0.3)
    for b, label in zip(bars, ["~50 MB", "~1.5 GB"]):
        ax.text(
            b.get_width() * 1.15,
            b.get_y() + b.get_height() / 2,
            label,
            va="center",
            fontsize=12,
            fontweight="bold",
            color=BLUE_DARK,
        )
    save(fig, "footprint.png")


def soft_errors() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=160)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("Common Arabic Soft Spelling Error Types", fontsize=15, fontweight="bold", color=BLUE_DARK, pad=12)
    cards = [
        (0.3, 2.7, "Alef Variants", "ا  أ  إ  آ", "Form confusion"),
        (3.8, 2.7, "Hamza Placement", "ء  ئ  ؤ  أ", "Position mistakes"),
        (7.3, 2.7, "Teh Marbuta / Heh", "ة  ↔  ه", "Ending swap"),
        (0.3, 0.3, "Yaa / Alif Maqsura", "ي  ↔  ى", "Final form mix"),
        (3.8, 0.3, "Punctuation Drift", "،   .   \" \"", "Marks and spacing"),
        (7.3, 0.3, "Confusable Letters", "ض/ظ  ب/ت", "Visual similarity"),
    ]
    for x, y, title, ex, sub in cards:
        box = FancyBboxPatch(
            (x, y),
            3.2,
            2.1,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#F0F7FB",
            edgecolor=BLUE,
            linewidth=2,
        )
        ax.add_patch(box)
        ax.text(x + 1.6, y + 1.55, title, ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE_DARK)
        ax.text(x + 1.6, y + 0.95, ex, ha="center", va="center", fontsize=16, color=BLUE)
        ax.text(x + 1.6, y + 0.35, sub, ha="center", va="center", fontsize=10, color=GRAY, style="italic")
    save(fig, "soft_errors.png")


def corruption_budget() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    cats = ["Substitutions\n10%", "Deletions\n5%", "Insertions\n5%", "Clean\n80%"]
    vals = [10, 5, 5, 80]
    colors = [BLUE, ACCENT, BLUE_LIGHT, "#D5E8F0"]
    bars = ax.bar(cats, vals, color=colors, edgecolor="white", width=0.6)
    ax.set_ylabel("Share of Characters (%)", fontsize=12)
    ax.set_title("Synthetic Corruption Budget (20% Noise)", fontsize=14, fontweight="bold", color=BLUE_DARK)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 2,
            f"{v}%",
            ha="center",
            fontsize=11,
            fontweight="bold",
            color=BLUE_DARK,
        )
    save(fig, "corruption_budget.png")


def contributions() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=160)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    ax.set_title("Main Contributions", fontsize=15, fontweight="bold", color=BLUE_DARK)
    items = [
        (0.4, "1", "Reproducible Workflow", "Corpus, normalize,\nnoise, train, deploy"),
        (3.9, "2", "Two Correction Paths", "Compact Transformer +\nAraBART multi-modal"),
        (7.4, "3", "Strong In-domain CER", "0.0364 CER on held-out\nsynthetic test split"),
    ]
    for x, num, title, desc in items:
        box = FancyBboxPatch(
            (x, 0.4),
            3.2,
            3.2,
            boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor="white",
            edgecolor=BLUE,
            linewidth=2.5,
        )
        ax.add_patch(box)
        circ = Circle((x + 1.6, 2.9), 0.45, facecolor=BLUE, edgecolor="none")
        ax.add_patch(circ)
        ax.text(x + 1.6, 2.9, num, ha="center", va="center", fontsize=16, fontweight="bold", color="white")
        ax.text(x + 1.6, 1.9, title, ha="center", va="center", fontsize=11, fontweight="bold", color=BLUE_DARK)
        ax.text(x + 1.6, 1.05, desc, ha="center", va="center", fontsize=9.5, color=GRAY)
    save(fig, "contributions.png")


def two_tier() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.0), dpi=160)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Two-Tier Arabic Correction Framework", fontsize=15, fontweight="bold", color=BLUE_DARK)

    # Shared input
    box = FancyBboxPatch(
        (3.5, 4.0), 4, 0.7, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=BLUE, edgecolor=BLUE_DARK
    )
    ax.add_patch(box)
    ax.text(5.5, 4.35, "Arabic Noisy Input", ha="center", va="center", color="white", fontsize=12, fontweight="bold")

    # Branch A
    box_a = FancyBboxPatch(
        (0.4, 0.5), 4.6, 3.0, boxstyle="round,pad=0.03,rounding_size=0.15", facecolor="#F0F7FB", edgecolor=BLUE, lw=2
    )
    ax.add_patch(box_a)
    ax.text(2.7, 3.15, "Branch A — Controlled Benchmarking", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(
        2.7,
        2.35,
        "Custom Character-level\nSeq2Seq Transformer",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=BLUE_DARK,
    )
    ax.text(
        2.7,
        1.35,
        "In-domain synthetic pairs\nCER 0.0364  |  ~50 MB\nBest for fidelity analysis",
        ha="center",
        fontsize=10,
        color=GRAY,
    )

    # Branch B
    box_b = FancyBboxPatch(
        (6.0, 0.5), 4.6, 3.0, boxstyle="round,pad=0.03,rounding_size=0.15", facecolor="#FFF8EC", edgecolor=ORANGE, lw=2
    )
    ax.add_patch(box_b)
    ax.text(8.3, 3.15, "Branch B — Real-world Deployment", ha="center", fontsize=11, fontweight="bold", color=ORANGE)
    ax.text(
        8.3,
        2.35,
        "CAMeL-Lab AraBART\nStreamlit Multi-modal Service",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color=BLUE_DARK,
    )
    ax.text(
        8.3,
        1.35,
        "Text / OCR / Audio / Speech\nCER 0.0950  |  ~1.5 GB\nBest for fluency & usability",
        ha="center",
        fontsize=10,
        color=GRAY,
    )

    ax.annotate("", xy=(2.7, 3.5), xytext=(4.5, 4.0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.annotate("", xy=(8.3, 3.5), xytext=(6.5, 4.0), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2))
    save(fig, "two_tier.png")


def dataset_split() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    labels = ["Train\n8,000", "Validation\n1,000", "Test\n1,000"]
    sizes = [8000, 1000, 1000]
    colors = [BLUE, BLUE_LIGHT, ACCENT]
    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
        textprops=dict(fontsize=11),
    )
    ax.set_title("Noisy-Clean Pair Split (10,000 total)", fontsize=14, fontweight="bold", color=BLUE_DARK)
    save(fig, "dataset_split.png")


if __name__ == "__main__":
    training_curve()
    error_pie()
    baseline_bars()
    cer_comparison()
    footprint()
    soft_errors()
    corruption_budget()
    contributions()
    two_tier()
    dataset_split()
    print("Done. Assets in", OUT)
