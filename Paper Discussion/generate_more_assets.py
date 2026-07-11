"""Generate additional presentation visuals for a stronger related-work and results story."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
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
RED = "#C0392B"
PURPLE = "#6C5CE7"


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def related_timeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Evolution of Arabic Text Correction Research", fontsize=15, fontweight="bold", color=BLUE_DARK)

    ax.plot([0.5, 11.5], [2.4, 2.4], color=BLUE, linewidth=3, zorder=1)

    events = [
        (1.2, "2014–15", "QALB Shared\nTasks", "L1/L2 Arabic GEC\nbenchmarks"),
        (3.5, "2020–21", "AraBERT &\nCAMeL", "Arabic-specific\npretraining"),
        (5.8, "2022", "BiLSTM Soft\nSpelling", "96.4% corr.\nCER 1.28%"),
        (8.1, "2023", "AraBART +\nGED (EMNLP)", "SOTA on QALB\n& ZAEBUC"),
        (10.4, "2024", "T5 Soft\nSpelling", "97.8% artif.\nCER 0.77%"),
    ]
    for x, year, title, detail in events:
        circ = Circle((x, 2.4), 0.18, facecolor=BLUE, edgecolor="white", linewidth=2, zorder=3)
        ax.add_patch(circ)
        ax.text(x, 3.55, year, ha="center", fontsize=11, fontweight="bold", color=BLUE)
        box = FancyBboxPatch(
            (x - 0.95, 0.35),
            1.9,
            1.55,
            boxstyle="round,pad=0.03,rounding_size=0.12",
            facecolor="#F0F7FB",
            edgecolor=BLUE,
            linewidth=1.5,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, 1.5, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=BLUE_DARK)
        ax.text(x, 0.75, detail, ha="center", va="center", fontsize=8, color=GRAY)

    # Our work marker
    ax.annotate(
        "This work (2026)\nEnd-to-end + multi-modal deployment",
        xy=(11.2, 2.55),
        xytext=(8.8, 4.4),
        fontsize=9,
        fontweight="bold",
        color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8),
        ha="center",
    )
    save(fig, "related_timeline.png")


def related_comparison_chart() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=160)
    methods = [
        "Rule / Dict.\nbaselines",
        "BiLSTM\n(Abandah'22)",
        "Transformer\n(Wiki soft err.)",
        "T5 Soft Err.\n(Al-Qaraghuli'24)",
        "AraBART+GED\n(Alhafni'23)",
        "This work\nCustom / Deploy",
    ]
    # Qualitative capability scores 0-5 for illustration of focus areas
    # Focus coverage: soft spelling, grammar/GEC, deployment multimodal, controlled synthetic analysis
    soft = [2, 5, 5, 5, 3, 5]
    gec = [1, 1, 1, 1, 5, 4]
    deploy = [1, 1, 1, 1, 2, 5]
    control = [2, 4, 4, 4, 3, 5]

    x = np.arange(len(methods))
    w = 0.18
    ax.bar(x - 1.5 * w, soft, w, label="Soft spelling focus", color=BLUE)
    ax.bar(x - 0.5 * w, gec, w, label="Grammar / GEC focus", color=BLUE_LIGHT)
    ax.bar(x + 0.5 * w, deploy, w, label="Multi-modal deployment", color=ORANGE)
    ax.bar(x + 1.5 * w, control, w, label="Controlled analysis", color=GREEN)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylabel("Relative Coverage (qualitative 0–5)", fontsize=11)
    ax.set_ylim(0, 5.8)
    ax.set_title("Related Work Positioning by Capability Coverage", fontsize=14, fontweight="bold", color=BLUE_DARK)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "related_positioning.png")


def research_gap() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.3), dpi=160)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    ax.set_title("Research Gap Addressed by This Work", fontsize=15, fontweight="bold", color=BLUE_DARK)

    # Prior work column
    box = FancyBboxPatch((0.3, 0.4), 4.8, 4.4, boxstyle="round,pad=0.04,rounding_size=0.15",
                         facecolor="#F7F9FB", edgecolor=GRAY, linewidth=2)
    ax.add_patch(box)
    ax.text(2.7, 4.45, "What Prior Work Does Well", ha="center", fontsize=12, fontweight="bold", color=GRAY)
    prior = [
        "Soft-error BiLSTM / Transformer / T5",
        "QALB / ZAEBUC GEC benchmarks",
        "AraBART + morphological GED tags",
        "Synthetic error injection recipes",
        "Strong CER on soft-spelling test sets",
    ]
    for i, t in enumerate(prior):
        ax.text(0.6, 3.7 - i * 0.55, f"•  {t}", fontsize=10.5, color=BLUE_DARK)

    # Gap arrow
    ax.annotate("", xy=(6.3, 2.6), xytext=(5.3, 2.6),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=3))
    ax.text(5.8, 3.15, "GAP", ha="center", fontsize=12, fontweight="bold", color=ORANGE)

    # Our work
    box2 = FancyBboxPatch((6.4, 0.4), 4.8, 4.4, boxstyle="round,pad=0.04,rounding_size=0.15",
                          facecolor="#F0F7FB", edgecolor=BLUE, linewidth=2.5)
    ax.add_patch(box2)
    ax.text(8.8, 4.45, "Our End-to-End Contribution", ha="center", fontsize=12, fontweight="bold", color=BLUE)
    ours = [
        "Modern Youm7 news corpus + pairs",
        "Two complementary correction paths",
        "In-domain custom Transformer analysis",
        "Zero-shot AraBART multi-modal app",
        "Text + OCR + ASR + live speech",
        "Trade-off study, not single winner claim",
    ]
    for i, t in enumerate(ours):
        ax.text(6.7, 3.7 - i * 0.5, f"•  {t}", fontsize=10.5, color=BLUE_DARK)
    save(fig, "research_gap.png")


def related_table_visual() -> None:
    """Create a visual summary table as an image for crisp rendering."""
    fig, ax = plt.subplots(figsize=(12.2, 5.4), dpi=170)
    ax.axis("off")
    ax.set_title("Comparative Landscape of Closely Related Systems", fontsize=14, fontweight="bold", color=BLUE_DARK, pad=14)

    cols = ["Work", "Year", "Approach", "Focus", "Key Result / Note", "Limitation vs Ours"]
    rows = [
        ["Abandah et al.", "2022", "BiLSTM\nchar seq", "Soft spelling", "96.4% corr.; CER 1.28%\n(real soft errors)", "Limited long-range\n+ no deployment path"],
        ["Al-Qaraghuli\n& Jaafar", "2024", "T5\n(4-layer)", "Soft spelling", "97.8% artif. corr.;\nCER 0.77% (Test200)", "Soft-error only;\nno OCR/ASR service"],
        ["Alhafni et al.\n(CAMeL)", "2023", "AraBART\n+ GED", "Arabic GEC", "SOTA on QALB &\nstrong ZAEBUC baseline", "Benchmark-centric;\nnot multi-modal UX"],
        ["QALB / ZAEBUC", "2014–22", "Shared tasks\n/ corpora", "GEC eval", "Standard L1/L2\nArabic GEC resources", "Not a full training\n+ serving workflow"],
        ["This work", "2026", "Custom TF +\nAraBART app", "End-to-end", "CER 0.0364 in-domain;\nmulti-modal deploy", "Fills workflow &\ndeployment gap"],
    ]

    # Manual table drawing
    col_x = [0.02, 0.16, 0.26, 0.40, 0.54, 0.76]
    col_w = [0.14, 0.10, 0.14, 0.14, 0.22, 0.22]
    y0 = 0.82
    row_h = 0.14

    # header
    for x, w, c in zip(col_x, col_w, cols):
        rect = FancyBboxPatch((x, y0), w - 0.005, 0.08, boxstyle="square,pad=0",
                              transform=ax.transAxes, facecolor=BLUE, edgecolor="white")
        ax.add_patch(rect)
        ax.text(x + w / 2, y0 + 0.04, c, transform=ax.transAxes, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    for i, row in enumerate(rows):
        y = y0 - (i + 1) * row_h
        bg = "#E8F4FB" if i == len(rows) - 1 else ("#F7F9FB" if i % 2 == 0 else "white")
        for x, w, val in zip(col_x, col_w, row):
            rect = FancyBboxPatch((x, y), w - 0.005, row_h - 0.01, boxstyle="square,pad=0",
                                  transform=ax.transAxes, facecolor=bg, edgecolor="#D0D7DE", linewidth=0.8)
            ax.add_patch(rect)
            weight = "bold" if i == len(rows) - 1 or x == col_x[0] else "normal"
            ax.text(x + w / 2, y + row_h / 2 - 0.005, val, transform=ax.transAxes,
                    ha="center", va="center", fontsize=8.2, fontweight=weight, color=BLUE_DARK)
    save(fig, "related_comparison_table.png")


def multimodal_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.8), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("AraBART Multi-Modal Correction Service", fontsize=15, fontweight="bold", color=BLUE_DARK)

    inputs = [
        (0.4, 3.6, "Manual Text"),
        (0.4, 2.5, "Text File"),
        (0.4, 1.4, "Image → OCR"),
        (0.4, 0.3, "Audio / Live\nSpeech → Whisper"),
    ]
    for x, y, label in inputs:
        box = FancyBboxPatch((x, y), 2.6, 0.9, boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor="#FFF8EC", edgecolor=ORANGE, linewidth=1.8)
        ax.add_patch(box)
        ax.text(x + 1.3, y + 0.45, label, ha="center", va="center", fontsize=10, fontweight="bold", color=BLUE_DARK)

    # preprocess
    box = FancyBboxPatch((3.6, 1.6), 2.4, 1.8, boxstyle="round,pad=0.03,rounding_size=0.12",
                         facecolor="#F0F7FB", edgecolor=BLUE, linewidth=2)
    ax.add_patch(box)
    ax.text(4.8, 2.9, "Preprocess", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(4.8, 2.2, "cleanup\nnormalize\nchunk", ha="center", fontsize=10, color=GRAY)

    # model
    box = FancyBboxPatch((6.6, 1.6), 2.6, 1.8, boxstyle="round,pad=0.03,rounding_size=0.12",
                         facecolor=BLUE, edgecolor=BLUE_DARK, linewidth=2)
    ax.add_patch(box)
    ax.text(7.9, 2.9, "AraBART", ha="center", fontsize=12, fontweight="bold", color="white")
    ax.text(7.9, 2.2, "CAMeL-Lab\nqalb15-gec-ged-13", ha="center", fontsize=9, color="#DCECF5")

    # output
    box = FancyBboxPatch((9.7, 1.6), 2.0, 1.8, boxstyle="round,pad=0.03,rounding_size=0.12",
                         facecolor=GREEN, edgecolor="#1F6B45", linewidth=2)
    ax.add_patch(box)
    ax.text(10.7, 2.5, "Corrected\nArabic Text", ha="center", va="center", fontsize=11, fontweight="bold", color="white")

    for y in [4.05, 2.95, 1.85, 0.75]:
        ax.annotate("", xy=(3.55, 2.5), xytext=(3.05, y),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    ax.annotate("", xy=(6.55, 2.5), xytext=(6.05, 2.5), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.annotate("", xy=(9.65, 2.5), xytext=(9.25, 2.5), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.text(6, 4.55, "Streamlit Application Layer", ha="center", fontsize=11, style="italic", color=GRAY)
    save(fig, "multimodal_flow.png")


def eval_protocol() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=160)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Two Evaluation Settings (Not One Leaderboard)", fontsize=15, fontweight="bold", color=BLUE_DARK)

    box = FancyBboxPatch((0.4, 0.5), 5.0, 3.9, boxstyle="round,pad=0.04,rounding_size=0.15",
                         facecolor="#F0F7FB", edgecolor=BLUE, linewidth=2.5)
    ax.add_patch(box)
    ax.text(2.9, 4.05, "Setting A — In-domain", ha="center", fontsize=13, fontweight="bold", color=BLUE)
    ax.text(2.9, 3.35, "Custom Character Transformer", ha="center", fontsize=11, fontweight="bold", color=BLUE_DARK)
    for i, t in enumerate([
        "1,000 held-out synthetic pairs",
        "Same corruption policy as train",
        "Metrics: CER, BLEU, confusion",
        "Best for fidelity & diagnostics",
        "Reported CER = 0.0364",
    ]):
        ax.text(0.75, 2.7 - i * 0.4, f"• {t}", fontsize=10.5, color=BLUE_DARK)

    box = FancyBboxPatch((6.1, 0.5), 5.0, 3.9, boxstyle="round,pad=0.04,rounding_size=0.15",
                         facecolor="#FFF8EC", edgecolor=ORANGE, linewidth=2.5)
    ax.add_patch(box)
    ax.text(8.6, 4.05, "Setting B — Zero-shot Deploy", ha="center", fontsize=13, fontweight="bold", color=ORANGE)
    ax.text(8.6, 3.35, "AraBART Streamlit Service", ha="center", fontsize=11, fontweight="bold", color=BLUE_DARK)
    for i, t in enumerate([
        "App-style inputs (text/OCR/ASR)",
        "Distribution ≠ synthetic generator",
        "Focus: fluency + usability",
        "No project-specific fine-tuning",
        "Reported CER = 0.0950",
    ]):
        ax.text(6.45, 2.7 - i * 0.4, f"• {t}", fontsize=10.5, color=BLUE_DARK)
    save(fig, "eval_protocol.png")


def before_after() -> None:
    fig, ax = plt.subplots(figsize=(11.8, 5.0), dpi=160)
    ax.set_xlim(0, 11.8)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Noisy → Clean Correction Examples", fontsize=15, fontweight="bold", color=BLUE_DARK)

    examples = [
        ("اعلنت كليه الصيدله عن مواعبد التسجيل", "أعلنت كلية الصيدلة عن مواعيد التسجيل", "ة/ه + Alef + letter swaps"),
        ("انلقت مبادرة مدارس النيل المصرية الودلية", "انطلقت مبادرة مدارس النيل المصرية الدولية", "Dropped / substituted chars"),
        ("هزا كتاب مفيد", "هذا كتاب مفيد", "Visual soft substitution"),
    ]
    y = 3.7
    for noisy, clean, note in examples:
        box_n = FancyBboxPatch((0.4, y), 5.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor="#FDEDEC", edgecolor=RED, linewidth=1.5)
        box_c = FancyBboxPatch((6.4, y), 5.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor="#E8F8F0", edgecolor=GREEN, linewidth=1.5)
        ax.add_patch(box_n)
        ax.add_patch(box_c)
        ax.text(0.6, y + 0.65, "Noisy", fontsize=9, fontweight="bold", color=RED)
        ax.text(6.6, y + 0.65, "Clean", fontsize=9, fontweight="bold", color=GREEN)
        ax.text(2.9, y + 0.35, noisy, ha="center", va="center", fontsize=12, color=BLUE_DARK)
        ax.text(8.9, y + 0.35, clean, ha="center", va="center", fontsize=12, color=BLUE_DARK)
        ax.annotate("", xy=(6.35, y + 0.5), xytext=(5.5, y + 0.5),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
        ax.text(5.95, y + 0.85, note, ha="center", fontsize=8, color=GRAY, style="italic")
        y -= 1.25
    save(fig, "before_after.png")


def noise_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=160)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.set_title("Controlled Noise Synthesis Pipeline", fontsize=15, fontweight="bold", color=BLUE_DARK)

    steps = [
        (0.3, "Clean\nYoum7 Text"),
        (2.5, "Arabic\nNormalization"),
        (4.7, "Error Position\nSelection"),
        (6.9, "Confusable /\nPunctuation Ops"),
        (9.1, "Noisy–Clean\nParallel Pair"),
    ]
    for i, (x, label) in enumerate(steps):
        color = GREEN if i == 0 or i == len(steps) - 1 else BLUE
        box = FancyBboxPatch((x, 1.7), 1.9, 1.5, boxstyle="round,pad=0.03,rounding_size=0.12",
                             facecolor="#F0F7FB", edgecolor=color, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 0.95, 2.45, label, ha="center", va="center", fontsize=10, fontweight="bold", color=BLUE_DARK)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 2.35, 2.45), xytext=(x + 2.0, 2.45),
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))

    ax.text(6, 0.9, "Budget: 20% corruption  =  10% substitutions  +  5% deletions  +  5% insertions",
            ha="center", fontsize=11, color=GRAY)
    ax.text(6, 0.35, "Validated for plausibility against common Arabic soft-error patterns",
            ha="center", fontsize=10, style="italic", color=ORANGE)
    save(fig, "noise_pipeline.png")


def takeaway_quadrant() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=160)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axhline(5, color="#CBD5E1", lw=1.5)
    ax.axvline(5, color="#CBD5E1", lw=1.5)
    ax.set_xlabel("Deployment / Multi-modal Readiness →", fontsize=11, color=GRAY)
    ax.set_ylabel("In-domain Character Fidelity →", fontsize=11, color=GRAY)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Where Each Path Wins", fontsize=14, fontweight="bold", color=BLUE_DARK)

    # Custom high fidelity, lower multimodal
    ax.scatter([2.8], [8.2], s=900, color=BLUE, zorder=3)
    ax.text(2.8, 8.2, "Custom\nTF", ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
    ax.text(2.8, 6.7, "CER 0.0364\n~50 MB", ha="center", fontsize=9, color=BLUE_DARK)

    # AraBART high multimodal, moderate fidelity in zero-shot
    ax.scatter([8.0], [5.8], s=900, color=ORANGE, zorder=3)
    ax.text(8.0, 5.8, "AraBART\nService", ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
    ax.text(8.0, 4.3, "CER 0.0950\nText/OCR/ASR", ha="center", fontsize=9, color=BLUE_DARK)

    ax.text(2.5, 1.2, "Lightweight &\nanalyzable", ha="center", fontsize=10, color=GRAY)
    ax.text(7.8, 1.2, "Practical &\nuser-facing", ha="center", fontsize=10, color=GRAY)
    save(fig, "takeaway_quadrant.png")


def metric_cards() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 3.8), dpi=160)
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 3.8)
    ax.axis("off")
    ax.set_title("Headline Results at a Glance", fontsize=15, fontweight="bold", color=BLUE_DARK)

    cards = [
        (0.3, "0.0364", "In-domain CER", "Custom Transformer"),
        (3.1, "91.69%", "Peak Token Acc.", "Epoch 6"),
        (5.9, "89.36%", "Final Acc.", "Epoch 10"),
        (8.7, "0.0950", "Deploy CER", "AraBART zero-shot"),
    ]
    for x, big, mid, small in cards:
        box = FancyBboxPatch((x, 0.4), 2.5, 2.8, boxstyle="round,pad=0.03,rounding_size=0.15",
                             facecolor="#F0F7FB", edgecolor=BLUE, linewidth=2)
        ax.add_patch(box)
        ax.text(x + 1.25, 2.35, big, ha="center", fontsize=22, fontweight="bold", color=BLUE)
        ax.text(x + 1.25, 1.5, mid, ha="center", fontsize=11, fontweight="bold", color=BLUE_DARK)
        ax.text(x + 1.25, 0.9, small, ha="center", fontsize=10, color=GRAY)
    save(fig, "metric_cards.png")


if __name__ == "__main__":
    related_timeline()
    related_comparison_chart()
    research_gap()
    related_table_visual()
    multimodal_flow()
    eval_protocol()
    before_after()
    noise_pipeline()
    takeaway_quadrant()
    metric_cards()
    print("Done")
