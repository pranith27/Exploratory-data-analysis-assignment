"""Generate figures/architecture.png: supervisor-orchestration diagram.

Documentation figure (not an agent output). Drawn with matplotlib only.
Shows the real AI Data Science Team pattern (left) and what this local
reproduction substitutes (right, hatched): deterministic planner instead
of an LLM router, sklearn stand-in instead of H2O AutoML.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).parent
FIG = HERE / "figures" / "architecture.png"


def box(ax, x, y, w, h, text, fc, ec="#333333", fs=9, hatch=None, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       facecolor=fc, edgecolor=ec, linewidth=1.2, hatch=hatch)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, label=""):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color="#333333", shrinkA=2, shrinkB=2))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + 0.02, (p1[1] + p2[1]) / 2, label,
                fontsize=8, va="center")


def main():
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("AI Data Science Team -- supervisor pattern (slide 226) and local reproduction",
                 fontsize=12, fontweight="bold", pad=12)

    user = box(ax, 0.02, 0.44, 0.16, 0.12, "USER\nnatural-language\nprompt", "#D9E6F2", bold=True)
    sup = box(ax, 0.30, 0.44, 0.20, 0.12, "SUPERVISOR\nplan + route +\nobserve + retry", "#FFE9B8", bold=True)
    arrow(ax, (0.18, 0.50), (0.30, 0.50), "prompt")
    arrow(ax, (0.40, 0.44), (0.40, 0.30), "")

    agents = [
        ("Data Loader\nTools Agent", 0.60, 0.78),
        ("Data Cleaning\nAgent", 0.60, 0.60),
        ("EDA Tools\nAgent", 0.60, 0.42),
        ("Data Viz\nAgent", 0.60, 0.24),
        ("Feature Eng.\nAgent", 0.60, 0.06),
    ]
    for name, x, y in agents:
        c = box(ax, x, y, 0.17, 0.11, name, "#E2EFDA")
        arrow(ax, (0.50, 0.50), (x, y + 0.055))
    ml = box(ax, 0.82, 0.42, 0.16, 0.11, "H2O ML Agent\n(sklearn stand-in\nin this demo)", "#FCE4D6", hatch="///")
    arrow(ax, (0.77, 0.50), (0.82, 0.475))
    ex = box(ax, 0.82, 0.66, 0.16, 0.11, "Code executor\n+ verification", "#EDEDED")
    arrow(ax, (0.77, 0.60), (0.82, 0.70))

    ax.text(0.30, 0.24, "Local reproduction substitutes (hatched / noted):\n"
            "- LLM supervisor router  ->  deterministic keyword/intent planner\n"
            "- LLM code generation  ->  templated code generation in agents.py\n"
            "- H2O AutoML  ->  scikit-learn RandomForest (QuickMLAgent)\n"
            "Executed code, retries and verification are real.",
            fontsize=8.5, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF7E6", edgecolor="#999999"))
    ax.text(0.02, 0.02, "Source: business-science/ai-data-science-team v0.0.0.9017, slide 226 of "
            "'Fall 2026 cmpe 255 2. Data Preparation'. No LLM ran in this reproduction.",
            fontsize=8, color="#555555")
    plt.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=130)
    plt.close(fig)
    print(f"Saved {FIG}")


if __name__ == "__main__":
    main()
