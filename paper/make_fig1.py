"""Fig. 1 for the AIEI 2027 camera-ready: causal-ablation dangerous-error rates with short, horizontal labels."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
(HERE / "figures").mkdir(exist_ok=True)

labels = ["Full\n(periodic\nAISOS)", "−Authority", "−Authority\n−Trust tier", "−Authority\n−Trust tier\n+Dream off"]
rates = [0.0, 0.0, 0.0, 1.0]
colors = ["#2f6fb5", "#2f6fb5", "#9aa5b4", "#c8506b"]

plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(figsize=(3.4, 1.9))
bars = ax.bar(range(len(rates)), rates, width=0.6, color=colors)
for b, v in zip(bars, rates):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(range(len(rates)))
ax.set_xticklabels(labels, fontsize=6.8)
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_ylabel("Dangerous-error rate", fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", color="#e3e7ee", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout(pad=0.3)
fig.savefig(HERE / "figures" / "fig1_ablation.pdf")
fig.savefig(HERE / "figures" / "fig1_ablation.png", dpi=300)
print("ok")
