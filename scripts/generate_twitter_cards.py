"""Generate pixel-perfect, high-impact Twitter benchmark cards for Verdict 2.0."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

out_dir = Path("assets")
out_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Style Setup: Dark Mode Modern Cyberpunk / Deep Slate
# -------------------------------------------------------------
BG_COLOR = "#080c15"        # Deep void slate
CARD_BG = "#0f172a"         # Slate-900 elevated card
BORDER_COLOR = "#1e293b"    # Slate-800 border
ACCENT_CYAN = "#00f2ff"     # Verdict 2.0 signature cyan
ACCENT_GREEN = "#10b981"    # Success emerald
ACCENT_PURPLE = "#a855f7"   # Modern violet
TEXT_WHITE = "#ffffff"
TEXT_SUB = "#cbd5e1"
TEXT_MUTED = "#94a3b8"
BAR_TRACK = "#1e293b"
BAR_COMPETITOR = "#475569"
BAR_WEAK = "#ef4444"

# =============================================================
# CARD 1: 4 Key Metric Showdown (Twitter 16:9, 1920x1080)
# =============================================================
fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=BG_COLOR)

# Header
fig.text(0.05, 0.94, "VERDICT 2.0", fontsize=30, fontweight="bold", color=ACCENT_CYAN)
fig.text(0.24, 0.94, "vs  LAYA & JEV: BENCHMARK BREAKDOWN", fontsize=26, fontweight="bold", color=TEXT_WHITE)
fig.text(0.05, 0.90, "Benchmark: LocalLLaMA/typed-decisions (Test Split, N = 2,000 decisions across 4 enterprise workflows)", fontsize=13, color=TEXT_MUTED)

# 4 Panels using exact add_axes
# [left, bottom, width, height]

# Panel 1: Top-1 Accuracy
ax1 = fig.add_axes([0.05, 0.50, 0.43, 0.36], facecolor=CARD_BG)
rect1 = patches.FancyBboxPatch((0, 0), 1, 1, transform=ax1.transAxes, boxstyle="round,pad=0.02,rounding_size=0.04",
                              facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.5, zorder=0)
ax1.add_patch(rect1)
ax1.axis("off")
ax1.text(0.05, 0.90, "1. TOP-1 ACCURACY", transform=ax1.transAxes, fontsize=15, fontweight="bold", color=TEXT_WHITE)
ax1.text(0.05, 0.82, "Higher is better (%) • 2,000 test decisions", transform=ax1.transAxes, fontsize=11, color=TEXT_MUTED)
ax1.text(0.62, 0.88, "+0.5% vs 421M Large", transform=ax1.transAxes, fontsize=10, fontweight="bold", color=BG_COLOR,
         bbox=dict(boxstyle="round,pad=0.3", facecolor=ACCENT_CYAN, edgecolor="none"))

models1 = ["Verdict 2.0 (Base 150M)", "Laya (Large 421M)", "Jev 1.13.0", "Verdict 1.0 Baseline"]
vals1 = [77.10, 76.60, 72.70, 26.10]
cols1 = [ACCENT_CYAN, "#6366f1", BAR_COMPETITOR, BAR_WEAK]
y_coords1 = [0.65, 0.46, 0.27, 0.08]

for m, v, c, y in zip(models1, vals1, cols1, y_coords1):
    norm_len = (v / 85.0) * 0.70
    ax1.add_patch(patches.FancyBboxPatch((0.05, y), 0.70, 0.10, transform=ax1.transAxes, boxstyle="round,pad=0.01", facecolor=BAR_TRACK, edgecolor="none"))
    ax1.add_patch(patches.FancyBboxPatch((0.05, y), norm_len, 0.10, transform=ax1.transAxes, boxstyle="round,pad=0.01", facecolor=c, edgecolor="none"))
    ax1.text(0.06, y + 0.03, m, transform=ax1.transAxes, fontsize=11, fontweight="bold" if c == ACCENT_CYAN else "normal", color=TEXT_WHITE)
    ax1.text(0.78, y + 0.03, f"{v:.1f}%", transform=ax1.transAxes, fontsize=13, fontweight="bold", color=c)

# Panel 2: Expected Calibration Error (ECE)
ax2 = fig.add_axes([0.52, 0.50, 0.43, 0.36], facecolor=CARD_BG)
rect2 = patches.FancyBboxPatch((0, 0), 1, 1, transform=ax2.transAxes, boxstyle="round,pad=0.02,rounding_size=0.04",
                              facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.5, zorder=0)
ax2.add_patch(rect2)
ax2.axis("off")
ax2.text(0.05, 0.90, "2. CALIBRATION ERROR (ECE)", transform=ax2.transAxes, fontsize=15, fontweight="bold", color=TEXT_WHITE)
ax2.text(0.05, 0.82, "Lower is better (%) • Calibration error", transform=ax2.transAxes, fontsize=11, color=TEXT_MUTED)
ax2.text(0.60, 0.88, "15x LOWER ERROR (1.44%)", transform=ax2.transAxes, fontsize=10, fontweight="bold", color=BG_COLOR,
         bbox=dict(boxstyle="round,pad=0.3", facecolor=ACCENT_GREEN, edgecolor="none"))

models2 = ["Verdict 2.0 (Confidence Head)", "Verdict 2.0 (Dist Channel)", "Jev 1.13.0", "Laya (Single Channel)"]
vals2 = [1.44, 15.13, 14.40, 21.40]
cols2 = [ACCENT_GREEN, "#6366f1", BAR_COMPETITOR, BAR_WEAK]
y_coords2 = [0.65, 0.46, 0.27, 0.08]

for m, v, c, y in zip(models2, vals2, cols2, y_coords2):
    norm_len = max(0.03, (v / 24.0) * 0.70)
    ax2.add_patch(patches.FancyBboxPatch((0.05, y), 0.70, 0.10, transform=ax2.transAxes, boxstyle="round,pad=0.01", facecolor=BAR_TRACK, edgecolor="none"))
    ax2.add_patch(patches.FancyBboxPatch((0.05, y), norm_len, 0.10, transform=ax2.transAxes, boxstyle="round,pad=0.01", facecolor=c, edgecolor="none"))
    ax2.text(0.06, y + 0.03, m, transform=ax2.transAxes, fontsize=11, fontweight="bold" if c == ACCENT_GREEN else "normal", color=TEXT_WHITE)
    ax2.text(0.78, y + 0.03, f"{v:.2f}%", transform=ax2.transAxes, fontsize=13, fontweight="bold", color=c)

# Panel 3: Latency Per Case
ax3 = fig.add_axes([0.05, 0.08, 0.43, 0.36], facecolor=CARD_BG)
rect3 = patches.FancyBboxPatch((0, 0), 1, 1, transform=ax3.transAxes, boxstyle="round,pad=0.02,rounding_size=0.04",
                              facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.5, zorder=0)
ax3.add_patch(rect3)
ax3.axis("off")
ax3.text(0.05, 0.90, "3. INFERENCE SPEED", transform=ax3.transAxes, fontsize=15, fontweight="bold", color=TEXT_WHITE)
ax3.text(0.05, 0.82, "Lower is better • 5-question case latency (ms)", transform=ax3.transAxes, fontsize=11, color=TEXT_MUTED)
ax3.text(0.60, 0.88, "28x FASTER THAN JEV", transform=ax3.transAxes, fontsize=10, fontweight="bold", color=BG_COLOR,
         bbox=dict(boxstyle="round,pad=0.3", facecolor=ACCENT_CYAN, edgecolor="none"))

models3 = ["Verdict 2.0 (Single-pass)", "Laya (Re-encoding)", "Jev 1.13.0"]
vals3 = [25, 156, 710]
cols3 = [ACCENT_CYAN, BAR_COMPETITOR, BAR_WEAK]
y_coords3 = [0.60, 0.36, 0.12]

for m, v, c, y in zip(models3, vals3, cols3, y_coords3):
    norm_len = max(0.03, (v / 750.0) * 0.70)
    ax3.add_patch(patches.FancyBboxPatch((0.05, y), 0.70, 0.12, transform=ax3.transAxes, boxstyle="round,pad=0.01", facecolor=BAR_TRACK, edgecolor="none"))
    ax3.add_patch(patches.FancyBboxPatch((0.05, y), norm_len, 0.12, transform=ax3.transAxes, boxstyle="round,pad=0.01", facecolor=c, edgecolor="none"))
    ax3.text(0.06, y + 0.035, m, transform=ax3.transAxes, fontsize=11, fontweight="bold" if c == ACCENT_CYAN else "normal", color=TEXT_WHITE)
    ax3.text(0.78, y + 0.035, f"{v} ms", transform=ax3.transAxes, fontsize=13, fontweight="bold", color=c)

# Panel 4: Parameters & Model Footprint
ax4 = fig.add_axes([0.52, 0.08, 0.43, 0.36], facecolor=CARD_BG)
rect4 = patches.FancyBboxPatch((0, 0), 1, 1, transform=ax4.transAxes, boxstyle="round,pad=0.02,rounding_size=0.04",
                              facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.5, zorder=0)
ax4.add_patch(rect4)
ax4.axis("off")
ax4.text(0.05, 0.90, "4. PARAMETERS & FOOTPRINT", transform=ax4.transAxes, fontsize=15, fontweight="bold", color=TEXT_WHITE)
ax4.text(0.05, 0.82, "Smaller footprint • WebGPU browser ready", transform=ax4.transAxes, fontsize=11, color=TEXT_MUTED)
ax4.text(0.58, 0.88, "2.8x SMALLER FOOTPRINT", transform=ax4.transAxes, fontsize=10, fontweight="bold", color=BG_COLOR,
         bbox=dict(boxstyle="round,pad=0.3", facecolor=ACCENT_PURPLE, edgecolor="none"))

models4 = ["Verdict 2.0 Base (Ours)", "Laya (ModernBERT-large)", "Kev-0.5B (Causal)"]
vals4 = [150, 421, 500]
cols4 = [ACCENT_PURPLE, BAR_COMPETITOR, BAR_COMPETITOR]
y_coords4 = [0.60, 0.36, 0.12]

for m, v, c, y in zip(models4, vals4, cols4, y_coords4):
    norm_len = max(0.03, (v / 550.0) * 0.70)
    ax4.add_patch(patches.FancyBboxPatch((0.05, y), 0.70, 0.12, transform=ax4.transAxes, boxstyle="round,pad=0.01", facecolor=BAR_TRACK, edgecolor="none"))
    ax4.add_patch(patches.FancyBboxPatch((0.05, y), norm_len, 0.12, transform=ax4.transAxes, boxstyle="round,pad=0.01", facecolor=c, edgecolor="none"))
    ax4.text(0.06, y + 0.035, m, transform=ax4.transAxes, fontsize=11, fontweight="bold" if c == ACCENT_PURPLE else "normal", color=TEXT_WHITE)
    ax4.text(0.78, y + 0.035, f"{v}M", transform=ax4.transAxes, fontsize=13, fontweight="bold", color=c)

# Footer
fig.text(0.05, 0.03, "Verdict 2.0 • Non-autoregressive Calibrated Decision Engine • Audited on LocalLLaMA/typed-decisions", fontsize=11, color="#64748b")
fig.text(0.84, 0.03, "github.com/Heman10x-NGU", fontsize=11, color=ACCENT_CYAN)

p1_path = out_dir / "benchmark_breakthrough_twitter.png"
plt.savefig(p1_path, facecolor=BG_COLOR, edgecolor="none")
plt.close(fig)
print(f"Saved: {p1_path}")


# =============================================================
# CARD 2: The Full Head-to-Head Competitive Matrix
# =============================================================
fig2 = plt.figure(figsize=(16, 9), dpi=200, facecolor=BG_COLOR)

# Header
fig2.text(0.05, 0.94, "VERDICT 2.0", fontsize=30, fontweight="bold", color=ACCENT_CYAN)
fig2.text(0.24, 0.94, "HEAD-TO-HEAD COMPETITIVE MATRIX", fontsize=26, fontweight="bold", color=TEXT_WHITE)
fig2.text(0.05, 0.90, "Why mathematical calibration and encoder architecture beat bloated generative models", fontsize=13, color=TEXT_MUTED)

# Main Card
ax_table = fig2.add_axes([0.05, 0.08, 0.90, 0.78], facecolor=CARD_BG)
rect_table = patches.FancyBboxPatch((0, 0), 1, 1, transform=ax_table.transAxes, boxstyle="round,pad=0.02,rounding_size=0.03",
                                    facecolor=CARD_BG, edgecolor=BORDER_COLOR, linewidth=1.5, zorder=0)
ax_table.add_patch(rect_table)
ax_table.axis("off")

# Column X-coordinates
c_metric = 0.04
c_v2     = 0.32
c_laya   = 0.54
c_jev    = 0.69
c_kev    = 0.83

# Headers
ax_table.text(c_metric, 0.92, "DIMENSION / METRIC", transform=ax_table.transAxes, fontsize=12, fontweight="bold", color=TEXT_MUTED)
ax_table.text(c_v2,     0.92, "VERDICT 2.0 (150M)",  transform=ax_table.transAxes, fontsize=13, fontweight="bold", color=ACCENT_CYAN)
ax_table.text(c_laya,   0.92, "LAYA (421M)",         transform=ax_table.transAxes, fontsize=12, fontweight="bold", color=TEXT_WHITE)
ax_table.text(c_jev,    0.92, "JEV 1.13.0",          transform=ax_table.transAxes, fontsize=12, fontweight="bold", color=TEXT_MUTED)
ax_table.text(c_kev,    0.92, "KEV-0.5B",            transform=ax_table.transAxes, fontsize=12, fontweight="bold", color=TEXT_MUTED)

ax_table.plot([0.02, 0.98], [0.89, 0.89], transform=ax_table.transAxes, color="#334155", linewidth=1.5)

rows_data = [
    ("Top-1 Exact Accuracy", "77.10% (Best)", "76.60%", "72.70%", "78.12%* (Diff bench)"),
    ("Brier Score (Soft Dist)", "0.0636 (Best)", "0.0660", "0.1480", "0.3021*"),
    ("Calibration Error (ECE)", "0.0144 / 1.44% (15x Win)", "0.2140 / 21.4%", "0.1440 / 14.4%", "0.0653*"),
    ("Inference Latency", "~25 ms (28x Faster)", "~156 ms", "710 ms", "~162 ms"),
    ("Option Order Flip Rate", "4.76% (35.8% Fewer Flips)", "Not tested", "Not tested", "7.41%"),
    ("p90 Probability Spread", "0.0915 (63% Tighter)", "Not tested", "Not tested", "0.2486"),
    ("Score Level Within-One", "99.00% (0.24 MAE)", "0.242 MAE", "1.44 MAE", "n/a"),
    ("Architecture & Size", "149.6M Encoder (Single-pass)", "421M Large Encoder", "150M Encoder", "500M Causal Decoder"),
    ("In-Browser WebGPU Ready", "YES (<600 MB)", "NO (Requires Cloud)", "NO (High Latency)", "NO (High VRAM)")
]

y_lines = np.linspace(0.81, 0.08, len(rows_data))
for i, (metric, v2_val, laya_val, jev_val, kev_val) in enumerate(rows_data):
    y = y_lines[i]
    if i % 2 == 0:
        row_box = patches.Rectangle((0.02, y - 0.025), 0.96, 0.07, transform=ax_table.transAxes,
                                    facecolor="#1e293b", alpha=0.4, edgecolor="none")
        ax_table.add_patch(row_box)
    
    ax_table.text(c_metric, y, metric, transform=ax_table.transAxes, fontsize=11, fontweight="bold", color=TEXT_WHITE)
    ax_table.text(c_v2,     y, v2_val,  transform=ax_table.transAxes, fontsize=11, fontweight="bold", color=ACCENT_CYAN)
    ax_table.text(c_laya,   y, laya_val, transform=ax_table.transAxes, fontsize=11, color=TEXT_SUB)
    ax_table.text(c_jev,    y, jev_val,  transform=ax_table.transAxes, fontsize=11, color=TEXT_MUTED)
    ax_table.text(c_kev,    y, kev_val,  transform=ax_table.transAxes, fontsize=11, color=TEXT_MUTED)

fig2.text(0.05, 0.03, "*Kev accuracy evaluated on proprietary NLP suite; stability measured on identical protocol. Verdict 2.0 Base tested on LocalLLaMA vault.", fontsize=11, color="#64748b")
fig2.text(0.84, 0.03, "github.com/Heman10x-NGU", fontsize=11, color=ACCENT_CYAN)

p2_path = out_dir / "verdict2_performance_matrix_twitter.png"
plt.savefig(p2_path, facecolor=BG_COLOR, edgecolor="none")
plt.close(fig2)
print(f"Saved: {p2_path}")
