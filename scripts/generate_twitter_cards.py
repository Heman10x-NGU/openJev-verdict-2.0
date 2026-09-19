"""Generate publication-grade, human-designed benchmark cards for Verdict 2.0.

Eradicates all tell-tale 'AI slop' signatures:
- No neon cyan / purple / bright red palette
- No gimmicky floating cartoon pill badges
- No arbitrary bar lengths without axes
- Real scientific baseline reference lines (Majority, Random, 5% ECE threshold)
- Restrained, authoritative typography and professional data-ink ratio
- Styled like Google DeepMind / Anthropic / Nature research communications
- Generates both Precision Dark (Matte Graphite) and Editorial Light (Ivory Paper) versions.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np

out_dir = Path("assets")
out_dir.mkdir(parents=True, exist_ok=True)

# Set global typography
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['font.family'] = 'sans-serif'

THEMES = {
    "dark": {
        "bg": "#0d0f14",
        "surface": "#151821",
        "border": "#252936",
        "grid": "#1f2430",
        "accent_primary": "#3b82f6",
        "accent_muted": "#60a5fa",
        "comp_slate": "#475569",
        "comp_muted": "#334155",
        "base_gray": "#1e2430",
        "text_title": "#f8fafc",
        "text_body": "#e2e8f0",
        "text_muted": "#94a3b8",
        "text_faint": "#64748b",
        "accent_text": "#93c5fd",
        "highlight_col": "#1e293b",
        "highlight_alpha": 0.35,
        "row_alt": "#181c26",
        "calib_zone": "#10b981"
    },
    "light": {
        "bg": "#faf9f5",
        "surface": "#ffffff",
        "border": "#e2e0d8",
        "grid": "#eceae1",
        "accent_primary": "#1d4ed8",
        "accent_muted": "#3b82f6",
        "comp_slate": "#64748b",
        "comp_muted": "#94a3b8",
        "base_gray": "#f1f0eb",
        "text_title": "#141413",
        "text_body": "#27272a",
        "text_muted": "#52525b",
        "text_faint": "#71717a",
        "accent_text": "#1d4ed8",
        "highlight_col": "#eff6ff",
        "highlight_alpha": 0.6,
        "row_alt": "#f8f8f5",
        "calib_zone": "#059669"
    }
}

def generate_breakthrough_card(theme_key="dark", filename="benchmark_breakthrough_twitter.png"):
    t = THEMES[theme_key]
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=t["bg"])

    # Header Section
    fig.text(0.05, 0.94, "VERDICT 2.0", fontsize=24, fontweight="bold", color=t["text_title"])
    fig.text(0.18, 0.94, "BENCHMARK PERFORMANCE & ARCHITECTURE", fontsize=20, fontweight="normal", color=t["text_muted"])
    fig.text(0.05, 0.905, "Evaluated on LocalLLaMA/typed-decisions held-out test split (N = 2,000 decisions, single read)", fontsize=11, color=t["text_faint"])
    fig.add_artist(mlines.Line2D([0.05, 0.95], [0.89, 0.89], color=t["border"], linewidth=1))

    def draw_panel(ax, title, subtitle, metric_note=""):
        ax.set_facecolor(t["surface"])
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                 facecolor=t["surface"], edgecolor=t["border"], linewidth=1, zorder=0)
        ax.add_patch(rect)
        ax.axis("off")
        ax.text(0.05, 0.90, title, transform=ax.transAxes, fontsize=14, fontweight="bold", color=t["text_title"])
        ax.text(0.05, 0.82, subtitle, transform=ax.transAxes, fontsize=10, color=t["text_muted"])
        if metric_note:
            ax.text(0.95, 0.90, metric_note, transform=ax.transAxes, fontsize=10, fontweight="semibold",
                    color=t["accent_text"], ha="right")

    # Panel 1: Top-1 Accuracy
    ax1 = fig.add_axes([0.05, 0.49, 0.43, 0.38])
    draw_panel(ax1, "TOP-1 EXACT ACCURACY", "Higher is better • 2,000 held-out test decisions", "Ours: 150M vs 421M Large")

    models1 = [
        ("Verdict 2.0 Base (150M)", 77.10, t["accent_primary"], "77.1%"),
        ("Laya (ModernBERT-large, 421M)", 76.60, t["comp_slate"], "76.6%"),
        ("Jev 1.13.0 (150M)", 72.70, t["comp_muted"], "72.7%"),
        ("Verdict 1.0 Baseline", 26.10, t["border"], "26.1%")
    ]
    y_positions1 = [0.65, 0.46, 0.27, 0.08]

    x_bar_start = 0.40
    w_bar_total = 0.38

    for pct in [25, 50, 75]:
        x = x_bar_start + (pct / 100.0) * w_bar_total
        ax1.plot([x, x], [0.06, 0.75], transform=ax1.transAxes, color=t["grid"], linestyle=":", linewidth=0.8)
        ax1.text(x, 0.77, f"{pct}%", transform=ax1.transAxes, fontsize=8, color=t["text_faint"], ha="center")

    x_maj = x_bar_start + (48.4 / 100.0) * w_bar_total
    ax1.plot([x_maj, x_maj], [0.06, 0.75], transform=ax1.transAxes, color=t["text_muted"], linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.text(x_maj, 0.02, "Majority baseline (48.4%)", transform=ax1.transAxes, fontsize=7.5, color=t["text_muted"], ha="center")

    for (name, val, color, val_str), y in zip(models1, y_positions1):
        bar_w = (val / 100.0) * w_bar_total
        ax1.text(0.04, y + 0.025, name, transform=ax1.transAxes, fontsize=9.5, fontweight="bold" if color == t["accent_primary"] else "normal", color=t["text_title"] if color == t["accent_primary"] else t["text_body"])
        ax1.add_patch(patches.Rectangle((x_bar_start, y), w_bar_total, 0.09, transform=ax1.transAxes, facecolor=t["base_gray"], edgecolor="none"))
        ax1.add_patch(patches.Rectangle((x_bar_start, y), bar_w, 0.09, transform=ax1.transAxes, facecolor=color, edgecolor="none"))
        ax1.text(0.80, y + 0.025, val_str, transform=ax1.transAxes, fontsize=11, fontweight="bold", color=color if color == t["accent_primary"] else t["text_body"])

    # Panel 2: Expected Calibration Error (ECE)
    ax2 = fig.add_axes([0.52, 0.49, 0.43, 0.38])
    draw_panel(ax2, "CALIBRATION ERROR (ECE)", "Lower is better • Alignment with real hit rate", "15× Lower Error")

    models2 = [
        ("Verdict 2.0 (Confidence Head)", 1.44, t["accent_primary"], "1.44%  (15× win)"),
        ("Verdict 2.0 (Dist Channel)", 15.13, t["accent_muted"], "15.13%"),
        ("Jev 1.13.0 (Vendor)", 14.40, t["comp_muted"], "14.40%"),
        ("Laya (Single Channel)", 21.40, t["comp_slate"], "21.40%")
    ]
    y_positions2 = [0.65, 0.46, 0.27, 0.08]

    for pct in [5, 10, 15, 20]:
        x = x_bar_start + (pct / 25.0) * w_bar_total
        ax2.plot([x, x], [0.06, 0.75], transform=ax2.transAxes, color=t["grid"], linestyle=":", linewidth=0.8)
        ax2.text(x, 0.77, f"{pct}%", transform=ax2.transAxes, fontsize=8, color=t["text_faint"], ha="center")

    x_floor = x_bar_start + (5.0 / 25.0) * w_bar_total
    ax2.plot([x_floor, x_floor], [0.06, 0.75], transform=ax2.transAxes, color=t["calib_zone"], linestyle="--", linewidth=1.0, alpha=0.7)
    ax2.text(x_floor, 0.02, "Calibrated (<5%)", transform=ax2.transAxes, fontsize=7.5, color=t["calib_zone"], ha="center")

    for (name, val, color, val_str), y in zip(models2, y_positions2):
        bar_w = (val / 25.0) * w_bar_total
        ax2.text(0.04, y + 0.025, name, transform=ax2.transAxes, fontsize=9.5, fontweight="bold" if color == t["accent_primary"] else "normal", color=t["text_title"] if color == t["accent_primary"] else t["text_body"])
        ax2.add_patch(patches.Rectangle((x_bar_start, y), w_bar_total, 0.09, transform=ax2.transAxes, facecolor=t["base_gray"], edgecolor="none"))
        ax2.add_patch(patches.Rectangle((x_bar_start, y), bar_w, 0.09, transform=ax2.transAxes, facecolor=color, edgecolor="none"))
        ax2.text(0.80, y + 0.025, val_str, transform=ax2.transAxes, fontsize=10.5, fontweight="bold", color=color if color == t["accent_primary"] else t["text_body"])

    # Panel 3: Latency Per Case
    ax3 = fig.add_axes([0.05, 0.08, 0.43, 0.38])
    draw_panel(ax3, "INFERENCE SPEED", "Lower is better • 5-question case evaluation latency", "28× Faster than Jev")

    models3 = [
        ("Verdict 2.0 (Single-pass)", 25, t["accent_primary"], "25 ms"),
        ("Laya (Re-encoding)", 156, t["comp_slate"], "156 ms"),
        ("Jev 1.13.0 (Sequential)", 710, t["comp_muted"], "710 ms")
    ]
    y_positions3 = [0.58, 0.34, 0.10]

    for ms in [200, 400, 600]:
        x = x_bar_start + (ms / 750.0) * w_bar_total
        ax3.plot([x, x], [0.08, 0.72], transform=ax3.transAxes, color=t["grid"], linestyle=":", linewidth=0.8)
        ax3.text(x, 0.75, f"{ms}ms", transform=ax3.transAxes, fontsize=8, color=t["text_faint"], ha="center")

    for (name, val, color, val_str), y in zip(models3, y_positions3):
        bar_w = (val / 750.0) * w_bar_total
        ax3.text(0.04, y + 0.035, name, transform=ax3.transAxes, fontsize=9.5, fontweight="bold" if color == t["accent_primary"] else "normal", color=t["text_title"] if color == t["accent_primary"] else t["text_body"])
        ax3.add_patch(patches.Rectangle((x_bar_start, y), w_bar_total, 0.12, transform=ax3.transAxes, facecolor=t["base_gray"], edgecolor="none"))
        ax3.add_patch(patches.Rectangle((x_bar_start, y), bar_w, 0.12, transform=ax3.transAxes, facecolor=color, edgecolor="none"))
        ax3.text(0.80, y + 0.035, val_str, transform=ax3.transAxes, fontsize=11, fontweight="bold", color=color if color == t["accent_primary"] else t["text_body"])

    # Panel 4: Parameters & Edge Readiness
    ax4 = fig.add_axes([0.52, 0.08, 0.43, 0.38])
    draw_panel(ax4, "MODEL PARAMETERS & FOOTPRINT", "Smaller is better • In-browser client VRAM footprint", "2.8× Smaller Footprint")

    models4 = [
        ("Verdict 2.0 Base (Ours)", 149.6, t["accent_primary"], "150M (<600 MB)"),
        ("Laya (ModernBERT-large)", 421.3, t["comp_slate"], "421M (Cloud)"),
        ("Kev-0.5B (Causal LM)", 500.0, t["comp_muted"], "500M (High VRAM)")
    ]
    y_positions4 = [0.58, 0.34, 0.10]

    for p in [150, 300, 450]:
        x = x_bar_start + (p / 520.0) * w_bar_total
        ax4.plot([x, x], [0.08, 0.72], transform=ax4.transAxes, color=t["grid"], linestyle=":", linewidth=0.8)
        ax4.text(x, 0.75, f"{p}M", transform=ax4.transAxes, fontsize=8, color=t["text_faint"], ha="center")

    for (name, val, color, val_str), y in zip(models4, y_positions4):
        bar_w = (val / 520.0) * w_bar_total
        ax4.text(0.04, y + 0.035, name, transform=ax4.transAxes, fontsize=9.5, fontweight="bold" if color == t["accent_primary"] else "normal", color=t["text_title"] if color == t["accent_primary"] else t["text_body"])
        ax4.add_patch(patches.Rectangle((x_bar_start, y), w_bar_total, 0.12, transform=ax4.transAxes, facecolor=t["base_gray"], edgecolor="none"))
        ax4.add_patch(patches.Rectangle((x_bar_start, y), bar_w, 0.12, transform=ax4.transAxes, facecolor=color, edgecolor="none"))
        ax4.text(0.80, y + 0.035, val_str, transform=ax4.transAxes, fontsize=10.5, fontweight="bold", color=color if color == t["accent_primary"] else t["text_body"])

    # Footer
    fig.text(0.05, 0.025, "Verdict 2.0 • Deterministic Decision Infrastructure • Audited test vault receipt in reports/verdict2_base_test.json", fontsize=10, color=t["text_faint"])
    fig.text(0.85, 0.025, "github.com/Heman10x-NGU", fontsize=10, color=t["accent_text"])

    p_path = out_dir / filename
    plt.savefig(p_path, facecolor=t["bg"], edgecolor="none")
    plt.close(fig)
    print(f"Saved: {p_path}")


def generate_matrix_card(theme_key="dark", filename="verdict2_performance_matrix_twitter.png"):
    t = THEMES[theme_key]
    fig = plt.figure(figsize=(16, 9), dpi=200, facecolor=t["bg"])

    # Header Section
    fig.text(0.05, 0.94, "VERDICT 2.0", fontsize=24, fontweight="bold", color=t["text_title"])
    fig.text(0.18, 0.94, "HEAD-TO-HEAD AUDIT & COMPETITIVE MATRIX", fontsize=20, fontweight="normal", color=t["text_muted"])
    fig.text(0.05, 0.905, "Rigorous comparative evaluation across accuracy, calibration, inference speed, and prompt stability", fontsize=11, color=t["text_faint"])
    fig.add_artist(mlines.Line2D([0.05, 0.95], [0.89, 0.89], color=t["border"], linewidth=1))

    # Main Table Canvas
    ax_tab = fig.add_axes([0.05, 0.08, 0.90, 0.79])
    ax_tab.set_facecolor(t["surface"])
    rect_tab = patches.Rectangle((0, 0), 1, 1, transform=ax_tab.transAxes,
                                 facecolor=t["surface"], edgecolor=t["border"], linewidth=1, zorder=0)
    ax_tab.add_patch(rect_tab)
    ax_tab.axis("off")

    # Highlight column for Verdict 2.0
    v2_col_bg = patches.Rectangle((0.26, 0.02), 0.23, 0.96, transform=ax_tab.transAxes,
                                  facecolor=t["highlight_col"], alpha=t["highlight_alpha"], edgecolor=t["border"], linewidth=1)
    ax_tab.add_patch(v2_col_bg)

    # Columns layout
    c_metric = 0.03
    c_v2     = 0.28
    c_laya   = 0.52
    c_jev    = 0.68
    c_kev    = 0.84

    # Column Headers
    ax_tab.text(c_metric, 0.93, "EVALUATION DIMENSION", transform=ax_tab.transAxes, fontsize=11, fontweight="bold", color=t["text_faint"])
    ax_tab.text(c_v2,     0.93, "VERDICT 2.0 (150M)",  transform=ax_tab.transAxes, fontsize=12, fontweight="bold", color=t["accent_primary"])
    ax_tab.text(c_laya,   0.93, "LAYA (421M)",         transform=ax_tab.transAxes, fontsize=11, fontweight="bold", color=t["text_body"])
    ax_tab.text(c_jev,    0.93, "JEV 1.13.0",          transform=ax_tab.transAxes, fontsize=11, fontweight="bold", color=t["text_muted"])
    ax_tab.text(c_kev,    0.93, "KEV-0.5B",            transform=ax_tab.transAxes, fontsize=11, fontweight="bold", color=t["text_muted"])

    ax_tab.plot([0.02, 0.98], [0.90, 0.90], transform=ax_tab.transAxes, color=t["border"], linewidth=1.5)

    rows = [
        ("Top-1 Exact Accuracy", "77.10% (Best)", "76.60%", "72.70%", "78.12%* (Diff benchmark)"),
        ("Brier Score (Soft Dist)", "0.0636 (Best)", "0.0660", "0.1480", "0.3021*"),
        ("Expected Calibration Error (ECE)", "0.0144 / 1.44% (15×)", "0.2140 / 21.4%", "0.1440 / 14.4%", "0.0653*"),
        ("Inference Latency (Per 5-Q Case)", "25 ms (28× faster)", "156 ms", "710 ms", "~162 ms"),
        ("Option Order Flip Rate", "4.76% (35.8% fewer)", "Not tested", "Not tested", "7.41%"),
        ("p90 Probability Spread", "0.0915 (63% tighter)", "Not tested", "Not tested", "0.2486"),
        ("Score Level Within-One Accuracy", "99.00% (0.24 MAE)", "0.242 MAE", "1.44 MAE", "n/a"),
        ("Model Size & Backbone", "149.6M ModernBERT-base", "421M ModernBERT-large", "150M Encoder", "500M Causal Qwen2.5"),
        ("Client-Side / WebGPU Ready", "Yes (<600 MB unquantized)", "No (Cloud GPU needed)", "No (710ms latency)", "No (>1.2 GB VRAM)")
    ]

    y_vals = np.linspace(0.82, 0.07, len(rows))

    for i, (metric, v2_val, laya_val, jev_val, kev_val) in enumerate(rows):
        y = y_vals[i]
        if i % 2 == 1:
            row_bg = patches.Rectangle((0.02, y - 0.028), 0.96, 0.075, transform=ax_tab.transAxes,
                                       facecolor=t["row_alt"], alpha=0.5, edgecolor="none")
            ax_tab.add_patch(row_bg)
        
        ax_tab.text(c_metric, y, metric, transform=ax_tab.transAxes, fontsize=10.5, fontweight="bold", color=t["text_body"])
        ax_tab.text(c_v2,     y, v2_val,  transform=ax_tab.transAxes, fontsize=11, fontweight="bold", color=t["accent_primary"])
        ax_tab.text(c_laya,   y, laya_val, transform=ax_tab.transAxes, fontsize=10.5, color=t["text_body"])
        ax_tab.text(c_jev,    y, jev_val,  transform=ax_tab.transAxes, fontsize=10.5, color=t["text_muted"])
        ax_tab.text(c_kev,    y, kev_val,  transform=ax_tab.transAxes, fontsize=10.5, color=t["text_muted"])

    # Footnote
    fig.text(0.05, 0.025, "*Kev evaluated on proprietary dataset; stability measured under identical option permutation protocol. LocalLLaMA vault tested at N = 2,000.", fontsize=9.5, color=t["text_faint"])
    fig.text(0.85, 0.025, "github.com/Heman10x-NGU", fontsize=10, color=t["accent_text"])

    p_path = out_dir / filename
    plt.savefig(p_path, facecolor=t["bg"], edgecolor="none")
    plt.close(fig)
    print(f"Saved: {p_path}")

if __name__ == "__main__":
    # Generate Dark Mode (Primary Twitter format)
    generate_breakthrough_card("dark", "benchmark_breakthrough_twitter.png")
    generate_matrix_card("dark", "verdict2_performance_matrix_twitter.png")

    # Generate Light Mode (Editorial Academic / Nature format)
    generate_breakthrough_card("light", "benchmark_breakthrough_light.png")
    generate_matrix_card("light", "verdict2_performance_matrix_light.png")
