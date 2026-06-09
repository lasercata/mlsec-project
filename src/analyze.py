#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data Analyst — Adversarial Robustness Analysis
================================================
Loads JSON result files, produces comparative tables and professional figures:
  - Accuracy-vs-epsilon curves (per group + all shared)
  - Rank evolution bump chart with rank-shift highlights
  - Accuracy heatmaps
  - Clean vs Robust bar chart
  - Overview dashboard (2×2 grid)
"""

##-Imports
import json
import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────

# BASE_DIR = Path(__file__).resolve().parent
#
# JSON_FILES: dict[str, Path] = {
#     "Sehwag2021":    BASE_DIR / "outmodel1Sehwag.json",
#     "Carmon2019":    BASE_DIR / "outmodel2Carmon.json",
#     "Amini2024":     BASE_DIR / "outmodelAmini.json",
#     "Bartoldson2024": BASE_DIR / "outmodelBartoldson4.json",
#     "Rebuffi2021":   BASE_DIR / "outmodelRebuff.json",
# }
#
# OUT_DIR = BASE_DIR / "figures"
# OUT_DIR.mkdir(exist_ok=True)

# Epsilon values used for ALL figures
SHARED_EPS = [4 / 255, 8 / 255, 12 / 255, 16 / 255]
EPS_TARGET  = [0.0] + SHARED_EPS   # 0.0 = clean accuracy

# Color palette — one colour per model
# _PALETTE = sns.color_palette("tab10", len(JSON_FILES))
MODEL_COLORS: dict[str, tuple] = {}      # filled in main()

# Short labels for human-readable axes
def eps_label(eps: float) -> str:
    v = round(eps * 255, 4)
    n = int(v) if v == int(v) else v
    return f"{n}/255" if eps > 0 else "Clean"


# ─────────────────────────────────────────────────────────────────────────────
#  RC style
# ─────────────────────────────────────────────────────────────────────────────

RC = {
    "figure.facecolor":   "#FFFFFF",
    "axes.facecolor":     "#F7F8FA",
    "axes.edgecolor":     "#CCCCCC",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         "#DDDDDD",
    "grid.linestyle":     "--",
    "grid.alpha":         0.7,
    "axes.labelsize":     12,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "axes.titlepad":      12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "legend.framealpha":  0.92,
    "legend.edgecolor":   "#BBBBBB",
    "font.family":        "DejaVu Sans",
}

PCT_FMT = mticker.FuncFormatter(lambda v, _: f"{v:.0f}%")


def _save(out_dir: Path, fig: plt.Figure, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
#  Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_all(json_files: dict[str, Path]) -> dict:
    """Load all JSON files → unified dict keyed by short model name."""
    data = {}
    for short, path in json_files.items():
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        full_name = next(iter(raw))
        data[short] = {
            "full_name": full_name,
            "clean_acc": raw[full_name]["clean_acc"],
            "attacks":   raw[full_name]["attacks"],
            "mode":      raw[full_name]["attacks"][0]["mode"],
        }
    return data


def build_dataframe(data: dict) -> pd.DataFrame:
    """Build long-form DataFrame (model, eps, eps_h, adv_acc)."""
    rows = []
    for short, info in data.items():
        rows.append({"model": short, "eps": 0.0, "eps_h": "Clean", "adv_acc": info["clean_acc"]})
        for atk in info["attacks"]:
            rows.append({
                "model":   short,
                "eps":     atk["eps"],
                "eps_h":   atk["eps_h"],
                "adv_acc": atk["adv_acc"],
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
#  Ranking utilities
# ─────────────────────────────────────────────────────────────────────────────

def compute_ranks(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Return DataFrame[model × eps] = rank (1 = best accuracy)."""
    sub = df[df["model"].isin(models)]
    rows = []
    for eps, group in sub.groupby("eps"):
        sorted_g = group.sort_values("adv_acc", ascending=False).reset_index(drop=True)
        rank_map = {row["model"]: idx + 1 for idx, row in sorted_g.iterrows()}
        rows.append({"eps": eps, **rank_map})
    return pd.DataFrame(rows).set_index("eps").sort_index()


def detect_rank_shifts(ranks_df: pd.DataFrame, threshold: int = 1) -> list[dict]:
    """Return list of rank-shift events between consecutive epsilon values."""
    shifts = []
    eps_list = ranks_df.index.tolist()
    for i in range(1, len(eps_list)):
        e_prev, e_curr = eps_list[i - 1], eps_list[i]
        for model in ranks_df.columns:
            r_prev = int(ranks_df.loc[e_prev, model])
            r_curr = int(ranks_df.loc[e_curr, model])
            delta = r_prev - r_curr          # positive = improved
            if abs(delta) >= threshold:
                shifts.append({
                    "model":     model,
                    "eps_from":  e_prev,
                    "eps_to":    e_curr,
                    "rank_from": r_prev,
                    "rank_to":   r_curr,
                    "delta":     delta,
                })
    return shifts


# ─────────────────────────────────────────────────────────────────────────────
#  Console tables
# ─────────────────────────────────────────────────────────────────────────────

def _divider(char: str, width: int): return char * width


def print_accuracy_table(df: pd.DataFrame, models: list[str], title: str):
    sub = df[df["model"].isin(models)].copy()
    eps_vals = sorted(sub["eps"].unique())
    headers  = [eps_label(e) for e in eps_vals]

    col_w  = 12
    name_w = 22
    width  = name_w + col_w * len(headers)

    print()
    print(_divider("═", width))
    print(f"  {title}")
    print(_divider("═", width))
    print(f"{'Model':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in headers))
    print(_divider("─", width))

    for model in models:
        row = [f"{model:<{name_w}}"]
        for eps in eps_vals:
            match = sub[(sub["model"] == model) & (sub["eps"] == eps)]["adv_acc"]
            row.append(f"{match.values[0]*100:>{col_w-1}.1f}%" if len(match) else f"{'—':>{col_w}}")
        print("".join(row))

    print(_divider("─", width))
    print(f"  Ranks (1 = best robustness at that ε)")
    print(_divider("─", width))

    for model in models:
        row = [f"{model:<{name_w}}"]
        for eps in eps_vals:
            eps_sub = sub[sub["eps"] == eps].sort_values("adv_acc", ascending=False).reset_index(drop=True)
            rank_map = {r["model"]: i + 1 for i, r in eps_sub.iterrows()}
            rank = rank_map.get(model)
            row.append(f"{'#'+str(rank):>{col_w}}" if rank else f"{'—':>{col_w}}")
        print("".join(row))

    print(_divider("═", width))


def print_shift_report(shifts: list[dict], title: str):
    print()
    print(_divider("─", 62))
    print(f"  {title}")
    print(_divider("─", 62))
    if not shifts:
        print("  No rank shifts detected.")
    for s in shifts:
        arrow  = "↑ improved" if s["delta"] > 0 else "↓ dropped"
        e_from = eps_label(s["eps_from"])
        e_to   = eps_label(s["eps_to"])
        print(f"  {s['model']:<22s} {arrow} by {abs(s['delta'])} | "
              f"#{s['rank_from']} → #{s['rank_to']}  "
              f"(ε: {e_from} → {e_to})")
    print(_divider("─", 62))


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1 — Accuracy curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_curves(out_dir: Path, df: pd.DataFrame, models: list[str], title: str, filename: str):
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 6))

        for model in models:
            sub   = df[df["model"] == model].sort_values("eps")
            x     = sub["eps"].values * 255
            y     = sub["adv_acc"].values * 100
            color = MODEL_COLORS[model]

            ax.plot(x, y, "o-", color=color, linewidth=2.2, markersize=6,
                    label=model, zorder=3)

            # Annotate last point
            ax.annotate(
                f"{y[-1]:.1f}%",
                xy=(x[-1], y[-1]),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=color, va="center", fontweight="bold",
            )

        ax.set_xlabel("Perturbation budget  ε  (×/255)", labelpad=8)
        ax.set_ylabel("Adversarial Accuracy (%)", labelpad=8)
        ax.set_title(title)
        ax.yaxis.set_major_formatter(PCT_FMT)
        ax.set_ylim(0, 108)
        ax.legend(loc="upper right")
        fig.tight_layout()
        _save(out_dir, fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2 — Rank bump chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_rank_evolution(out_dir: Path, ranks_df: pd.DataFrame, shifts: list[dict],
                        title: str, filename: str):
    n_models = len(ranks_df.columns)
    x_vals   = ranks_df.index.values * 255

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Red shading between eps pairs where a shift occurs
        shift_pairs = {(s["eps_from"] * 255, s["eps_to"] * 255) for s in shifts}
        for x0, x1 in shift_pairs:
            ax.axvspan(x0, x1, color="#FF4C4C", alpha=0.08, zorder=0, label="_nolegend_")

        for model in ranks_df.columns:
            color = MODEL_COLORS[model]
            y     = ranks_df[model].values.astype(float)
            ax.plot(x_vals, y, "o-", color=color, linewidth=2.5, markersize=9,
                    label=model, zorder=3)
            for xi, yi in zip(x_vals, y):
                ax.text(xi, yi - 0.11, f"#{int(yi)}",
                        ha="center", va="top", fontsize=8,
                        color=color, fontweight="bold", zorder=4)

        # Annotate shifts mid-span
        already_annotated = set()
        for s in shifts:
            key = (s["model"], s["eps_from"], s["eps_to"])
            if key in already_annotated:
                continue
            already_annotated.add(key)
            x_mid = ((s["eps_from"] + s["eps_to"]) / 2) * 255
            y_mid = (s["rank_from"] + s["rank_to"]) / 2
            arrow = "↑" if s["delta"] > 0 else "↓"
            ax.annotate(
                f"{s['model']}: {arrow}{abs(s['delta'])}",
                xy=(x_mid, y_mid),
                fontsize=7.5, ha="center", color=MODEL_COLORS[s["model"]],
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=MODEL_COLORS[s["model"]], alpha=0.85),
            )

        ax.invert_yaxis()
        ax.set_yticks(range(1, n_models + 1))
        ax.set_yticklabels([f"Rank #{i}" for i in range(1, n_models + 1)], fontsize=10)
        ax.set_xlabel("Perturbation budget  ε  (×/255)", labelpad=8)
        ax.set_ylabel("Ranking  (1 = best)", labelpad=8)
        ax.set_title(title)

        # Legend with shift-zone patch
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Patch(color="#FF4C4C", alpha=0.25, label="Rank-shift zone"))
        labels.append("Rank-shift zone")
        ax.legend(handles, labels, loc="lower left")

        fig.tight_layout()
        _save(out_dir, fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3 — Accuracy heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmap(out_dir: Path, df: pd.DataFrame, models: list[str], title: str, filename: str):
    sub   = df[df["model"].isin(models)]
    pivot = (sub.pivot_table(index="model", columns="eps", values="adv_acc") * 100)
    pivot.columns = [eps_label(c) for c in pivot.columns]

    annot = pivot.map(lambda v: f"{v:.1f}%" if not np.isnan(v) else "—")

    with plt.rc_context(RC):
        fig_h = max(3, len(models) * 0.9)
        fig_w = max(8, len(pivot.columns) * 1.1)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        sns.heatmap(
            pivot, ax=ax,
            cmap="RdYlGn", annot=annot, fmt="",
            linewidths=0.6, linecolor="#CCCCCC",
            vmin=0, vmax=100,
            cbar_kws={"label": "Accuracy (%)", "shrink": 0.8},
        )
        ax.set_title(title)
        ax.set_xlabel("ε  (×/255)", labelpad=8)
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        _save(out_dir, fig, filename)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 4 — Clean vs Robust bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_clean_vs_robust(out_dir: Path, data: dict):
    models = list(data.keys())
    clean  = [data[m]["clean_acc"] * 100 for m in models]
    # Robust at smallest common ε (4/255)
    robust = []
    for m in models:
        atk_at_4 = [a for a in data[m]["attacks"] if abs(a["eps"] - 4/255) < 1e-6]
        robust.append(atk_at_4[0]["adv_acc"] * 100 if atk_at_4 else np.nan)

    x     = np.arange(len(models))
    width = 0.35

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(11, 6))
        colors = [MODEL_COLORS[m] for m in models]

        bars1 = ax.bar(x - width / 2, clean,  width, label="Clean accuracy",
                       color=colors, alpha=0.92, edgecolor="white", linewidth=1.2)
        bars2 = ax.bar(x + width / 2, robust, width, label="Robust accuracy  (ε = 4/255)",
                       color=colors, alpha=0.55, hatch="///",
                       edgecolor="white", linewidth=1.2)

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar in bars2:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=12, ha="right")
        ax.set_ylabel("Accuracy (%)")
        ax.yaxis.set_major_formatter(PCT_FMT)
        ax.set_ylim(0, 115)
        ax.set_title("Clean Accuracy vs. Robust Accuracy  (ε = 4/255)")
        ax.legend()
        fig.tight_layout()
        _save(out_dir, fig, "clean_vs_robust.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 5 — Overview dashboard (2 × 2)
# ─────────────────────────────────────────────────────────────────────────────

def plot_overview_dashboard(out_dir: Path, df: pd.DataFrame, data: dict):
    all_models  = list(data.keys())
    shared_eps  = [0.0] + SHARED_EPS
    df_shared   = df[df["eps"].round(8).isin([round(e, 8) for e in shared_eps])
                     & df["model"].isin(all_models)]

    ranks_all   = compute_ranks(df_shared, all_models)
    shifts_all  = detect_rank_shifts(ranks_all, threshold=1)

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(20, 13))
        fig.patch.set_facecolor("#FFFFFF")
        fig.suptitle(
            "Adversarial Robustness — Full Dashboard  (CIFAR-10 · L∞ threat model)",
            fontsize=18, fontweight="bold", y=1.005,
        )
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        # ── Panel 1: Accuracy curves (shared ε) ──────────────────────────
        for model in all_models:
            sub   = df_shared[df_shared["model"] == model].sort_values("eps")
            x     = sub["eps"].values * 255
            y     = sub["adv_acc"].values * 100
            color = MODEL_COLORS[model]
            ax1.plot(x, y, "o-", color=color, lw=2.2, ms=5.5, label=model, zorder=3)

        ax1.set_xlabel("ε  (×/255)")
        ax1.set_ylabel("Adversarial Accuracy (%)")
        ax1.set_title("Accuracy vs ε  —  All Models (shared ε)")
        ax1.yaxis.set_major_formatter(PCT_FMT)
        ax1.set_ylim(0, 108)
        ax1.legend(fontsize=8)

        # ── Panel 2: Rank bump chart ──────────────────────────────────────
        x_vals = ranks_all.index.values * 255
        shift_pairs = {(s["eps_from"] * 255, s["eps_to"] * 255) for s in shifts_all}
        for x0, x1 in shift_pairs:
            ax2.axvspan(x0, x1, color="#FF4C4C", alpha=0.10, zorder=0)

        for model in all_models:
            color = MODEL_COLORS[model]
            y     = ranks_all[model].values.astype(float)
            ax2.plot(x_vals, y, "o-", color=color, lw=2.2, ms=6, label=model, zorder=3)
            for xi, yi in zip(x_vals, y):
                ax2.text(xi, yi - 0.12, f"#{int(yi)}",
                         ha="center", va="top", fontsize=7,
                         color=color, fontweight="bold")

        ax2.invert_yaxis()
        ax2.set_yticks(range(1, len(all_models) + 1))
        ax2.set_yticklabels([f"#{i}" for i in range(1, len(all_models) + 1)])
        ax2.set_xlabel("ε  (×/255)")
        ax2.set_ylabel("Rank  (1 = best)")
        ax2.set_title("Rank Evolution  —  All Models")
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles2.append(Patch(color="#FF4C4C", alpha=0.3, label="Rank-shift zone"))
        labels2.append("Rank-shift zone")
        ax2.legend(handles2, labels2, fontsize=8, loc="lower left")

        # ── Panel 3: Heatmap ──────────────────────────────────────────────
        pivot = df_shared[df_shared["model"].isin(all_models)].pivot_table(
            index="model", columns="eps", values="adv_acc") * 100
        pivot.columns = [eps_label(c) for c in pivot.columns]
        annot = pivot.map(lambda v: f"{v:.1f}%" if not np.isnan(v) else "—")
        sns.heatmap(
            pivot, ax=ax3, cmap="RdYlGn", annot=annot, fmt="",
            linewidths=0.5, linecolor="#CCCCCC",
            vmin=0, vmax=100,
            cbar_kws={"label": "Accuracy (%)", "shrink": 0.85},
        )
        ax3.set_title("Accuracy Heatmap  (shared ε)")
        ax3.set_xlabel("ε  (×/255)")
        ax3.set_ylabel("")
        plt.setp(ax3.get_xticklabels(), rotation=35, ha="right")
        plt.setp(ax3.get_yticklabels(), rotation=0)

        # ── Panel 4: Clean vs Robust bars ─────────────────────────────────
        models = all_models
        clean  = [data[m]["clean_acc"] * 100 for m in models]
        robust = []
        for m in models:
            atk4 = [a for a in data[m]["attacks"] if abs(a["eps"] - 4/255) < 1e-6]
            robust.append(atk4[0]["adv_acc"] * 100 if atk4 else np.nan)

        x     = np.arange(len(models))
        width = 0.35
        colors = [MODEL_COLORS[m] for m in models]
        ax4.bar(x - width / 2, clean,  width, color=colors, alpha=0.92,
                label="Clean", edgecolor="white")
        ax4.bar(x + width / 2, robust, width, color=colors, alpha=0.55,
                hatch="///", label="Robust (ε=4/255)", edgecolor="white")
        for xi, ci, ri in zip(x, clean, robust):
            ax4.text(xi - width / 2, ci + 0.8, f"{ci:.1f}%",
                     ha="center", va="bottom", fontsize=7.5, fontweight="bold")
            if not np.isnan(ri):
                ax4.text(xi + width / 2, ri + 0.8, f"{ri:.1f}%",
                         ha="center", va="bottom", fontsize=7.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(models, rotation=12, ha="right", fontsize=9)
        ax4.set_ylim(0, 115)
        ax4.set_ylabel("Accuracy (%)")
        ax4.yaxis.set_major_formatter(PCT_FMT)
        ax4.set_title("Clean vs Robust Accuracy  (ε = 4/255)")
        ax4.legend(fontsize=9)

        fig.tight_layout()
        _save(out_dir, fig, "dashboard_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(json_files: dict[str, Path], out_dir: Path):
    global MODEL_COLORS

    print("\n" + "█" * 64)
    print("  ADVERSARIAL ROBUSTNESS — DATA ANALYSIS")
    print("█" * 64)

    # 1. Load & build DataFrame
    data = load_all(json_files)
    df   = build_dataframe(data)
    # Keep only the 4 target epsilon values (+ clean) for all figures
    df   = df[df["eps"].isin(EPS_TARGET)].copy()
    all_models = list(data.keys())

    # Assign stable colours per model
    _PALETTE = sns.color_palette("tab10", len(json_files))
    MODEL_COLORS = {m: _PALETTE[i] for i, m in enumerate(all_models)}

    # Group by attack mode
    mode0 = [m for m in all_models if data[m]["mode"] == 0]   # Sehwag, Carmon
    mode1 = [m for m in all_models if data[m]["mode"] == 1]   # Amini, Bartoldson, Rebuffi

    # ── Console tables ────────────────────────────────────────────────────────

    if mode0:
        print_accuracy_table(df, mode0,
            "Mode 0 — APGD-CE  (Sehwag, Carmon)  ·  full ε sweep  1–18/255")
    if mode1:
        print_accuracy_table(df, mode1,
            "Mode 1 — APGD-CE + APGD-T  (Amini, Bartoldson, Rebuffi)  ·  ε = 4,8,12,16/255")

    # Shared-ε table (all models)
    df_s = df[df["eps"].isin([0.0] + SHARED_EPS)]
    print_accuracy_table(df_s, all_models,
        "All Models — Shared ε values  (Clean, 4, 8, 12, 16/255)")

    # ── Rank shifts ───────────────────────────────────────────────────────────

    if mode0:
        ranks_m0  = compute_ranks(df[df["model"].isin(mode0)], mode0)
        shifts_m0 = detect_rank_shifts(ranks_m0)
        print_shift_report(shifts_m0, "Rank Shifts — Mode 0 (Sehwag vs Carmon)")

    if mode1:
        ranks_m1  = compute_ranks(df[df["model"].isin(mode1)], mode1)
        shifts_m1 = detect_rank_shifts(ranks_m1)
        print_shift_report(shifts_m1, "Rank Shifts — Mode 1 (Amini, Bartoldson, Rebuffi)")

    ranks_all  = compute_ranks(df_s, all_models)
    shifts_all = detect_rank_shifts(ranks_all)
    print_shift_report(shifts_all, "Rank Shifts — All Models on Shared ε")

    # ── Figures ───────────────────────────────────────────────────────────────

    print()
    print(_divider("─", 64))
    print("  GENERATING FIGURES  →  ./figures/")
    print(_divider("─", 64))

    if mode0:
        df_m0 = df[df["model"].isin(mode0)]
        plot_accuracy_curves(out_dir, df_m0, mode0,
            "Accuracy vs ε  —  Mode 0 Models  (APGD-CE · L∞ · CIFAR-10)",
            "curves_mode0.png")
        plot_rank_evolution(out_dir, ranks_m0, shifts_m0,
            "Rank Evolution  —  Mode 0 Models", "ranks_mode0.png")
        plot_heatmap(out_dir, df_m0, mode0,
            "Accuracy Heatmap  —  Mode 0  (Sehwag, Carmon)", "heatmap_mode0.png")

    if mode1:
        df_m1 = df[df["model"].isin(mode1)]
        plot_accuracy_curves(out_dir, df_m1, mode1,
            "Accuracy vs ε  —  Mode 1 Models  (APGD-CE + APGD-T · L∞ · CIFAR-10)",
            "curves_mode1.png")
        plot_rank_evolution(out_dir, ranks_m1, shifts_m1,
            "Rank Evolution  —  Mode 1 Models", "ranks_mode1.png")
        plot_heatmap(out_dir, df_m1, mode1,
            "Accuracy Heatmap  —  Mode 1  (Amini, Bartoldson, Rebuffi)",
            "heatmap_mode1.png")

    # All-model comparison on shared ε
    plot_accuracy_curves(out_dir, df_s, all_models,
        "All Models — Shared ε Values  (L∞ · CIFAR-10)",
        "curves_all_shared.png")
    plot_rank_evolution(out_dir, ranks_all, shifts_all,
        "Rank Evolution  —  All Models  (Shared ε)", "ranks_all_shared.png")
    plot_heatmap(out_dir, df_s, all_models,
        "Accuracy Heatmap  —  All Models  (Shared ε)", "heatmap_all_shared.png")

    # Clean vs Robust
    plot_clean_vs_robust(out_dir, data)

    # Dashboard
    plot_overview_dashboard(out_dir, df, data)

    print()
    print("  All figures written to:", out_dir)
    print("  Analysis complete.")
    print(_divider("█", 64))


if __name__ == "__main__":
    main()
