"""The two charts. Plain matplotlib, one shared look."""

import matplotlib

matplotlib.use("Agg")   # draw straight to a file, no window
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


MODEL_NAMES = {"linear": "One-layer baseline", "cnn": "This CNN", "their-gru": "UC Davis GRU, their weights"}


def _new_chart(title, subtitle, size, panels=1):
    """A figure with `panels` plots side by side on one shared y scale. Returns (fig, list of axes)."""
    fig, axes = plt.subplots(1, panels, figsize=size, dpi=160, sharey=True, squeeze=False)
    axes = list(axes[0])
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.suptitle(title, x=0.02, y=0.975, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=9, color=INK_SOFT)
    return fig, axes


def latency_chart(results, path):
    """One row per model and device: typical, p99 and worst step time on a log scale."""
    rows = list(results.items())[::-1]   # first result at the top
    fig, (ax,) = _new_chart("Time for one decoding step",
                            "Each row is one model on one processor. Log scale, milliseconds.",
                            (9.6, 1.9 + 0.5 * len(rows)))
    for y, (label, r) in enumerate(rows):
        ax.plot([r["typical"], r["worst"]], [y, y], color=GRID, linewidth=2, zorder=1, solid_capstyle="round")
        ax.plot([r["budget_ms"]] * 2, [y - 0.24, y + 0.24], color=INK_SOFT, linewidth=1.6, zorder=2)
        for key, colour in (("worst", AQUA), ("p99", ORANGE), ("typical", BLUE)):
            ax.scatter(r[key], y, s=46, color=colour, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        # Numbers sit in their own column to the right, so they never land on a mark.
        ax.annotate(f"{r['typical']:.2f} / {r['p99']:.2f} / {r['worst']:.1f} ms", xy=(1.02, y),
                    xycoords=("axes fraction", "data"), va="center", fontsize=8.5, color=INK_SOFT,
                    annotation_clip=False)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows], color=INK, fontsize=9)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xscale("log")
    low = min(r["typical"] for _, r in rows)
    high = max(max(r["worst"], r["budget_ms"]) for _, r in rows)
    ax.set_xlim(low / 2, high * 2)
    ax.set_xlabel("milliseconds per step", color=INK_SOFT, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, markersize=7, label=name)
               for name, c in (("typical (median)", BLUE), ("p99", ORANGE), ("worst", AQUA))]
    handles.append(plt.Line2D([], [], marker="|", linestyle="", color=INK_SOFT, markersize=11,
                              markeredgewidth=1.6, label="deadline: next step is due"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_SOFT, ncols=4,
              bbox_to_anchor=(1.28, 1.0), handletextpad=0.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.24, right=0.78, top=0.9 - 0.35 / fig.get_figheight(), bottom=0.6 / fig.get_figheight())
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def size_chart(table, path):
    """Phoneme error rate against the size of the 8-bit model file, one point per width.

    table is sizes.py's {width: row}. Each width was trained several times; the dot is
    the mean and the bar runs from the best run to the worst.
    """
    rows = sorted(table.items(), key=lambda item: item[1]["int8_bytes"])
    runs_each = len(rows[0][1]["runs"])
    fig, (ax,) = _new_chart("How small can the model get?",
                            f"Phoneme error rate on the 267 validation sentences against the size of the model file. "
                            f"Dot: mean of {runs_each} training runs. Bar: best to worst run.", (9.6, 4.8))
    for kind, colour, name, nudge in (("float_per", ORANGE, "32-bit floats (same model, 4x the bytes)", 0.97),
                                      ("int8_per", BLUE, "8-bit integers", 1.0)):
        x = [row["int8_bytes"] / 1000 * nudge for _, row in rows]
        runs = [[100 * run[kind] for run in row["runs"]] for _, row in rows]
        means = [sum(r) / len(r) for r in runs]
        for xi, r in zip(x, runs):
            ax.plot([xi, xi], [min(r), max(r)], color=colour, linewidth=2, alpha=0.35, solid_capstyle="round", zorder=1)
        ax.plot(x, means, color=colour, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1, label=name, zorder=3 if kind == "int8_per" else 2)
        if kind == "int8_per":
            for xi, mean, (width, row) in zip(x, means, rows):
                ax.annotate(f"width {width}\n{row['int8_bytes'] / 1000:,.0f} kB, {mean:.0f}%", (xi, mean), xytext=(7, 9),
                            textcoords="offset points", fontsize=8.5, color=INK_SOFT)
    ax.set_xscale("log")
    ax.set_xlim(rows[0][1]["int8_bytes"] / 1000 / 1.5, rows[-1][1]["int8_bytes"] / 1000 * 2.6)
    ax.set_ylim(0, 100)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f} kB")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_xlabel("size of the 8-bit model file (log scale)", color=INK_SOFT, fontsize=9)
    ax.set_ylabel("phoneme error rate", color=INK_SOFT, fontsize=9)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5, labelcolor=INK_SOFT)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.12)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def drop_chart(results, path):
    """Phoneme error rate against the share of frames lost, one line per thing the receiver can do.

    results is one drop_test result, or a list of them to draw side by side on the same scale.
    """
    panels = results if isinstance(results, list) else [results]
    names = {"ignore": "ignore the gap", "zeros": "fill with zeros", "hold": "repeat last row"}
    colours = {"ignore": BLUE, "zeros": ORANGE, "hold": AQUA}
    burst = panels[0]["burst"]
    how = "one at a time" if burst == 1 else f"in runs of {burst} ({burst * 20} ms holes)"
    wide = len(panels) > 1
    who = "" if wide else MODEL_NAMES.get(panels[0]["model"], panels[0]["model"]) + ". "
    fig, axes = _new_chart("Phoneme error rate as frames go missing", f"{who}Frames lost {how}. Lower is better.",
                           (12.6, 4.6) if wide else (8.6, 4.6), len(panels))
    span = max(100 * p for r in panels for per in r["per"].values() for p in per) or 1
    for ax, r in zip(axes, panels):
        x = range(len(r["loss"]))
        ends = []
        for fill, per in r["per"].items():
            y = [100 * p for p in per]
            ax.plot(x, y, color=colours[fill], linewidth=2, marker="o", markersize=5.5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=names[fill])
            ends.append((y[-1], fill))
        # Label each line at its right end, nudging labels apart if two lines finish close together.
        ends.sort()
        placed = []
        for y_end, fill in ends:
            y_text = y_end if not placed else max(y_end, placed[-1] + 0.055 * span)
            placed.append(y_text)
            ax.annotate(f"{names[fill]}  {y_end:.1f}%", (len(x) - 1, y_end), xytext=(len(x) - 0.82, y_text),
                        va="center", fontsize=8.5, color=INK_SOFT, annotation_clip=False)
        if wide:
            ax.set_title(MODEL_NAMES.get(r["model"], r["model"]), loc="left", fontsize=10, fontweight="bold",
                         color=INK, pad=8)
        ax.set_xticks(list(x), [f"{100 * loss:g}%" for loss in r["loss"]])
        ax.set_xlim(-0.3, len(x) - 0.7)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("share of frames lost", color=INK_SOFT, fontsize=9)
    axes[0].set_ylim(0, span * 1.12)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:g}%")
    axes[0].set_ylabel("phoneme error rate", color=INK_SOFT, fontsize=9)
    look = dict(ncols=3, frameon=False, fontsize=8.5, labelcolor=INK_SOFT, handletextpad=0.4, columnspacing=1.2)
    if wide:   # one legend for both panels, top right, level with the subtitle
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.935), **look)
        fig.subplots_adjust(left=0.06, right=0.865, top=0.80, bottom=0.12, wspace=0.44)
    else:
        axes[0].legend(loc="lower right", bbox_to_anchor=(1, 1.0), **look)
        fig.subplots_adjust(left=0.09, right=0.80, top=0.82, bottom=0.12)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
