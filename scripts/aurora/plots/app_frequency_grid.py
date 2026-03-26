import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


FREQUENCIES = [1600, 1550, 1500]

bench_col = "Benchmark"
run_col = " Run. ID"
freq_trace_col = "Frequency trace"


def parse_trace(trace_str):
    """Parse strings like '[(0;1600)(10;1350)...]' into [(timestamp, value), ...]."""
    if pd.isna(trace_str):
        return []

    matches = re.findall(r"\(([^;]+);([^)]+)\)", str(trace_str).strip())
    return [(float(ts), float(value)) for ts, value in matches]


def load_frequency_trace(csv_path):
    """Return all frequency samples from one CSV, preserving each row as a separate series."""
    df = pd.read_csv(csv_path)
    records = []

    for row_idx, row in df.iterrows():
        run_id = row.get(run_col, row_idx)
        bench_name = row.get(bench_col, "")
        series_id = f"{csv_path.stem}:{run_id}:{row_idx}"
        for timestamp_ms, freq_mhz in parse_trace(row[freq_trace_col]):
            records.append(
                {
                    "series_id": series_id,
                    "benchmark": bench_name,
                    "run_id": run_id,
                    "time_ms": timestamp_ms,
                    "freq_mhz": freq_mhz,
                }
            )

    return pd.DataFrame(records)


def find_app_dirs(app_folder):
    """Return app directories that contain at least one CSV for the requested frequencies."""
    app_dirs = []
    for app_dir in sorted(Path(app_folder).iterdir()):
        if not app_dir.is_dir():
            continue
        if any(any(app_dir.glob(f"*_{freq}.csv")) for freq in FREQUENCIES):
            app_dirs.append(app_dir)
    return app_dirs


def parse_app_filter(app_names):
    if not app_names:
        return None

    selected_apps = [app.strip() for app in app_names.split(",") if app.strip()]
    return selected_apps or None


def plot_grid(app_folder, out_dir, selected_apps=None):
    app_dirs = find_app_dirs(app_folder)
    if selected_apps:
        app_dir_by_name = {app_dir.name: app_dir for app_dir in app_dirs}
        missing_apps = [app_name for app_name in selected_apps if app_name not in app_dir_by_name]
        if missing_apps:
            raise SystemExit(
                "The following apps were not found under "
                f"{app_folder}: {', '.join(missing_apps)}"
            )
        app_dirs = [app_dir_by_name[app_name] for app_name in selected_apps]

    if not app_dirs:
        raise SystemExit(f"No app directories with CSVs found under: {app_folder}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    nrows = len(FREQUENCIES)
    ncols = len(app_dirs)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 2.7 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    line_color = sns.color_palette("deep")[0]

    for col_idx, app_dir in enumerate(app_dirs):
        app_label = app_dir.name.replace("_", " ").replace("-", " ").title()

        for row_idx, freq in enumerate(FREQUENCIES):
            ax = axes[row_idx][col_idx]
            csv_candidates = sorted(app_dir.glob(f"*_{freq}.csv"))

            if not csv_candidates:
                ax.text(
                    0.5,
                    0.5,
                    f"Missing {freq} CSV",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                continue

            trace_df = load_frequency_trace(csv_candidates[0])
            if trace_df.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No trace data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            else:
                sns.lineplot(
                    data=trace_df,
                    x="time_ms",
                    y="freq_mhz",
                    estimator="mean",
                    errorbar="sd",
                    color=line_color,
                    linewidth=1.5,
                    ax=ax,
                )

                aggregated_df = (
                    trace_df.groupby("time_ms", as_index=False)["freq_mhz"]
                    .mean()
                    .sort_values("time_ms")
                )
                aggregated_df["freq_delta"] = aggregated_df["freq_mhz"].diff()
                first_drop = aggregated_df[aggregated_df["freq_delta"] < 0].head(1)

                if not first_drop.empty:
                    drop_point = first_drop.iloc[0]
                    drop_time = drop_point["time_ms"]
                    drop_freq = drop_point["freq_mhz"]

                    ax.scatter(
                        drop_time,
                        drop_freq,
                        color="red",
                        marker="x",
                        s=45,
                        linewidths=1.5,
                        zorder=5,
                    )
                    ax.annotate(
                        f"{drop_time:.0f} ms",
                        xy=(drop_time, drop_freq),
                        xytext=(6, 6),
                        textcoords="offset points",
                        color="red",
                        fontsize=8,
                    )

            ax.set_title(app_label if row_idx == 0 else "")
            ax.set_xlabel("Time (ms)" if row_idx == nrows - 1 else "")
            ax.set_ylabel(f"{freq} MHz\nFrequency (MHz)" if col_idx == 0 else "")
            ax.grid(alpha=0.25)

    fig.suptitle("Frequency traces by app and target frequency", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_dir / "app_frequency_grid.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot frequency traces for apps in a 3xN grid."
    )
    parser.add_argument(
        "--app-path",
        "--app-folder",
        dest="app_path",
        required=True,
        help="Path to the folder that contains one subfolder per app.",
    )
    parser.add_argument(
        "--out-path",
        "--out-dir",
        dest="out_path",
        required=True,
        help="Directory where the generated plot will be written.",
    )
    parser.add_argument(
        "--apps",
        help="Optional comma-separated list of app directory names to plot, for example: black-scholes,knn,mol_dyn",
    )

    args = parser.parse_args()
    plot_grid(args.app_path, args.out_path, parse_app_filter(args.apps))


if __name__ == "__main__":
    main()
