# Generate a plot where the y axis represents the power in W while the x-axis is the time interval (step 50 ms)
# The script take as input prameter a csv file with the power power_trace in the format: "[(timestamp;power_uw) ...]" for each tile
# create an array of timestamp and one of power for each tile and then use seabor to generate the plot


import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re
import os

power_trace_col="Power trace"
freq_trace_col="Frequency trace"
bench_col="Benchmark"

def parse_power_trace(trace_str):
    """
    Parse a string like:
    "[(0;74000000)(50;73000000)(100;69000000)...]"
    into a list of (timestamp, power_w) tuples.
    """
    # Remove the outer brackets if present
    trace_str = trace_str.strip("[]")

    # Add commas between entries by replacing ')(' with '),('
    trace_str = trace_str.replace(")(", "),(")

    # Now replace ; with , to make it a valid tuple
    trace_str = trace_str.replace(";", ",")

    # Wrap each entry in parentheses (needed for eval)
    trace_str = "[" + trace_str + "]"

    # Safely evaluate
    tuples = pd.eval(trace_str)  # or ast.literal_eval(trace_str)
    
    # Convert µW to W
    return [(ts, pwr * 1e-6) for ts, pwr in tuples]

def parse_freq_trace(trace_str):
    """
    Parse a string like:
    "[(0;74000000)(50;73000000)(100;69000000)...]"
    into a list of (timestamp, power_w) tuples.
    """
    # Remove the outer brackets if present
    trace_str = trace_str.strip("[]")

    # Add commas between entries by replacing ')(' with '),('
    trace_str = trace_str.replace(")(", "),(")

    # Now replace ; with , to make it a valid tuple
    trace_str = trace_str.replace(";", ",")

    # Wrap each entry in parentheses (needed for eval)
    trace_str = "[" + trace_str + "]"

    # Safely evaluate
    tuples = pd.eval(trace_str)  # or ast.literal_eval(trace_str)
    
    # Convert µW to W
    return [(ts, freq) for ts, freq in tuples]

def main(csv_file, out_path):
    df = pd.read_csv(csv_file)
    csv_name = os.path.basename(csv_file)
    all_data = []
    tile_freq_0 = csv_name.split("_")[2].replace("f", "")     
    tile_freq_1 = csv_name.split("_")[3].replace("f", "").replace(".csv", "")
    kernel_0 = csv_name.split("_")[3].replace("f", "").replace(".csv", "")
    kernel_1 = csv_name.split("_")[3].replace("f", "").replace(".csv", "")
    
       

    for _, row in df.iterrows():
        bench_name = row[bench_col]
        split_bench_name = bench_name.split("_")
        

        
        tile_id = int(split_bench_name[2].replace("tile", ""))
        tile_freq = 0
        if tile_id == 0:
            tile_freq = csv_name.split("_")[2].replace("f", "")     
        else:
            tile_freq = csv_name.split("_")[3].replace("f", "").replace(".csv", "")
            
        power_trace = parse_power_trace(row[power_trace_col])
        freq_trace = parse_freq_trace(row[freq_trace_col])
        
        for (timestamp_ms, power_w), (_, freq_mhz) in zip(power_trace, freq_trace):
            all_data.append({
                "tile": f"Tile {tile_id} {tile_freq} MHz",
                "time_ms": timestamp_ms,
                "power_w": power_w,
                "freq_mhz": freq_mhz
            })

    plot_df = pd.DataFrame(all_data)
    # Normalize time to 50 ms steps
    plot_df["time_step"] = plot_df["time_ms"]
    plot_df.to_csv(f'{out_path}/power_trace_f{tile_freq_0}_{tile_freq_1}.csv')
    print(plot_df)

    sns.set_theme()
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    # --- Top plot: Power ---
    sns.lineplot(
        data=plot_df,
        x="time_step",
        y="power_w",
        hue="tile",
        errorbar=("ci", 100),
        ax=axes[0]
    )

    axes[0].set_title("Power over Time")
    axes[0].set_ylabel("Power (W)")

    # --- Bottom plot: Frequency ---
    sns.lineplot(
        data=plot_df,
        x="time_step",
        y="freq_mhz",
        hue="tile",
        errorbar=("ci", 100),
        ax=axes[1]
    )

    axes[1].set_title("Frequency over Time")
    axes[1].set_ylabel("Frequency (MHz)")
    axes[1].set_xlabel("Time Step")

    # plt.xlabel("Time interval (50 ms steps)")
    # plt.ylabel("Power (W)")
    plt.title("Tile-based freq. scaling")
    plt.tight_layout()
    plt.savefig(f'{out_path}/power_trace_f{tile_freq_0}_{tile_freq_1}_{kernel}.pdf')
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", help="CSV file containing power traces")
    parser.add_argument("--out", help="Output path to the figures")
    
    args = parser.parse_args()

    main(args.csv_file, args.out)
