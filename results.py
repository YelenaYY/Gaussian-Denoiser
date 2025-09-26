# Authors:Rongfei Jin and Yelena Yu,
# Date: 2025-09-24, 
# Course: CS 7180 Advanced Perception
# File Description:
# This file contains the functions to plot the results and generate the latex table.

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# This function is used to get the latest log with the maximum epoch.
def get_latest_log_with_max_epoch(log_dir: Path, model_type: str):
    log_dir = log_dir / model_type
    logs = list(log_dir.glob("log_*.txt"))
    dfs = []
    for log in logs:
        df = pd.read_csv(log, sep="\t")
        # get the max epoch
        max_epoch = df["epoch"].max()
        if pd.isna(max_epoch):
            continue
        dfs.append((df, model_type, max_epoch, log.stat().st_mtime))

    if len(dfs) == 0:
        print(f"No logs found for {model_type}")
        return None

    # sort by time and by epoch
    return sorted(dfs, key=lambda x: (x[2], x[3]), reverse=True)[0]


# This function is used to plot the average loss vs epoch.
def plot_average_loss(logs, result_dir: Path):
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for i, log in enumerate(logs):
        if log is None:
            print(f"Msising one plot average loss {i}")
            continue
        df, model_type, max_epoch, _ = log
        ax.plot(df["epoch"], df["avg_loss"], label=f"Model {model_type}, Max Epoch: {max_epoch}")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Loss")
        ax.set_title("Average Loss vs Epoch")
    
    ax.set_yscale("log")
    plt.savefig(result_dir / "average_loss_vs_epoch.png")

# This function is used to plot the psnr out vs epoch.
def plot_psnr_out_vs_epoch(logs, result_dir: Path):
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.figsize'] = (12, 8)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for i, log in enumerate(logs):
        if log is None:
            print(f"Msising one plot psnr out vs epoch {i}")
            continue
        df, model_type, max_epoch, _ = log
        ax.plot(df["epoch"], df["psnr_out"], label=f"Model {model_type}, Max Epoch: {max_epoch}")
        ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("PSNR Out (dB)")
        ax.set_title("PSNR Out vs Epoch")
    plt.savefig(result_dir / "psnr_out_vs_epoch.png")

# This function is used to generate the latex table.
def generate_latex_table(model_type: str, result_dir: Path):
    model_results_dir = result_dir / model_type

    summary = []
    for tests_dir in model_results_dir.glob("*"):
        if not tests_dir.is_dir():
            continue
        for average_stats_file in tests_dir.glob("average_stats.csv"):
            average_stats = pd.read_csv(average_stats_file)
            average_stats["test_set"] = tests_dir.stem
            summary.append(average_stats)

    summary = pd.concat(summary)

    summary = summary.groupby(["test_set", "noise_type"]).mean().reset_index()

    summary = summary[["test_set", "noise_type", "noisy_psnr", "noisy_ssim", "denoised_psnr", "denoised_ssim"]]

    summary.rename(columns={"test_set": "Test Set", "noise_type": "Noise Type", "noisy_psnr": "Noisy PSNR", "noisy_ssim": "Noisy SSIM", "denoised_psnr": "Denoised PSNR", "denoised_ssim": "Denoised SSIM"}, inplace=True)
    
    latex_table = summary.to_latex(index=False, float_format="%.2f")
    with open(result_dir / f"{model_type}_summary.tex", "w") as f:
        f.write(latex_table)

def main():
    log_dir = Path("logs")
    result_dir = Path("results")

    s_log = get_latest_log_with_max_epoch(log_dir , "s")
    b_log = get_latest_log_with_max_epoch(log_dir , "b")
    cb_log = get_latest_log_with_max_epoch(log_dir , "cb")
    three_log = get_latest_log_with_max_epoch(log_dir , "3")

    plot_average_loss([s_log, b_log, cb_log, three_log], result_dir)
    plot_psnr_out_vs_epoch([s_log, b_log, cb_log, three_log], result_dir)

    generate_latex_table("s", result_dir)
    generate_latex_table("b", result_dir)
    generate_latex_table("cb", result_dir)
    generate_latex_table("3", result_dir)

if __name__ == "__main__":
    main()