import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_benchmark_data(csv_path):
    """
    Load benchmark results from a CSV file.

    Args:
        csv_path (str or Path): Path to the benchmark CSV.

    Returns:
        pandas.DataFrame: Benchmark results.
    """
    return pd.read_csv(csv_path)


def plot_throughput(df, save_path):
    """
    Plot images processed per second against batch size.

    A separate line is drawn for each number of DataLoader workers.

    Args:
        df (pandas.DataFrame): Benchmark results.
        save_path (str or Path): Output image path.
    """
    plt.figure()

    for workers, group in df.groupby("num_workers"):
        group = group.sort_values("batch_size")

        plt.plot(
            group["batch_size"],
            group["images_per_second"],
            marker="o",
            label=f"{workers} workers"
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Images/sec")
    plt.title("Training Throughput")
    plt.legend()
    #plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_gpu_utilization(df, save_path):
    """
    Plot average GPU utilization against batch size.

    Args:
        df (pandas.DataFrame): Benchmark results.
        save_path (str or Path): Output image path.
    """
    plt.figure()

    for workers, group in df.groupby("num_workers"):
        group = group.sort_values("batch_size")

        plt.plot(
            group["batch_size"],
            group["gpu_avg"],
            marker="o",
            label=f"{workers} workers"
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Average GPU Utilization (%)")
    plt.title("GPU Utilization")
    plt.legend()
    #plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_data_loading(df, save_path):
    """
    Plot data-loading time against batch size.

    Args:
        df (pandas.DataFrame): Benchmark results.
        save_path (str or Path): Output image path.
    """
    plt.figure()

    for workers, group in df.groupby("num_workers"):
        group = group.sort_values("batch_size")

        plt.plot(
            group["batch_size"],
            group["data_time"] * 1000,
            marker="o",
            label=f"{workers} workers"
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Data Loading Time (ms)")
    plt.title("Data Loading Time")
    plt.legend()
    #plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_gpu_memory(df, save_path):
    """
    Plot peak PyTorch GPU memory allocation against batch size.

    Args:
        df (pandas.DataFrame): Benchmark results.
        save_path (str or Path): Output image path.
    """
    plt.figure()

    for workers, group in df.groupby("num_workers"):
        group = group.sort_values("batch_size")

        plt.plot(
            group["batch_size"],
            group["peak_gpu_memory_mb"],
            marker="o",
            label=f"{workers} workers"
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Peak PyTorch Memory (MB)")
    plt.title("GPU Memory Usage")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """
    Generate plots from collected benchmark results.
    """
    csv_path = "logs/benchmarks/benchmark_results.csv"
    output_dir = Path("logs/benchmarks/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_benchmark_data(csv_path)

    plot_throughput(
        df, output_dir / "throughput.png"
    )

    plot_gpu_utilization(
        df, output_dir / "gpu_utilization.png"
    )

    plot_data_loading(
        df, output_dir / "data_loading.png"
    )

    plot_gpu_memory(
        df, output_dir / "gpu_memory.png"
    )

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()