import pandas as pd


def load_benchmarks(path="logs/benchmarks/benchmark_results.csv"):
    """
    Load benchmark results into a pandas DataFrame.

    Args:
        path (str): Path to benchmark CSV.

    Returns:
        pandas.DataFrame: Benchmark results.
    """
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_benchmarks()

    print(df.to_string(index=False))