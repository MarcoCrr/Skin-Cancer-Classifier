import argparse
import time

import torch
import platform

from src.data import get_dataloaders
from src.model import get_model

from datetime import datetime
from pathlib import Path

# monitoring
import threading
import psutil
import pynvml


def synchronize(device):
    """
    Synchronize CUDA operations when running on a GPU.

    CUDA operations are asynchronous, so synchronization is required
    for accurate GPU timing. On CPU, this function does nothing.

    Args:
        device (str): Device used for benchmarking.
    """
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark_training(
    model,
    dataloader,
    device,
    warmup_batches=10,
    benchmark_batches=50,
):
    """
    Benchmark the main stages of a PyTorch training loop.

    The benchmark measures data-loading, forward, backward, optimizer,
    and total batch times, together with throughput and peak GPU memory.

    Args:
        model (torch.nn.Module): Model to benchmark.
        dataloader (DataLoader): Training dataloader.
        device (str): Device used for computation.
        warmup_batches (int): Number of batches excluded from measurements.
        benchmark_batches (int): Number of batches included in measurements.

    Returns:
        dict: Benchmark metrics.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

    model.train()

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    data_iterator = iter(dataloader)

    for _ in range(warmup_batches):
        try:
            images, labels = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            images, labels = next(data_iterator)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    synchronize(device)

    # Reset CUDA peak-memory statistics after warm-up.
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ---------------------------------------------------------
    # Measurement
    # ---------------------------------------------------------

    data_time = 0.0
    transfer_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    optimizer_time = 0.0
    total_time = 0.0

    images_processed = 0
    batches_processed = 0

    start_total = time.perf_counter()

    for _ in range(benchmark_batches):

        # -----------------------------
        # Data loading
        # -----------------------------

        start = time.perf_counter()

        try:
            images, labels = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            images, labels = next(data_iterator)

        # Not needed here since data loading is on CPU and doesn't involve CUDA operations.
        # synchronize(device)
        data_time += time.perf_counter() - start

        batch_size = images.size(0)
        images_processed += batch_size
        batches_processed += 1

        # -----------------------------
        # CPU -> GPU transfer
        # -----------------------------
        synchronize(device)
        start = time.perf_counter()

        images = images.to(device)
        labels = labels.to(device)

        synchronize(device)
        transfer_time += time.perf_counter() - start

        # -----------------------------
        # Forward pass
        # -----------------------------

        optimizer.zero_grad()

        synchronize(device)
        start = time.perf_counter()

        outputs = model(images)
        loss = criterion(outputs, labels)

        synchronize(device)
        forward_time += time.perf_counter() - start

        # -----------------------------
        # Backward pass
        # -----------------------------

        synchronize(device)
        start = time.perf_counter()

        loss.backward()

        synchronize(device)
        backward_time += time.perf_counter() - start

        # -----------------------------
        # Optimizer
        # -----------------------------

        synchronize(device)
        start = time.perf_counter()

        optimizer.step()

        synchronize(device)
        optimizer_time += time.perf_counter() - start

    synchronize(device)

    total_time = time.perf_counter() - start_total

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    images_per_second = images_processed / total_time
    batch_time = total_time / batches_processed

    results = {
        "images_per_second": images_per_second,
        "batch_time": batch_time,
        "data_time": data_time / batches_processed,
        "transfer_time": transfer_time / batches_processed,
        "forward_time": forward_time / batches_processed,
        "backward_time": backward_time / batches_processed,
        "optimizer_time": optimizer_time / batches_processed,
        "total_time": total_time,
        "batches": batches_processed,
        "images": images_processed,
    }

    if device == "cuda":
        results["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated() / (1024 ** 2) # memory allocated specifically by PyTorch
        )

    return results


def format_results(results, device, batch_size):
    """
    Print benchmark results in a human-readable format.

    Args:
        results (dict): Results returned by benchmark_training().
        device (str): Device used for benchmarking.
        batch_size (int): Size of each batch. Taken from the CLI argument.
    """
    results = f"""
    =======================================================
    PyTorch Training Benchmark
    =======================================================
    Date:               {datetime.now().isoformat(timespec="seconds")}
    Device:             {device}

    PyTorch:            {torch.__version__}
    CUDA available:     {torch.cuda.is_available()}
    CUDA version:       {torch.version.cuda}
    GPU:                {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}
    Python:             {platform.python_version()}

    Configuration
    -------------------------------------------------------
    Batch Size:         {batch_size}
    Measured Batches:   {results['batches']}
    Images processed:   {results['images']}

    Performance
    -------------------------------------------------------
    Images / second:    {results['images_per_second']:.2f}
    Batch time:         {results['batch_time'] * 1000:.2f} ms
    Total time:         {results['total_time']:.2f} s

    Timing breakdown
    -------------------------------------------------------
    Data loading:       {results['data_time'] * 1000:.2f} ms
    CPU->GPU time:      {results['transfer_time'] * 1000:.2f} ms
    Forward pass:       {results['forward_time'] * 1000:.2f} ms
    Backward pass:      {results['backward_time'] * 1000:.2f} ms
    Optimizer:          {results['optimizer_time'] * 1000:.2f} ms

    GPU memory
    -------------------------------------------------------
    PyTorch peak allocated:   {results['peak_gpu_memory_mb']:.2f} MB
    =======================================================
    """
    return results


class SystemMonitor:
    """Periodically collect CPU and NVIDIA GPU utilization statistics."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False

        self.cpu_usage = []
        self.gpu_usage = []
        self.gpu_memory = []

    def start(self):
        """Start collecting system statistics."""
        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def stop(self):
        """Stop collecting statistics and wait for the monitor thread."""
        self.running = False
        self.thread.join()

    def _monitor(self):
        """Collect CPU and GPU statistics periodically."""
        pynvml.nvmlInit()

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        while self.running:
            # CPU utilization
            cpu = psutil.cpu_percent(interval=None)
            self.cpu_usage.append(cpu)

            # GPU utilization
            gpu = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_usage.append(gpu.gpu)

            # GPU memory
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.gpu_memory.append(memory.used / 1024**2) # memory used by whole GPU

            time.sleep(self.interval)

        pynvml.nvmlShutdown()

    def summary(self): # create function that converts it into a string
        """Return average and peak system utilization statistics."""

        if not self.cpu_usage: # protects against ZeroDivisionError if no data was collected
            return {
                "cpu_avg": 0.0,
                "cpu_max": 0.0,
                "gpu_avg": 0.0,
                "gpu_max": 0.0,
                "gpu_memory_avg": 0.0,
                "gpu_memory_max": 0.0,
            }
    
        return {
        "cpu_avg": sum(self.cpu_usage) / len(self.cpu_usage),
        "cpu_max": max(self.cpu_usage),
        "gpu_avg": sum(self.gpu_usage) / len(self.gpu_usage),
        "gpu_max": max(self.gpu_usage),
        "gpu_memory_avg": sum(self.gpu_memory) / len(self.gpu_memory),
        "gpu_memory_max": max(self.gpu_memory),
    }

    def format_summary(self, stats):
        """
        Format system monitoring statistics for human-readable output.
        Args:
            stats (dict): Statistics returned by summary().
        Returns:
            str: Formatted statistics suitable for console output or logging.
        """
        return f"""
    System Monitoring
    -------------------------------------------------------
    CPU utilization
        Average:          {stats['cpu_avg']:.2f} %
        Maximum:          {stats['cpu_max']:.2f} %

    GPU utilization
        Average:          {stats['gpu_avg']:.2f} %
        Maximum:          {stats['gpu_max']:.2f} %

    GPU total memory used
        Average:          {stats['gpu_memory_avg']:.2f} MB
        Maximum:          {stats['gpu_memory_max']:.2f} MB
    -------------------------------------------------------
    """


def main():
    """
    Run the training benchmark from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch training performance"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used during benchmarking."
    )

    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=10,
        help="Number of warm-up batches."
    )

    parser.add_argument(
        "--benchmark-batches",
        type=int,
        default=50,
        help="Number of measured batches."
    )

    parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Number of DataLoader worker processes."
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, _ = get_dataloaders(
        "data/train",
        "data/val",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_model()
    model = model.to(device)

    # saving the benchmark results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = Path("logs/benchmarks")
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_dir / f"benchmark_{timestamp}.txt"

    warmup_batches = args.warmup_batches
    benchmark_batches = args.benchmark_batches

    # Start CPU and GPU monitoring 
    monitor = SystemMonitor()
    monitor.start()

    results = benchmark_training(
        model,
        train_loader,
        device,
        warmup_batches,
        benchmark_batches
    )

    monitor.stop()
    system_stats = monitor.summary()
    system_stats_text = monitor.format_summary(system_stats)

    print(format_results(results, device, args.batch_size))
    print(system_stats_text)

    with open(benchmark_path, "w") as f:
        f.write(format_results(results, device, args.batch_size))
        f.write(system_stats_text)

    print(f"\nBenchmark saved to: {benchmark_path}")


if __name__ == "__main__":
    main()