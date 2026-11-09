"""Latency and throughput of the inference service's ``POST /score``.

Each request goes through the whole FastAPI application in-process: JSON
parsing, validation, inference and serialisation. There is no HTTP server and
no network, so the numbers are a lower bound on what a client would see.

    python benchmarks/serving.py                      # run, then write the report
    python benchmarks/serving.py --report-only        # rebuild the report from the CSV

Writes, next to ``--output``:

- ``serving.csv``: one row per timed request, with the hardware, software and
  git commit in every row;
- ``serving.md``: p50/p95/p99 latency and throughput with bootstrap 95% CIs;
- ``serving.png``: the same as a figure.

The model is the MNIST architecture (784 -> 256 -> 128) with random weights:
latency does not depend on the weights, and no dataset or training is needed.
Request logging is silenced. Peak RSS is the process peak after each setting,
so it accumulates over settings, which run in increasing batch size.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import platform
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from radon.models import RadonAutoencoder, save_checkpoint
from radon.service.app import Settings, create_app

ARCHITECTURE = {"input_dim": 784, "hidden_dim": 256, "latent_dim": 128}
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "serving.csv"


# --------------------------------------------------------------------------- #
# Environment description
# --------------------------------------------------------------------------- #


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        windll = ctypes.windll  # type: ignore[attr-defined,unused-ignore]
        # Without explicit types ctypes passes the 64-bit process handle as a 32-bit int.
        windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        process = windll.kernel32.GetCurrentProcess()
        if not windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return float("nan")
        return float(counters.PeakWorkingSetSize) / 2**20

    import resource

    usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined,unused-ignore]
    peak = float(usage.ru_maxrss)
    # ru_maxrss is in bytes on macOS and in KiB on Linux.
    return peak / 2**20 if platform.system() == "Darwin" else peak / 2**10


def cpu_model() -> str:
    """Marketing name of the CPU, falling back to platform.processor()."""
    try:
        if platform.system() == "Windows":
            import winreg

            key = winreg.OpenKey(  # type: ignore[attr-defined,unused-ignore]
                winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined,unused-ignore]
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0]  # type: ignore[attr-defined,unused-ignore]
            return str(name).strip()
        if platform.system() == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            return out.stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def ram_gb() -> float:
    """Installed RAM in GiB, or NaN when it cannot be read."""
    try:
        if platform.system() == "Windows":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))  # type: ignore[attr-defined,unused-ignore]
            return float(status.ullTotalPhys) / 2**30
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            return int(out.stdout) / 2**30
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 2**20
    except (OSError, ValueError):
        pass
    return float("nan")


def git_commit() -> str:
    """Short commit hash, with ``+dirty`` when the working tree has changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{head}+dirty" if dirty else head


def environment(device: torch.device) -> dict[str, object]:
    """Hardware and software description stored with every row."""
    gpu, driver = "none", "none"
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        driver = f"CUDA {torch.version.cuda}"
    return {
        "cpu": cpu_model(),
        "ram_gb": round(ram_gb(), 1),
        "gpu": gpu,
        "gpu_driver": driver,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "git_commit": git_commit(),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def measure(
    checkpoint: Path,
    device: torch.device,
    batch_sizes: list[int],
    repetitions: int,
    warmup: int,
    seed: int,
) -> pd.DataFrame:
    """Time ``POST /score`` for each batch size on one device."""
    client = TestClient(
        create_app(Settings(checkpoint=checkpoint, device=str(device), max_batch_size=1024))
    )
    rng = np.random.default_rng(seed)
    env = environment(device)
    rows: list[dict[str, object]] = []

    for batch_size in sorted(batch_sizes):
        records = rng.random((batch_size, ARCHITECTURE["input_dim"])).round(6).tolist()
        body = json.dumps({"records": records}).encode()
        headers = {"content-type": "application/json"}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        for _ in range(warmup):
            client.post("/score", content=body, headers=headers).raise_for_status()

        latencies = []
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            response = client.post("/score", content=body, headers=headers)
            latencies.append((time.perf_counter_ns() - start) / 1e6)
            response.raise_for_status()

        rss = peak_rss_mb()
        gpu_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        for repetition, latency in enumerate(latencies):
            rows.append(
                {
                    "device": device.type,
                    "batch_size": batch_size,
                    "repetition": repetition,
                    "latency_ms": latency,
                    "peak_rss_mb": round(rss, 1),
                    "gpu_peak_mb": round(gpu_peak, 1),
                    **env,
                }
            )
        print(f"{device.type} batch {batch_size}: median {np.median(latencies):.2f} ms")
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def bootstrap_ci(
    values: np.ndarray, statistic: str, resamples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Point estimate and percentile-bootstrap 95% interval of a statistic."""

    def compute(sample: np.ndarray) -> np.ndarray:
        if statistic == "mean":
            return np.asarray(sample.mean(axis=-1))
        return np.asarray(np.percentile(sample, float(statistic.lstrip("p")), axis=-1))

    estimate = float(compute(values))
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    low, high = np.percentile(compute(values[indices]), [2.5, 97.5])
    return estimate, float(low), float(high)


def summarise(raw: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    """One row per (device, batch size) with point estimates and 95% CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    for (device, _), group in raw.groupby(["device", "batch_size"], sort=True):
        batch_size = int(group["batch_size"].iloc[0])
        latency = group["latency_ms"].to_numpy()
        throughput = batch_size * 1000.0 / latency  # records per second, per request
        row: dict[str, object] = {
            "device": device,
            "batch_size": batch_size,
            "n": latency.size,
        }
        for name in ("p50", "p95", "p99"):
            row[name], row[f"{name}_low"], row[f"{name}_high"] = bootstrap_ci(
                latency, name, resamples, rng
            )
        row["throughput"], row["throughput_low"], row["throughput_high"] = bootstrap_ci(
            throughput, "mean", resamples, rng
        )
        row["peak_rss_mb"] = float(group["peak_rss_mb"].max())
        row["gpu_peak_mb"] = float(group["gpu_peak_mb"].max())
        rows.append(row)
    frame: pd.DataFrame = pd.DataFrame(rows)
    return frame


def markdown(summary: pd.DataFrame, raw: pd.DataFrame) -> str:
    """The summary as a markdown table, preceded by the environment."""
    first = raw.iloc[0]
    lines = [
        f"CPU: {first['cpu']} ({first['torch_threads']} torch threads), RAM {first['ram_gb']} GB, "
        f"GPU: {first['gpu']}; {first['os']}; Python {first['python']}, torch {first['torch']}; "
        f"commit {first['git_commit']}, {first['timestamp_utc']}.",
        "",
        "| Device | Batch | Requests | p50 ms | p95 ms | p99 ms | Records/s | Peak RSS MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    def cell(row: pd.Series, name: str, digits: int) -> str:
        return (
            f"{row[name]:.{digits}f} [{row[name + '_low']:.{digits}f}, "
            f"{row[name + '_high']:.{digits}f}]"
        )

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['device']} | {row['batch_size']} | {row['n']} | {cell(row, 'p50', 2)} | "
            f"{cell(row, 'p95', 2)} | {cell(row, 'p99', 2)} | {cell(row, 'throughput', 0)} | "
            f"{row['peak_rss_mb']:.0f} |"
        )
    lines += ["", "Brackets: bootstrap 95% confidence intervals over the timed requests."]
    return "\n".join(lines) + "\n"


def figure(summary: pd.DataFrame, path: Path) -> None:
    """Median latency and throughput against batch size, with 95% CIs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    for device, group in summary.groupby("device"):
        x = group["batch_size"]
        for axis, name, label in (
            (left, "p50", "p50 latency (ms)"),
            (right, "throughput", "records/s"),
        ):
            axis.errorbar(
                x,
                group[name],
                yerr=[group[name] - group[f"{name}_low"], group[f"{name}_high"] - group[name]],
                marker="o",
                capsize=3,
                label=str(device),
            )
            axis.set_xscale("log", base=2)
            axis.set_xticks(list(x), [str(v) for v in x])
            axis.set_xlabel("batch size (records per request)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.3)
    left.set_title("POST /score latency (median)")
    right.set_title("Throughput")
    right.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_report(csv_path: Path, resamples: int, seed: int) -> None:
    """Rebuild the markdown table and the figure from a raw CSV."""
    raw = pd.read_csv(csv_path)
    summary = summarise(raw, resamples, seed)
    csv_path.with_suffix(".md").write_text(markdown(summary, raw), encoding="utf-8")
    figure(summary, csv_path.with_suffix(".png"))
    print(markdown(summary, raw))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--repetitions", type=int, default=200, help="timed requests per setting")
    parser.add_argument("--warmup", type=int, default=20, help="untimed requests per setting")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="raw CSV path")
    parser.add_argument("--report-only", action="store_true", help="skip the run, rebuild report")
    args = parser.parse_args(argv)

    if args.repetitions < 30:
        parser.error("--repetitions must be at least 30.")

    if not args.report_only:
        logging.getLogger("radon").setLevel(logging.WARNING)
        torch.manual_seed(args.seed)
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = save_checkpoint(RadonAutoencoder(**ARCHITECTURE), Path(tmp) / "bench.pth")
            frames = [
                measure(
                    checkpoint, device, args.batch_sizes, args.repetitions, args.warmup, args.seed
                )
                for device in devices
            ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames).to_csv(args.output, index=False)

    write_report(args.output, args.bootstrap, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
