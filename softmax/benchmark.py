"""
Benchmark all softmax implementations:
  - torch_softmax    : torch.nn.functional.softmax (baseline)
  - naive_softmax    : pure PyTorch manual ops
  - triton_flatten   : Triton, flattens ND→2D, single-pass
  - triton_grid      : Triton, explicit 3D grid (batch×rows launch)
  - online_softmax   : Triton, two-pass tiled (handles large rows)

Run: python softmax/benchmark.py
Outputs:
  benchmark_results.png           — latency + throughput vs cols
  benchmark_throughput_bar.png    — grouped bar chart
  benchmark_row_scaling.png       — throughput vs total rows (GPU occupancy)
  benchmark_fp16_vs_fp32.png      — fp32 vs fp16 side-by-side
  benchmark_fp16_speedup.png      — fp16/fp32 speedup ratio
"""

import os, sys, importlib.util
import torch
import triton
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── module loader ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR  = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def _load(alias, filename):
    path = os.path.join(SCRIPT_DIR, filename)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod

m_naive   = _load("m_naive",   "03_naive_softmax.py")
m_flatten = _load("m_flatten", "02_3d_softmax_flatten.py")
m_grid    = _load("m_grid",    "02_3D_softmax_grid.py")
m_online  = _load("m_online",  "01_online_softmax.py")

naive_fn   = m_naive.naive_softmax
flatten_fn = m_flatten.triton_softmax_3d
grid_fn    = m_grid.triton_softmax_3d
online_fn  = m_online.online_softmax

# ── correctness check ──────────────────────────────────────────────────────────
def check_correctness(tol=1e-5):
    x = torch.randn(4, 8, 512, device="cuda", dtype=torch.float32)
    ref = torch.nn.functional.softmax(x, dim=-1)
    bs  = triton.next_power_of_2(512)

    fns = {
        "naive_softmax":  lambda: naive_fn(x),
        "triton_flatten": lambda: flatten_fn(x, block_size=bs),
        "triton_grid":    lambda: grid_fn(x,    block_size=bs),
        "online_softmax": lambda: online_fn(x,  block_size=min(512, bs)),
    }
    print("Correctness check (vs torch.softmax, tol=1e-5):")
    for name, fn in fns.items():
        out     = fn()
        max_err = (out - ref).abs().max().item()
        status  = "PASS" if max_err < tol else "FAIL"
        print(f"  {name:<20} max_err={max_err:.2e}  [{status}]")
    print()

# ── benchmark helpers ──────────────────────────────────────────────────────────
BATCH = 8
ROWS  = 64
DTYPE = torch.float32
COLS  = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

def throughput_GBs(ms: float, x: torch.Tensor) -> float:
    nbytes = x.numel() * x.element_size()
    # 2× = 1 read + 1 write (useful-work metric, same for all kernels).
    # online_softmax actually touches 3× (reads input twice), so its bar
    # shows ~67% of single-pass at bandwidth-bound sizes — that gap is the cost.
    return 2 * nbytes / (ms * 1e-3) / 1e9

def run_benchmarks():
    records = {k: [] for k in ("torch", "naive", "flatten", "grid", "online")}

    hdr = f"{'cols':>6} | {'torch':>8} | {'naive':>8} | {'flatten':>8} | {'grid':>8} | {'online':>8}   (ms)"
    print(f"Latency  |  B={BATCH}, R={ROWS}, dtype=fp32")
    print(hdr)
    print("-" * len(hdr))

    for cols in COLS:
        x    = torch.randn(BATCH, ROWS, cols, device="cuda", dtype=DTYPE)
        bs   = triton.next_power_of_2(cols)       # full-row block size
        obs  = min(1024, bs)                      # online-softmax tile size

        t = {}
        t["torch"]   = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=-1))
        t["naive"]   = triton.testing.do_bench(lambda: naive_fn(x))
        t["flatten"] = triton.testing.do_bench(lambda: flatten_fn(x, block_size=bs))
        t["grid"]    = triton.testing.do_bench(lambda: grid_fn(x,    block_size=bs))
        t["online"]  = triton.testing.do_bench(lambda: online_fn(x,  block_size=obs))

        for k in records:
            records[k].append(t[k])

        row = (f"{cols:>6} | {t['torch']:>8.3f} | {t['naive']:>8.3f} | "
               f"{t['flatten']:>8.3f} | {t['grid']:>8.3f} | {t['online']:>8.3f}")
        print(row)

    print()
    return records

# ── plots ──────────────────────────────────────────────────────────────────────
COLORS = {
    "torch":   "#4878cf",
    "naive":   "#6acc65",
    "flatten": "#d65f5f",
    "grid":    "#b47cc7",
    "online":  "#c4ad66",
}
MARKERS = {"torch": "o", "naive": "s", "flatten": "^", "grid": "D", "online": "P"}

def save_plots(records):
    # throughput
    throughputs = {}
    for name, times in records.items():
        throughputs[name] = []
        for ms, cols in zip(times, COLS):
            x   = torch.empty(BATCH, ROWS, cols, dtype=DTYPE)
            throughputs[name].append(throughput_GBs(ms, x))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: latency ─────────────────────────────────────────────────────────
    ax = axes[0]
    for name, times in records.items():
        ax.plot(COLS, times, marker=MARKERS[name], color=COLORS[name],
                label=name, linewidth=2, markersize=7)
    ax.set_xlabel("Sequence length (cols)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(f"Softmax Latency  (B={BATCH}, R={ROWS}, fp32)", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(COLS)
    ax.set_xticklabels([str(c) for c in COLS])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── right: throughput ─────────────────────────────────────────────────────
    ax = axes[1]
    for name, tputs in throughputs.items():
        ax.plot(COLS, tputs, marker=MARKERS[name], color=COLORS[name],
                label=name, linewidth=2, markersize=7)
    ax.set_xlabel("Sequence length (cols)", fontsize=12)
    ax.set_ylabel("Throughput (GB/s)", fontsize=12)
    ax.set_title(f"Softmax Throughput  (B={BATCH}, R={ROWS}, fp32)", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(COLS)
    ax.set_xticklabels([str(c) for c in COLS])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "benchmark_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")

    # ── throughput-only zoomed plot ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x_idx = range(len(COLS))
    width = 0.15
    names = list(records.keys())
    for i, name in enumerate(names):
        offsets = [xi + i * width for xi in x_idx]
        ax.bar(offsets, throughputs[name], width=width,
               color=COLORS[name], label=name, alpha=0.85)
    ax.set_xticks([xi + width * (len(names) - 1) / 2 for xi in x_idx])
    ax.set_xticklabels([str(c) for c in COLS])
    ax.set_xlabel("Sequence length (cols)", fontsize=12)
    ax.set_ylabel("Throughput (GB/s)", fontsize=12)
    ax.set_title(f"Softmax Throughput Comparison  (B={BATCH}, R={ROWS}, fp32)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out2 = os.path.join(PLOTS_DIR, "benchmark_throughput_bar.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Bar chart saved → {out2}")


# ── row-scaling benchmark ──────────────────────────────────────────────────────
# Fixed cols, sweep total rows — reveals GPU occupancy / SM saturation behavior.
TOTAL_ROWS_SWEEP = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
ROW_BENCH_COLS   = 1024

def run_row_scaling():
    records = {k: [] for k in ("torch", "naive", "flatten", "grid", "online")}
    bs  = triton.next_power_of_2(ROW_BENCH_COLS)
    obs = min(1024, bs)

    hdr = (f"{'rows':>6} | {'torch':>8} | {'naive':>8} | "
           f"{'flatten':>8} | {'grid':>8} | {'online':>8}   (ms)")
    print(f"Row-scaling  |  cols={ROW_BENCH_COLS}, dtype=fp32")
    print(hdr)
    print("-" * len(hdr))

    for total_rows in TOTAL_ROWS_SWEEP:
        x = torch.randn(total_rows, 1, ROW_BENCH_COLS, device="cuda", dtype=torch.float32)

        t = {}
        t["torch"]   = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=-1))
        t["naive"]   = triton.testing.do_bench(lambda: naive_fn(x))
        t["flatten"] = triton.testing.do_bench(lambda: flatten_fn(x, block_size=bs))
        t["grid"]    = triton.testing.do_bench(lambda: grid_fn(x,    block_size=bs))
        t["online"]  = triton.testing.do_bench(lambda: online_fn(x,  block_size=obs))

        for k in records:
            records[k].append(t[k])

        print(f"{total_rows:>6} | {t['torch']:>8.3f} | {t['naive']:>8.3f} | "
              f"{t['flatten']:>8.3f} | {t['grid']:>8.3f} | {t['online']:>8.3f}")

    print()
    return records


def save_row_scaling_plot(records):
    throughputs = {}
    for name, times in records.items():
        throughputs[name] = [
            throughput_GBs(ms, torch.empty(r, 1, ROW_BENCH_COLS, dtype=torch.float32))
            for ms, r in zip(times, TOTAL_ROWS_SWEEP)
        ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, times in records.items():
        ax.plot(TOTAL_ROWS_SWEEP, times, marker=MARKERS[name], color=COLORS[name],
                label=name, linewidth=2, markersize=7)
    ax.set_xlabel("Total rows  (batch × seq_rows)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(f"Latency vs Parallelism  (cols={ROW_BENCH_COLS}, fp32)", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TOTAL_ROWS_SWEEP)
    ax.set_xticklabels([str(r) for r in TOTAL_ROWS_SWEEP])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name, tputs in throughputs.items():
        ax.plot(TOTAL_ROWS_SWEEP, tputs, marker=MARKERS[name], color=COLORS[name],
                label=name, linewidth=2, markersize=7)
    ax.set_xlabel("Total rows  (batch × seq_rows)", fontsize=12)
    ax.set_ylabel("Throughput (GB/s)", fontsize=12)
    ax.set_title(f"Throughput vs Parallelism  (cols={ROW_BENCH_COLS}, fp32)", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(TOTAL_ROWS_SWEEP)
    ax.set_xticklabels([str(r) for r in TOTAL_ROWS_SWEEP])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "benchmark_row_scaling.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")


# ── fp16 vs fp32 benchmark ─────────────────────────────────────────────────────
# Same shape as main benchmark, compare fp32 vs fp16 throughput side-by-side.
# fp16 = half DRAM traffic → expect ~2× if BW-bound; less if compute or launch bound.
FP_BENCH_COLS = [512, 1024, 2048, 4096, 8192]

def run_dtype_benchmark():
    results = {}   # {dtype_str: {kernel_name: [throughput_GBs, ...]}}

    for dtype, dtype_str in [(torch.float32, "fp32"), (torch.float16, "fp16")]:
        tputs = {k: [] for k in ("torch", "naive", "flatten", "grid", "online")}
        print(f"dtype={dtype_str}  |  B={BATCH}, R={ROWS}")

        for cols in FP_BENCH_COLS:
            x   = torch.randn(BATCH, ROWS, cols, device="cuda", dtype=dtype)
            bs  = triton.next_power_of_2(cols)
            obs = min(1024, bs)

            t = {}
            t["torch"]   = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=-1))
            t["naive"]   = triton.testing.do_bench(lambda: naive_fn(x))
            t["flatten"] = triton.testing.do_bench(lambda: flatten_fn(x, block_size=bs))
            t["grid"]    = triton.testing.do_bench(lambda: grid_fn(x,    block_size=bs))
            t["online"]  = triton.testing.do_bench(lambda: online_fn(x,  block_size=obs))

            for k in tputs:
                x_cpu = torch.empty(BATCH, ROWS, cols, dtype=dtype)
                tputs[k].append(throughput_GBs(t[k], x_cpu))

        results[dtype_str] = tputs

    print()
    return results


def save_dtype_plot(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, dtype_str in zip(axes, ["fp32", "fp16"]):
        for name in results[dtype_str]:
            ax.plot(FP_BENCH_COLS, results[dtype_str][name],
                    marker=MARKERS[name], color=COLORS[name],
                    label=name, linewidth=2, markersize=7)
        ax.set_xlabel("Sequence length (cols)", fontsize=12)
        ax.set_ylabel("Throughput (GB/s)", fontsize=12)
        ax.set_title(f"Softmax Throughput — {dtype_str}  (B={BATCH}, R={ROWS})", fontsize=13)
        ax.set_xscale("log", base=2)
        ax.set_xticks(FP_BENCH_COLS)
        ax.set_xticklabels([str(c) for c in FP_BENCH_COLS])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "benchmark_fp16_vs_fp32.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out}")

    # speedup ratio plot — fp16 tput / fp32 tput per kernel
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in results["fp32"]:
        speedups = [h / f for h, f in zip(results["fp16"][name], results["fp32"][name])]
        ax.plot(FP_BENCH_COLS, speedups, marker=MARKERS[name], color=COLORS[name],
                label=name, linewidth=2, markersize=7)
    ax.axhline(y=2.0, color="gray", linestyle="--", linewidth=1.5, alpha=0.6, label="ideal 2×")
    ax.set_xlabel("Sequence length (cols)", fontsize=12)
    ax.set_ylabel("fp16 / fp32 throughput ratio", fontsize=12)
    ax.set_title(f"fp16 Speedup over fp32  (B={BATCH}, R={ROWS})", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(FP_BENCH_COLS)
    ax.set_xticklabels([str(c) for c in FP_BENCH_COLS])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = os.path.join(PLOTS_DIR, "benchmark_fp16_speedup.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out2}")


# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    check_correctness()

    print("=" * 60)
    records = run_benchmarks()
    save_plots(records)

    print("=" * 60)
    row_records = run_row_scaling()
    save_row_scaling_plot(row_records)

    print("=" * 60)
    dtype_results = run_dtype_benchmark()
    save_dtype_plot(dtype_results)
