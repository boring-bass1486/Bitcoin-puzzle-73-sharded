# Bitcoin Puzzle 73 — AVX-512 Sharded Solver

High-performance C brute-forcer for [Bitcoin Puzzle 73](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx) using AVX-512 SIMD, processing **16 keys in parallel per CPU instruction**.

## Performance

| Optimization | Fleet speed | Gain |
|---|---|---|
| Baseline (sequential) | 69 Mk/s | — |
| Single-pipeline + thread tuning | 83 Mk/s | +20% |
| RIPEMD-160 L/R interleaving | **96 Mk/s** | +16% |

Measured on 6 logical CPUs (4 shards × 3 threads each).

## How it works

### Key insight: 16-wide SIMD batching

Each AVX-512 ZMM register holds 16 × 32-bit lanes. By packing 16 different private keys into a single ZMM vector, every SIMD instruction processes 16 keys simultaneously — SHA-256 compression, RIPEMD-160 rounds, and comparisons all run 16-wide.

### SHA-256 optimisation (sha256_16x_keys)

The input is a 73-bit private key: only `W[0]`, `W[1]`, and part of `W[2]` are non-zero. `SHA256_COMPRESS_KEYS` skips the zero message-word additions for rounds 3–31, reducing SHA-256 to its minimum instruction count. The compiler (`clang -O3`) folds any remaining `add(x, zero)` via InstCombine, so the 195 `vpaddd` instructions in the binary are already optimal.

### RIPEMD-160 dual-chain interleaving (ripemd160_16x)

RIPEMD-160 runs two independent hash chains in parallel (left: `al..el`, right: `ar..er`) that are combined only at the very end. The key insight: by interleaving left and right rounds **pair-by-pair** in the source, the CPU's out-of-order engine can issue both chains simultaneously, filling **both AVX-512 execution ports** every cycle.

```
Sequential (before):          Interleaved (after):
  left_round_0                  left_round_0 ; right_round_0
  left_round_1                  left_round_1 ; right_round_1
  ...                           ...
  left_round_79                 left_round_79 ; right_round_79
  right_round_0
  ...
  right_round_79
```

Register budget: 10 state (5L + 5R) + 16 W = **26 ZMM** — fits in 32 with zero register spill. This single structural change delivered +16% throughput with no change in instruction count.

### Thread model

```
4 shards  ×  3 threads/shard  =  12 threads on 6 logical CPUs
```

Thread count formula: `max(2, floor(nproc × 2 / shard_count))`. Each thread is pinned to a specific logical CPU via `pthread_setaffinity_np`. Oversubscription (2 threads per logical CPU) deliberately hides AVX-512 instruction latency: while one thread stalls on a dependency chain, the other issues independent work.

## Build

Requires a CPU with AVX-512F/BW/DQ/VL/VBMI/IFMA support (Intel Skylake-X / Ice Lake or newer).

```bash
make
```

Compiler flags used:
```
clang -O3 -funroll-loops -fno-plt -fomit-frame-pointer
      -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vbmi -mavx512ifma
      -mbmi2 -march=native -mtune=native
```

## Run

```bash
# 4 shards with non-scrolling display
./run_shards.sh 4

# Single shard (manual)
./main --shard 1/4
```

### Display output

```
[1/4] range:0x100-0x13f  cov:1.46e-11  key:0x10afb61ad205389f49e    22.6 Mk/s
[2/4] range:0x140-0x17f  cov:1.63e-11  key:0x16be5b4baa0fee9263c    25.3 Mk/s
[3/4] range:0x180-0x1bf  cov:1.46e-11  key:0x18e505ccac8a780bb8f    22.5 Mk/s
[4/4] range:0x1c0-0x1ff  cov:1.63e-11  key:0x1c6e70205cca59527f4    25.3 Mk/s
[fleet] 95.7 Mk/s  |  73.00G keys  |  12m elapsed  |  4/4 shards
```

If the key is found, it is written to `FOUND.txt` and printed to stdout.

## Assembly profile (thread_func, clang -O3)

| Section | `vpaddd` count | Share |
|---|---|---|
| SHA-256 (inlined) | 195 | 18% |
| RIPEMD-160 (inlined) | 891 | 82% |
| ZMM register spills | 7 | — |

## Target

Bitcoin Puzzle 73 — a known public key whose private key lies in the range `[2^72, 2^73)`.  
The solver searches this range randomly, reporting coverage as the fraction of the range tested.

## Files

| File | Description |
|---|---|
| `main.c` | Entire solver (~1 065 lines, single translation unit) |
| `Makefile` | Build rules |
| `run_shards.sh` | Launches N shards + aggregated display |
| `replit.md` | Architecture notes and gotchas |
