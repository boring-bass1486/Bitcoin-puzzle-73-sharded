# Bitcoin Puzzle 73 Solver

High-performance C brute-forcer for Bitcoin Puzzle 73 using AVX-512 SIMD.

## Architecture
- 16-way parallel SHA-256 + RIPEMD-160 in ZMM registers (AVX-512F/BW/DQ/VL/VBMI/IFMA)
- **Single-pipeline per thread** (16 keys/iter): uses ~27 of 32 ZMM registers — no register spill
  - Dual-pipeline (32 keys/iter, two inlined SHA-256 bodies) needed 48 ZMM → 16 spills → killed throughput
- **Interleaved RIPEMD-160**: left chain (al..el) and right chain (ar..er) rounds interleaved pair-by-pair
  so the CPU's OoO engine issues both chains simultaneously, utilizing both AVX-512 execution ports.
  Register budget: 10 state + 16 W = 26 ZMM — fits in 32 with zero spill.
- Thread count: `floor(nproc × 2 / shard_count)`, min 2 — targets 2 threads per logical CPU across all shards
- Each thread pinned via `pthread_setaffinity_np`: shard s, thread t → CPU `(s × threads + t) % nproc`
- sha256_16x_keys() builds SHA-256 W[0–2] directly from key values via SIMD (no key_to_bytes scatter)
- SHA256_COMPRESS_KEYS macro: specialized body for rounds 0-31 (skips zero W[3..14] adds); rounds 32-63 use normal EROUNDz
- xorshift128+ RNG, one anchor per 16-key batch + sequential offsets

## Performance
- **~96 Mk/s fleet total** across 4 shards on 6 logical CPUs (Replit container)
- Balanced shard speeds: ~23–25 Mk/s each
- Progression: 69 → 83 (+20% single-pipeline+threads) → 96 Mk/s (+16% RIPEMD interleaving)
- Checkpointing disabled (always starts fresh); `PUZZLE_COMPACT=1` enables single-line shard output

## Assembly profile (thread_func)
- SHA-256 section: ~195 vpaddd (18% of work)
- RIPEMD-160 section: ~891 vpaddd (82% of work)
- ZMM spills: ~7 total (negligible)

## Files
- `main.c` — solver (single TU, ~1065 lines)
- `run_shards.sh` — launches 4 shards, inline C display helper aggregates per-shard `att:`/`el:` fields into fleet line
- `Makefile` — TAB-formatted, unsets `NIX_ENFORCE_NO_NATIVE`, uses
  `-march=native -mtune=native -mavx512{f,bw,dq,vl,vbmi,ifma} -mbmi2 -O3
  -funroll-loops -fno-plt -fomit-frame-pointer`

## CLI
- `./run_shards.sh 4` — run 4 shards with non-scrolling display
- `./main --shard I/N` — run shard I (1..N) manually
- `./main --help`

## Workflow
- `Start application` runs `make && ./run_shards.sh 4`

## Notes for future edits
- Makefile recipes MUST start with a literal TAB (not spaces).
- Recipes prefix the compiler with
  `unset NIX_ENFORCE_NO_NATIVE NIX_ENFORCE_NO_NATIVE_clang_wrapper && \`
  to defeat the Nix wrapper that otherwise strips `-march=native`.
- Never reintroduce dual-pipeline (two sha256_16x_keys inlined per thread) —
  it exceeds 32 ZMM registers and spills to memory, costing ~20% throughput.
- RIPEMD-160 interleaving is the key structural optimisation: left and right chains
  must remain interleaved round-by-round. Reverting to sequential (all 80 left then
  all 80 right) loses ~16% throughput.
- The GitHub OAuth integration was declined; suggest the Git sidebar or PAT route.
