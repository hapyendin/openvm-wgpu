# GATE W1 — Adapter discovery + capability-driven selection (port-lab run, 2026-11-16)

Host: Vultr port-lab (45.77.145.125), Ubuntu, NVIDIA A16 (GRID vGPU, driver 620.70.06),
rustc 1.91.1, wgpu =30.0.2 (pinned, `--locked`), `WGPU_BACKEND=vulkan`.
Tree: /root/wgpu-w1 (public tree + W1 changes), commit `287743f` (git init inside port-lab).

**Verdict: PASS.** Goldens byte-exact on the A16 through the NEW selector; no WGSL/kernel
changes; `--list-adapters` on all three bins; fail-closed verified.

## 1. Adapter table (as `--list-adapters` prints it, honest wgpu output)

```
=== ADAPTER TABLE (Vulkan, via wgpu 30.0.2 pinned in Cargo.toml) ===
IDX NAME                               VENDOR     DEVICE     TYPE           I64     I64ATOM        BUF_MAX    STORAGE_B  WG_SZ
0   NVIDIA A16                         0x10de     0x25b6     DiscreteGpu    yes     all             4.00Gi       2.00Gi   1024
1   llvmpipe (LLVM 19.1.7, 256 bits)   0x10005    0x0000     Cpu            yes     all             2.00Gi        128Mi   1024
[0] driver="NVIDIA" driver_info="620.70.06" backend=Vulkan mem_hints=n/a (wgpu 30 Adapter exposes no memory_hints) wg_smem=49152 wg_per_dim=65535 bindings_per_bg=4294967295
[1] driver="llvmpipe" driver_info="Mesa 25.0.7-2+deb13u1 (LLVM 19.1.7)" backend=Vulkan mem_hints=n/a (wgpu 30 Adapter exposes no memory_hints) wg_smem=32768 wg_per_dim=65535 bindings_per_bg=1000000
```

Community matrix rows (this table is the module's output format):

| Adapter | Vendor:Device | Type | SHADER_INT64 | I64 atomics | max_buffer_size | max storage binding | WG size | Gate results |
|---|---|---|---|---|---|---|---|---|
| NVIDIA A16 (GRID, driver 620.70.06) | 0x10de:0x25b6 | DiscreteGpu | yes | all-ops | 4.00 GiB (4294967295) | 2.00 GiB (2147483644) | 1024 | **all 3 gates PASS byte-exact** (below) |
| llvmpipe (Mesa 25.0.7, LLVM 19.1.7) | 0x10005:0x0000 | Cpu | yes (advertised) | all (advertised) | 2.00 GiB | 128 MiB | 1024 | **GPU_MERKLE FAIL** — see §5 |
| Intel Arc Pro B70 (ANV) | 0x8086:… | DiscreteGpu | (expected yes) | (expected minmax/all) | ~2 GiB ANV-class cap | — | — | *row to be filled by parent on b70* |

Note: wgpu 30 exposes **no** `Adapter::memory_hints()` — the table prints that honestly
instead of inventing a value. The A16's max_buffer_size (4 GiB) is above the ~2 GiB
ANV-class cap that pipeline.rs dispatches around; the binding cap (2 GiB-4) still applies.

## 2. Selection algorithm (host layer only; no WGSL touched)

`src/adapters.rs` enumerates via `Instance::enumerate_adapters(Backends::VULKAN)`
(async in wgpu 30 → `pollster::block_on`). `WgpuDevice::init` no longer uses
`request_adapter` ("first suitable") — it selects:

1. `WGPU_DEVICE=<substring>` (case-insensitive substring of adapter name) pins the adapter.
2. Otherwise score adapters lexicographically: **(has SHADER_INT64, max_buffer_size, is_discrete)** —
   highest wins; ties keep enumeration order.
3. `require_intel_arc` (unchanged strict mode: `--arc` / `OPENVM_WGPU_ARC=1`) filters to vendor 0x8086 first.
4. `allow_u32_only=false` (ntt_test) filters to SHADER_INT64 adapters.
5. No adapter survives → **FAIL CLOSED**: the table is printed, then panic
   `FAIL CLOSED: <reason>` (verified: `WGPU_DEVICE=rtx9090` → table + fail-closed, exit 101).

`InitOpts` gained `device_substring: Option<String>` (all three bins wire
`WGPU_DEVICE` env); `--list-adapters` exits 0 after printing the table in all bins.

## 3. Gate results on A16 through the NEW selector

Kernel path engaged: **u64-REDC (SHADER_INT64 enabled + used)** — `FEATURES_ENABLED:
SHADER_INT64,SHADER_INT64_ATOMIC_ALL_OPS`, `Adapter: NVIDIA A16 (INT64=true)`.
The u32 fallback did NOT need to engage (A16 advertises int64).

Auto-pick (no env overrides beyond `WGPU_BACKEND=vulkan`): `SELECTED idx=0/2
strategy=capability (int64, buffer_max, discrete) score=(1, 4294967295, 1)` — chose A16
over llvmpipe correctly.

| Run | PASS lines (identical with `WGPU_DEVICE=A16` pin, `strategy=WGPU_DEVICE substring pin`) |
|---|---|
| gate_c1 | `GPU_PERM all_zero/counting/lcg: PASS` ×3 · `GPU_MERKLE root=1264a0fd266c3e7b… [PASS]` · `GPU_MERKLE sha256=e561ffc5…1917 [PASS]` · **`GATE_C1: PASS`** |
| ntt_test | `GPU_NTT sha256=a11b282970362f31…be0e PASS` · `GPU_NTT full correctness: PASS (0 bad words vs CPU)` · **`NTT_TEST: PASS`** |
| fri_commit_test | all 6 layer roots `[PASS]` · `GOLDEN_99FC8DF2_CPU: PASS` · `GOLDEN_99FC8DF2_GPU: PASS` · `GPU==CPU parity: 0/3008 words bad` |

Cross-vendor re-proof: merkle `1264a0fd…`, NTT `a11b2829…`, FRI `99fc8df2…` — byte-exact
on NVIDIA via the W1 selector (farm-c1 proved them via the old first-adapter path; zero
WGSL diff between them).

## 4. FINDING (pre-existing, NOT a W1 regression): atomic-probe WGSL fails naga@30

`gate_c1` crashed **on the PRISTINE public tree too** (`/root/wgpu-baseline`,
`gate_c1_pristine.out`, built `--locked` from the staged tree):

```
Shader 'atomic-probe' parsing error: numeric literal not representable by target type: `0xDEADBEEFCAFE1234`
  wgsl:15  atomicCompareExchangeWeak(&at.v, u64(0), u64(0xDEADBEEFCAFE1234) + u64(i));
```

naga@30 rejects u64 hex literals > i64::MAX. The A16 driver exposes
`SHADER_INT64_ATOMIC_ALL_OPS`, so the probe compiles → panics (wgpu 30's error sink is a
panic). ANV (Arc) likely never ran the probe (feature not advertised there), which is why
this shipped undetected. **WGSL left untouched per W1 rules.** Host-layer mitigation
applied in this commit: `probe_u64_atomics` now runs under `catch_unwind` and returns
`Err`; gate_c1 prints `U64_ATOMICS: PROBE-FAILED — … golden path unaffected` and the goldens
run (verified device stays healthy after the caught panic). **Parent decision needed:** fix
`ATOMIC_PROBE` literal as `(u64(0xDEADBEEF) << 32) | u64(0xCAFE1234)` in a WGSL commit,
or keep probe opt-in.

## 5. FINDING: llvmpipe advertises int64 but FAILS the goldens

`WGPU_DEVICE=llvmpipe ./gate_c1` → `GPU_PERM ×3 PASS` but `GPU_MERKLE … [FAIL]`
(`GATE_C1: FAIL`, exit 2). Mesa's software path produces wrong results in the merkle
derive/compress leg despite advertising SHADER_INT64 + all-ops atomics. This is
exactly why capability-driven auto-selection (never picks llvmpipe: score
`(1, 2147483647, 0)` < A16) plus honest printed tables matter; keep as community-matrix row.

## 6. Parent-note follow-up: GRID vGPU Vulkan

Parent suspected port-lab A16 exposes no Vulkan compute. **Refuted by evidence:** wgpu
enumerates `NVIDIA A16` vendor 0x10de device 0x25b6 DiscreteGpu, driver 620.70.06,
Vulkan instance 1.4.309 (vulkaninfo), and all three gates PASS byte-exact on it. Only
the probe WGSL (§4) failed, identically on the pristine baseline.

## 7. Files

- NEW `src/adapters.rs` (enumerate/select/print_table/list_adapters/device_substring_from_env)
- `src/device.rs` (InitOpts.device_substring; capability selection; fail-closed; probe catch_unwind)
- `src/lib.rs` (`pub mod adapters;`)
- `src/bin/{gate_c1,ntt_test,fri_commit_test}.rs` (--list-adapters, WGPU_DEVICE wiring, probe Err arm)
- `README.md` (Running-it + Status updates)
- port-lab artifacts: /root/wgpu-w1/{W1-REPORT.md, fri_auto.out→/tmp/*.out, build.log}, /root/wgpu-baseline/gate_c1_pristine.out

## Parent addendum (post-review)

The naga probe rejection (finding #1) was fixed on the same day, two WGSL edits:
1. `u64(0xDEADBEEFCAFE1234)` -> halves-built `(u64(0xDEADBEEFu) << 32u) | u64(0xCAFE1234u)`
   (literals > 2^63 are not universal-WGSL; naga@30 rejects them).
2. `r.consumed` -> `r.exchanged` (spec member name; `consumed` was a dead Tint-era alias).

Verified on port-lab A16 through the new selector, 3/3 runs:
`U64_ATOMICS: WORKING (cas_winners=1 sum=0xdeadbeefcafe1314 expect base+winner+224)`
+ GATE_C1 PASS, ntt_test PASS, fri_commit_test root0 PASS. The Arc B70 row + probe
re-verification is the parent's next step once the b70 engine yields (engine-on-card rule).
