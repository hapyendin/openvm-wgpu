# openvm-wgpu — a WGSL proving-backend substrate for OpenVM STARKs

> **Note:** this GitHub repository is a one-way mirror; the canonical history is maintained upstream. Issues and PRs are welcome here.

**TL;DR:** a full OpenVM 2.1.0-class STARK **prove path** — commitments, transcript, fractional-GKR, logup zerocheck, WHIR folding, PoW grind, proof assembly — reimplemented in Rust + [wgpu](https://github.com/gfx-rs/wgpu) (WGSL compute shaders), producing **byte-identical proofs on Intel Arc, NVIDIA, and AMD (RADV) GPUs through Vulkan — same artifact, zero vendor-specific kernels, no CUDA** — with a spec-portable pure-u32 math mode that removes even the int64 shader requirement, opening the browser/WebGPU tier.

As of 2026-11-18 the full single-process GPU prove is closed: every stage is golden-tested and run-deterministic, and the complete proof path replicates byte-for-byte across silicon vendors. See [Status](#status) and the [portability matrix](#portability-matrix).

## Why

Production zk provers are effectively CUDA-only. Meanwhile the `ProverBackend`/hal traits in `openvm-stark-sdk` are backend-generic by design — which means *any* GPU that speaks Vulkan (or WebGPU) can host the prover with one artifact. This repo demonstrates that claim at the primitive and layer level, with bit-exactness as the acceptance bar rather than a demo.

## Cross-vendor evidence

The same WGSL, compiled by the same naga, run on two vendors' GPUs, produces byte-identical PASS lines:

| Gate | What it checks | Intel Arc Pro B70 (Vulkan/ANV) | NVIDIA A16 (Vulkan) |
|---|---|---|---|
| `gate_c1` | Poseidon2 batch perms, Merkle root, NTT 4096 | ✅ `1264a0fd…` `a11b2829…` | ✅ byte-identical |
| `ntt_test` | vendored-NTT golden fwd/inv | ✅ | ✅ |
| `fri_commit_test` (v6) | real FRI/LDE layer commits vs golden `99fc8df2…` | ✅ CPU+GPU, parity 0/3008 | ✅ CPU+GPU |
| DuplexSponge | transcript gamma/betas from golden stream | ✅ | ✅ |

Numbers (Arc B70): NTT-4096 WGSL kernel ≈ 20–23 µs (vs ≈ 83 µs for a hand-tuned SYCL port of the same math); device init ≈ 183 ms once per process.

## What's in the box

```
src/
  field.rs          BabyBear Montgomery field (u64-REDC WGSL + u32 limb-split fallback path)
  constants_gen.rs  two-adic generators, Poseidon2 params (from upstream constants)
  ntt_wgsl.rs       radix-2/4 NTT (matches vendored sppark ntt.cu semantics)
  wgsl.rs           WGSL sources: poseidon2, merkle, ntt, fri_commit kernels
                    (field math injected via interchangeable u64-REDC / pure-u32 preludes)
  gkr.rs / gkr_wgsl.rs     fractional-GKR sumcheck host + WGSL eval/fold kernels
  logup_leaf.rs            logup-zerocheck leaf build + fold (2^26-row tables on-GPU)
  w2c.rs / whir_fold.rs    WHIR fold (mobius + DIT-FFT over EF4 + keccak merkle) + grind
  pipeline.rs       pipeline cache, 2 GB-binding-safe dispatch
  device.rs         wgpu device/queue bootstrap
  mat.rs            DeviceMatrix (hal::Matrix impl)
  traits.rs         ProverBackend/hal plumbing (CpuBackend interop)
src/bin/
  gate_c1.rs        primitive goldens (poseidon2 batch, merkle root, NTT)
  ntt_test.rs       NTT roundtrip goldens
  fri_commit_test.rs FRI commit vs golden layer roots
  u32mul_test.rs     host+GPU proof of the pure-u32 REDC (full-domain, 16.7M pairs)
  w2a_gkr.rs         GKR sumcheck gate (multi-instance, determinism xN)
  w2b_logup.rs       logup zerocheck gate (leaf->fold->roots)
  w2c_fold.rs / w2c_grind.rs  WHIR fold matrix + PoW grind (90-site golden)
  w2d_prove.rs       end-to-end single-process GPU prove -> proof stream + report
goldens/vectors.inc generated test vectors (pure math; no app data)
```

## Running it

```sh
cargo build --release
cargo run --release --bin gate_c1           # primitive goldens
cargo run --release --bin fri_commit_test   # FRI commit vs golden 99fc8df2…
```

Runs on any Vulkan-capable GPU exposing `SHADER_INT64` (NVIDIA, AMD/RADV, Intel/ANV), **or on any GPU at all with `OPENVM_WGPU_FORCE_U32=1`** — the spec-portable pure-u32 field path (no int64 shader features required, WebGPU/browser-tier math). Set `WGPU_BACKEND=vulkan`. Adapter choice is capability-driven: `--list-adapters` prints the full capability table, `WGPU_DEVICE=<substring>` pins by name, `--arc` (or `OPENVM_WGPU_ARC=1`) strict-modes to Intel Arc + int64; no qualifying adapter fails closed. The u32-limb path (no int64 requirement, browser-tier) is implemented in the field layer but not yet wired end-to-end.

## Status

- ✅ device layer, Poseidon2 (12,287-perm batches bit-exact), Merkle, NTT (4096/2²⁰ roundtrip), DuplexSponge transcript
- ✅ FRI/LDE commit pipeline: all golden layer roots + concat sha, GPU==CPU bit-exact parity
- ✅ fractional-GKR sumcheck + logup zerocheck kernels (WGSL), byte-exact vs CPU goldens, 5x run-deterministic
- ✅ WHIR fold + PoW grind (sequential-scan first-winner: determinism over atomics)
- ✅ end-to-end single-process GPU prove (`w2d_prove`): transcript→leaf→GKR→fold→grind→proof stream, ~5 GB VRAM / <500 MB host, matching an upstream reference `pv`
- ✅ cross-vendor proof: zero WGSL edits between Arc and NVIDIA; 5/5 matrix rows byte-identical incl. the full prove (see [portability matrix](#portability-matrix))
- ✅ WebGPU browser tier COMPLETE (11-20): a wasm32 build of the same backend runs inside real headless Chromium (Dawn→Vulkan) reproducing the **ENTIRE ladder byte-exact in the browser** — u32 REDC full domain, Poseidon2 perms, merkle `1264a0fd…`, NTT `a11b2829…`, PoW grind (`e69fddee…`, deterministic, u32-only — zero int64 features), w2a GKR folded claim (`fd53a33e…`), w2b logup leaf (`f5ecc63d…`/`8c06f8ac…`), w2c WHIR fold rows (`5b91a9e3…`/`28948222…`), and the **full w2d proof chain: proof sha `3e3bfbec62b817f5f9813a326af4698816bd9f95fd4a7d83a9ea9e6f4b7a3d0d` — the same bytes the native Arc/RADV/NVIDIA runs emit — produced end-to-end in a browser tab** (transcript→leaf→GKR→fold→22 GPU grinds→proof stream, ~4 min, ~6.9 GB world streamed in via chunked uploads). Bind-group atlases + batched mapAsync keep JS overhead ~2–3x lower. Harness in [`wasm-pack/`](wasm-pack/) (build scripts + mkpage_*/runfile_* per leg, no-modules + file://). Remaining polish only: performance (minutes, not seconds, in-browser) and packaging.
- ✅ AMD/RADV third vendor: 12/12 ladder items byte-identical on Radeon AI PRO R9700 (Mesa RADV) in all three modes (default u64, `FORCE_U32`, `LIMITS+U32`) incl. the full prove `3e3bfbec…`; int64-REDC works natively on RADV. One naga portability fix was required (documented below); it is byte-neutral on Arc.
- ✅ capability-driven adapter selection (W1): enumerate, score, pin, fail closed
- ✅ spec-portable u32-limb field path wired end-to-end: with `OPENVM_WGPU_FORCE_U32=1` (zero int64 features requested), the **entire gate ladder up to the full proof reproduces byte-identical** — 16.7M-pair host proof of the emulated REDC, u32 kernels within ~5% of u64 on Arc (memory-bound)
- ✅ WebGPU browser-tier compliance mode: with `OPENVM_WGPU_WEBGPU_LIMITS=1` (+ `OPENVM_WGPU_FORCE_U32=1`) the device is pinned to WebGPU spec-floor limits (128 MiB storage binding, 256 MiB buffer, 16 KiB workgroup memory, 256 invocations/workgroup) and wgpu's own validation audits every allocation. All buffers >floor transparently shard (pow2 windows ≤ binding cap, `ShardBuf`/`leaf_chunks`); **the full ladder incl. the ~2 GiB-leaf proof runs byte-identical under the floors with no measurable perf cost**. This is a native self-test for browser-tier feasibility — oversized bindings panic instead of silently passing.
- 🔜 browser/WebGPU tier phase 3: wasm build + async init/mapAsync, bind-group atlases, u32-CAS grind (sharding done)
- 🔜 third-silicon validation: AMD/RADV, Apple/Metal — gate suite ready (both mul paths), hardware calls welcome

## Portability matrix

### C-generation gates (2026-11-16, A16 driver 550.163.01)
| Gate | Arc BMG G31 | NVIDIA A16 |
|---|---|---|
| gate_c1 (poseidon2+merkle) | PASS 1264a0fd266c3e7b... | PASS byte-identical |
| ntt_test | PASS a11b282970362f31... | PASS byte-identical |
| fri_commit_test | PASS 99fc8df2 / 5f51333d... | PASS byte-identical |

### GATE W2 rows (2026-11-18; Arc = closed-green parent-verified; A16 = recheck this round)
A16 box: cloud VM, NVIDIA A16 vGPU,
adapter name=\"NVIDIA A16\" vendor=0x10de device=0x25b6 driver=\"NVIDIA\"
driver_info=\"620.70.06\" type=DiscreteGpu, Vulkan, wgpu 30.0.2, rustc 1.91.1.
Arc adapter: name=\"Intel(R) Graphics (BMG G31)\" vendor=0x8086 device=0xe223
Mesa 26.1.6. Host-side adapter-select bypass only (`OPENVM_WGPU_ANY=1` or `--any-gpu` adapter
bypass in device.rs); WGSL bytes identical (src/ sha256-tree 74fc5898c465a871...
identical on both boxes).

| Gate | Check | Arc BMG G31 (Vulkan/ANV) | NVIDIA A16 (Vulkan) |
|---|---|---|---|
| w2c_grind | 90 PoW sites GPU==CPU==recorded; 90/90 min==recorded | PASS run-hash e69fddee611e56ab609a2928c6ab1827a743c6d9235b3cc5a0e48eff43f715a5 x5 | PASS 2x run-hash byte-identical e69fddee...; u64 atomics = SHADER_INT64_ATOMIC_ALL_OPS |
| w2a_gkr agg-leaf | levels==cpu goldens; 0 value/snap mismatches; 5 runs det | PASS run-hash fd53a33e82d7319b551a97c0710e5e75fac26980c0b6c1b02bd79bef0929f691 x5 | PASS 5x run-hash byte-identical fd53a33e...; leaf_sha 5c65a7bbc355594d... + lvl13/14/16/18/20 snap shas == AT5 |
| w2c_fold agg-leaf | WHIR fold (mobius+DIT-FFT+keccak merkle) vs w2c-r1 dumps | PASS golden_ok x5; grs_sha r0 5b91a9e38bfa47fb..., r1 28948222729453e2... | PASS golden_ok x5; 15/15 report rows byte-identical vs Arc 5x5 report (ex-timing) |
| w2b_logup | 2^26 logup leaf GPU build + 25 GKR rounds, cpu9 world | PASS leaf f5ecc63de1646b06..., out_sha 8c06f8ac2b4070d6... x5 | PASS 5x leaf/out_sha byte-identical; results JSON == Arc byte-for-byte ex run times (A16 18s vs Arc 6.5s/run) |
| w2d_prove | full proof path, app-segments | PASS (proof sha == recorded, pv 5ed5d617...) | PASS — world 7,310,589,717 B shipped to the A16 box 2026-11-18; emitted proof sha 3e3bfbec62b817f5f9813a326af4698816bd9f95fd4a7d83a9ea9e6f4b7a3d0d == recorded == Arc (byte-identical, 15492 words); chain 2237/2237, leaf 01b3cb7e5db3a2de..., grinds 22/22, fold 4/4, gkr valmism=0, stream_equal=true, green=true; vm_hwm=739MB, wall 28s |

**Divergences found: NONE.** All shipped-byte gates are byte-identical Arc vs A16.
Driver delta across rounds (550.163.01 -> 620.70.06) did not move any golden.

## Provenance
- Reference runs on the primary Intel Arc dev box; per-gate result reports: W2{A,B,C,D}-REPORT.md.
- A16 recheck: same tree (sha-verified), independent box, per-gate result JSONs byte-identical to the Arc runs (ex timings).
- Browser-tier evidence: headless Chromium (Dawn->Vulkan) logs under `wasm-pack/`.

## Testing your GPU (we want your matrix)

If you have a GPU this can run on — **especially AMD, Apple (Metal), or anything unusual** — run the two binaries and open an issue with the output (or a PR adding your adapter to a community results table). The long-term goal is a public cross-vendor bit-exactness matrix; every data point counts.

## License

MIT. Derived by clean hand-translation from openvm's MIT-licensed CUDA kernels and constants tables; no third-party copyleft in the kernel path.
