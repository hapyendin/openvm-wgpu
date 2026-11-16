# openvm-wgpu — a WGSL proving-backend substrate for OpenVM STARKs

**TL;DR:** the memory/commit/hash layer of an OpenVM 2.1.0-class STARK prover, reimplemented in Rust + [wgpu](https://github.com/gfx-rs/wgpu) (WGSL compute shaders), producing **byte-identical results on Intel Arc and NVIDIA GPUs through Vulkan — same artifact, zero vendor-specific kernels, no CUDA**.

This is a work-in-progress backend substrate: the commit/NTT/Merkle/Poseidon2/transcript layer is closed and golden-tested; the fractional-GKR/zerocheck kernel surface is the remaining piece to a full GPU prove. See [Status](#status).

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
  pipeline.rs       pipeline cache, 2 GB-binding-safe dispatch
  device.rs         wgpu device/queue bootstrap
  mat.rs            DeviceMatrix (hal::Matrix impl)
  traits.rs         ProverBackend/hal plumbing (CpuBackend interop)
src/bin/
  gate_c1.rs        primitive goldens (poseidon2 batch, merkle root, NTT)
  ntt_test.rs       NTT roundtrip goldens
  fri_commit_test.rs FRI commit vs golden layer roots (the big one)
goldens/vectors.inc generated test vectors (pure math; no app data)
```

## Running it

```sh
cargo build --release
cargo run --release --bin gate_c1           # primitive goldens
cargo run --release --bin fri_commit_test   # FRI commit vs golden 99fc8df2…
```

Runs on any Vulkan-capable GPU (NVIDIA, AMD/RADV, Intel/ANV, llvmpipe in CI). Set `WGPU_BACKEND=vulkan`. Pass `--arc` (or `OPENVM_WGPU_ARC=1`) to strict-mode: assert Intel Arc + int64 path. The first adapter is used for now — device *selection* by capability and an install-time self-test are tracked in `C1-ARCH.md`.

## Status

- ✅ device layer, Poseidon2 (12,287-perm batches bit-exact), Merkle, NTT (4096/2²⁰ roundtrip), DuplexSponge transcript
- ✅ FRI/LDE commit pipeline: all golden layer roots + concat sha, GPU==CPU bit-exact parity
- ✅ cross-vendor proof: zero WGSL edits between Arc and NVIDIA
- 🔜 fractional-GKR / zerocheck kernels (the openvm stacked-commit exec path)
- 🔜 adapter auto-selection + capability-gated kernel variants (u64 REDC ⇄ u32 limb-split)
- 🔜 end-to-end prove matching an upstream reference `pv`

## Testing your GPU (we want your matrix)

If you have a GPU this can run on — **especially AMD, Apple (Metal), or anything unusual** — run the two binaries and open an issue with the output (or a PR adding your adapter to a community results table). The long-term goal is a public cross-vendor bit-exactness matrix; every data point counts.

## License

MIT. Derived by clean hand-translation from openvm's MIT-licensed CUDA kernels and constants tables; no third-party copyleft in the kernel path.
