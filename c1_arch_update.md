# C1-ARCH.md - FORK-C GPU Prove Path Architecture Map

## Executive Summary

FORK-C is a WebGPU/WGSL-based proving backend for OpenVM 2.1.0, targeting portability across platforms (browser, non-Vulkan, macOS Metal). The primary hardware target is Intel Arc Pro B70 on b70 (Debian 13) with Mesa 26.1.6-1~bpo13+1 (ANV).

## GPU Prove Path Inventory

| Component | Source | WGSL Module | Status |
|-----------|--------|-------------|--------|
| **Poseidon2** | openvm-stark-backend-2.1.0 poseidon2.cuh | `wgsl.rs::source()` | ✅ PASS (C1 gate) |
| **NTT 4096-pt** | openvm-stark-backend-2.1.0 sppark ntt.cu | `wgsl.rs::NTT_WGSL` | ✅ PASS (C2 kernel) |
| **Merkle (build path)** | openvm-cuda-backend merkle_tree.cu | `wgsl.rs::source()` | ✅ PASS (C1 gate) |
| **FRI commit** | openvm-cuda-backend (entangled in WHIR) | `wgsl.rs::source()` | ✅ PASS (C3 kernel) |

### FRI Commit (Layer 0-5) - C3

- **Function**: Merkle commit on LDE output rows
- **Entry points**: `fri_commit_8col` (Layer 0), `fri_commit_1col` (Layers 1-5)
- **Workgroup**: 32 threads per kernel
- **Compute**: Poseidon2 full-width-16 permutation per row

**WGSL Layout** (shared 4-slot layout):
```wgsl
@group(0) @binding(0) var<storage, read> kc: array<u32>;      // constants
@group(0) @binding(1) var<storage, read_write> io1: array<u32>; // input
@group(0) @binding(2) var<storage, read_write> io2: array<u32>; // output
```

**Layer Pipeline**:
1. **Layer 0**: LDE output (2^7 × 8 rows) → `fri_commit_8col` → 1024 digests → merkle root
2. **Layer 1**: Col-fold (gamma) → 1 col → `fri_commit_1col` → 128 digests → merkle root
3. **Layers 2-5**: Row-fold (beta) → halve → `fri_commit_1col` → merkle root

### Poseidon2 (Width-16 BabyBear)
