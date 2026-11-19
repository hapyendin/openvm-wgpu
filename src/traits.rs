// Thin trait mirrors of openvm-stark-backend-2.1.0 src/prover/hal.rs (+ the
// device-internal traits of openvm-cuda-backend). Shape preserved 1:1 where
// names/arity exist; deltas documented in C2-REPORT.md:
//  - generics over StarkProtocolConfig/SC dropped (BabyBear-only for C2)
//  - Result<_, Self::Error> -> direct return (error enum lands with the
//    vendored-trait gate C3)
//  - MultiRapProver/OpeningProver/StackedReductionProver/WhirProver bodies
//    stubbed with todo!(): kernels land B7-B10-equivalent milestones.

pub trait MatrixDimensions {
    fn height(&self) -> usize;
    fn width(&self) -> usize;
}

/// Threading bound for `ProverBackendShim` types. Native keeps the CUDA-shape
/// Send+Sync; on wasm the wgpu web backend is Rc-based (single-threaded JS
/// event loop), so only MatrixDimensions applies there.
#[cfg(not(target_arch = "wasm32"))]
pub trait MatrixBound: MatrixDimensions + Send + Sync {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: MatrixDimensions + Send + Sync> MatrixBound for T {}
#[cfg(target_arch = "wasm32")]
pub trait MatrixBound: MatrixDimensions {}
#[cfg(target_arch = "wasm32")]
impl<T: MatrixDimensions> MatrixBound for T {}

/// Mirrors `ProverBackend` (hal.rs:23) — associated-type surface.
pub trait ProverBackendShim {
    type Val: Copy + Send + Sync;
    type Challenge: Copy + Send + Sync;
    type Commitment: Clone + Send + Sync;
    type Matrix: MatrixBound;
    type PcsData: Send + Sync;
}

/// Mirrors `DeviceDataTransporter` (hal.rs:141) — the four required fns
/// (the trait's 6, minus 2 defaulted, exactly as cuda-backend implements).
pub trait DeviceDataTransporterShim<B: ProverBackendShim> {
    fn transport_matrix_to_device(&self, host: &[u32], width: usize) -> B::Matrix;
    fn transport_matrix_from_device_to_host(&self, m: &B::Matrix) -> Vec<u32>;
}

/// Mirrors `TraceCommitter::commit` (hal.rs:84).
pub trait TraceCommitterShim<B: ProverBackendShim> {
    fn commit(&self, traces: &[&B::Matrix]) -> (B::Commitment, B::PcsData);
}

/// Mirrors `ProverDevice::device_ctx` (hal.rs:65) minimal.
pub trait ProverDeviceShim {
    type DeviceCtx: Clone + Send + Sync;
    fn device_ctx(&self) -> &Self::DeviceCtx;
}

/// cuda-backend internal: MerkleTreeConstructor::new_merkle_tree +
/// GpuMerkleHash row-hash ops (hash_scheme.rs / merkle_tree.rs).
pub trait MerkleHasherShim {
    /// derive rows then build root over 2^depth leaves (b1 vector form);
    /// returns canonical 8-word root.
    fn merkle_root_from_seed(&self, seed_canon: u32, num_leaves: usize) -> [u32; 8];
}

/// cuda-backend internal: poseidon2 batch perm over device state words
/// (mont domain) — the Chip<...> base-chip analogue at C2 shape.
pub trait Poseidon2ChipShim {
    /// in-place perm of n states (each 16 mont words) held in `m`.
    fn perm_batch(&self, m: &mut crate::mat::WgpuMat, n_states: usize);
}
