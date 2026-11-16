pub mod adapters;
pub mod device;
pub mod field;
pub mod mat;
pub mod pipeline;
pub mod traits;
pub mod wgsl;

pub use device::{InitOpts, WgpuDevice};
pub use mat::WgpuMat;
pub use pipeline::{KernelCtx, NttPipelineSet, PipelineSet};

use crate::field::{constants_buffer_mont, decode, Poseidon2, MERKLE_SEED};
use crate::traits::*;

/// BabyBear backend over wgpu (FORK-C). Analogous to GenericGpuBackend<HS>
/// + GpuDevice pair in openvm-cuda-backend, collapsed since the hash scheme
/// is fixed to Poseidon2-BabyBear in C2.
pub struct WgpuBackend {
    pub dev: WgpuDevice,
    pub ps: PipelineSet,
    pub kctx: KernelCtx,
    pub dummy: wgpu::Buffer,
    pub cpu: Poseidon2, // CPU reference (golden checks)
    pub ntt_ps: NttPipelineSet,
}

#[derive(Clone)]
pub struct WgpuCtx {
    pub adapter_name: String,
}

pub struct DummyCommitter;

impl WgpuBackend {
    pub fn new(opts: InitOpts) -> WgpuBackend {
        let dev = WgpuDevice::init(opts);
        let u64_mul = dev.has_shader_int64;
        let ps = PipelineSet::new(&dev, u64_mul);
        let kctx = KernelCtx::new(&dev, &constants_buffer_mont());
        let dummy = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ntt_ps = NttPipelineSet::new(&dev, u64_mul);
        WgpuBackend {
            dev,
            ps,
            kctx,
            dummy,
            cpu: Poseidon2::new(),
            ntt_ps,
        }
    }

    fn dispatch_chunked(&self, pipe: &wgpu::ComputePipeline, m: &WgpuMat, words: usize) {
        // single-chunk fast path (all C2 sizes are single-chunk)
        assert_eq!(m.chunks.len(), 1, "chunked dispatch = C3 work");
        let _ = words;
        let bg = self
            .kctx
            .bind(&self.dev, &self.ps, &m.chunks[0], &self.dummy);
        let threads = m.len_words / 16; // perm_batch: 16 words/thread
        pipeline::dispatch(&self.dev, pipe, &bg, threads.max(1));
    }
    
    /// FRI commit 8-col layer (Layer 0): hash each 8-col row
    /// Input: n_rows x 8 words (row-major)
    /// Output: n_rows x 8-word digests (canonical)
    pub fn fri_commit_8col(&self, input: &WgpuMat, output: &WgpuMat) {
        assert_eq!(input.width, 8, "fri_commit_8col requires 8-col input");
        assert_eq!(output.width, 8, "fri_commit_8col produces 8-word digests");
        assert_eq!(input.height, output.height, "input/output row count mismatch");
        let n_rows = input.height;
        let bg = self
            .kctx
            .bind(&self.dev, &self.ps, &input.chunks[0], &output.chunks[0]);
        pipeline::dispatch(&self.dev, &self.ps.fri_commit_8col, &bg, n_rows);
    }
    
    /// FRI commit 1-col layer (Layer 1-5): hash 1-col padded to 8
    /// Input: n_rows x 1 word (column-major, padded to 8)
    /// Output: n_rows x 8-word digests (first word = hash, rest = 0)
    pub fn fri_commit_1col(&self, input: &WgpuMat, output: &WgpuMat) {
        assert_eq!(input.width, 1, "fri_commit_1col requires 1-col input");
        assert_eq!(output.width, 8, "fri_commit_1col produces 8-word digests");
        assert_eq!(input.height, output.height, "input/output row count mismatch");
        let n_rows = input.height;
        let bg = self
            .kctx
            .bind(&self.dev, &self.ps, &input.chunks[0], &output.chunks[0]);
        pipeline::dispatch(&self.dev, &self.ps.fri_commit_1col, &bg, n_rows);
    }
}

impl ProverBackendShim for WgpuBackend {
    type Val = u32;
    type Challenge = u32;
    type Commitment = Vec<u32>;
    type Matrix = WgpuMat;
    type PcsData = ();
}

impl MatrixDimensions for WgpuMat {
    fn height(&self) -> usize {
        self.height
    }
    fn width(&self) -> usize {
        self.width
    }
}

impl ProverDeviceShim for WgpuBackend {
    type DeviceCtx = WgpuCtx;
    fn device_ctx(&self) -> &WgpuCtx {
        // leaked-once static: fine for a singleton device per process
        static CTX: std::sync::OnceLock<WgpuCtx> = std::sync::OnceLock::new();
        CTX.get_or_init(|| WgpuCtx {
            adapter_name: self.dev.report.name.clone(),
        })
    }
}

impl DeviceDataTransporterShim<WgpuBackend> for WgpuBackend {
    fn transport_matrix_to_device(&self, host: &[u32], width: usize) -> WgpuMat {
        WgpuMat::from_host(&self.dev, host, width)
    }
    fn transport_matrix_from_device_to_host(&self, m: &WgpuMat) -> Vec<u32> {
        m.copy_out(&self.dev)
    }
}

impl Poseidon2ChipShim for WgpuBackend {
    fn perm_batch(&self, m: &mut WgpuMat, _n_states: usize) {
        self.dispatch_chunked(&self.ps.perm_batch, m, m.len_words);
    }
}

impl MerkleHasherShim for WgpuBackend {
    /// GPU legs: derive_row -> merkle_init -> 12x merkle_compress (all on the
    /// WGSL paths copied from wgpu-probe + b1 ported kernels).
    fn merkle_root_from_seed(&self, seed_canon: u32, num_leaves: usize) -> [u32; 8] {
        assert_eq!(seed_canon, (MERKLE_SEED % crate::field::P as u64) as u32);
        let n = num_leaves;
        self.kctx.set_seed(&self.dev, seed_canon);
        // rows: n x 8 mont words
        let rows = WgpuMat::zeros(&self.dev, n, 8);
        let derive_threads = n;
        let bg = self
            .kctx
            .bind(&self.dev, &self.ps, &rows.chunks[0], &self.dummy);
        pipeline::dispatch(&self.dev, &self.ps.derive_row, &bg, derive_threads);
        // leaves = init(rows)
        let mut cur = WgpuMat::zeros(&self.dev, n, 8);
        {
            let bg = self
                .kctx
                .bind(&self.dev, &self.ps, &rows.chunks[0], &cur.chunks[0]);
            pipeline::dispatch(&self.dev, &self.ps.merkle_init, &bg, n);
        }
        // compress levels
        while cur.len_words > 8 {
            let in_nodes = cur.len_words / 8;
            let out_nodes = in_nodes / 2;
            let nxt = WgpuMat::zeros(&self.dev, out_nodes, 8);
            {
                let bg = self
                    .kctx
                    .bind(&self.dev, &self.ps, &cur.chunks[0], &nxt.chunks[0]);
                pipeline::dispatch(&self.dev, &self.ps.merkle_compress, &bg, out_nodes);
            }
            cur = nxt;
        }
        let root_mont = cur.copy_out(&self.dev);
        let mut root = [0u32; 8];
        for i in 0..8 {
            root[i] = decode(root_mont[i]);
        }
        root
    }
}

impl TraceCommitterShim<WgpuBackend> for DummyCommitter {
    fn commit(&self, _traces: &[&WgpuMat]) -> (Vec<u32>, ()) {
        // LDE-NTT + row-hash commit = b2/b3-equivalent milestones; not in C2.
        unimplemented!("commit path scheduled for the b7-equivalent WGSL milestone")
    }
}
