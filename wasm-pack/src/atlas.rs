//! Phase 3b: bind-group atlas + batched mapAsync readback for the browser tier.
//!
//! Atlas: pre-built bind groups keyed by (tag, buffer-set) — the buffer-set
//! key is (host address of each bound wgpu::Buffer, byte offset, byte size).
//! wgpu::Buffer is an Rc handle, so the atlas keeps clones of every bound
//! buffer alive in `keep`, guaranteeing the host addresses in the key are
//! never recycled while their cache entries exist (no phantom hits).
//!
//! ReadBatch: collect (buffer, byte offset, word count) read requests, copy
//! them all into ONE MAP_READ staging buffer in ONE encoder, and await a
//! SINGLE mapAsync round-trip instead of one per dispatch.
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::num::NonZeroU64;

/// One binding slot for an atlas bind group (entry index == position).
pub enum Bind {
    Whole(wgpu::Buffer),
    Slice(wgpu::Buffer, u64, Option<u64>),
}

impl Bind {
    pub fn whole(b: &wgpu::Buffer) -> Bind {
        Bind::Whole(b.clone())
    }
    pub fn slice(b: &wgpu::Buffer, off: u64, len: u64) -> Bind {
        Bind::Slice(b.clone(), off, Some(len))
    }
    pub fn slice_to_end(b: &wgpu::Buffer, off: u64) -> Bind {
        Bind::Slice(b.clone(), off, None)
    }
    fn buf(&self) -> &wgpu::Buffer {
        match self {
            Bind::Whole(b) | Bind::Slice(b, _, _) => b,
        }
    }
    fn res(&self) -> wgpu::BindingResource<'_> {
        match self {
            Bind::Whole(b) => b.as_entire_binding(),
            Bind::Slice(b, off, len) => {
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: b,
                    offset: *off,
                    size: len.map(|l| NonZeroU64::new(l).expect("nonzero len")),
                })
            }
        }
    }
    fn key(&self) -> (usize, u64, Option<u64>) {
        match self {
            Bind::Whole(b) => (std::ptr::from_ref(b) as usize, u64::MAX, None),
            Bind::Slice(b, o, l) => (std::ptr::from_ref(b) as usize, *o, *l),
        }
    }
}

pub struct Atlas {
    map: RefCell<HashMap<(&'static str, Vec<(usize, u64, Option<u64>)>), wgpu::BindGroup>>,
    keep: RefCell<Vec<wgpu::Buffer>>,
    hits: Cell<u64>,
    misses: Cell<u64>,
}

impl Atlas {
    pub fn new() -> Atlas {
        Atlas {
            map: RefCell::new(HashMap::new()),
            keep: RefCell::new(Vec::new()),
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }
    /// Cached bind group: entries are the `binds` vec in order (binding i).
    pub fn bind(
        &self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        tag: &'static str,
        binds: &[Bind],
    ) -> wgpu::BindGroup {
        let key: Vec<_> = binds.iter().map(|b| b.key()).collect();
        if let Some(bg) = self.map.borrow().get(&(tag, key.clone())) {
            self.hits.set(self.hits.get() + 1);
            return bg.clone();
        }
        self.misses.set(self.misses.get() + 1);
        let entries: Vec<wgpu::BindGroupEntry> = binds
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.res(),
            })
            .collect();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(tag),
            layout,
            entries: &entries,
        });
        for b in binds {
            self.keep.borrow_mut().push(b.buf().clone());
        }
        self.map.borrow_mut().insert((tag, key), bg.clone());
        bg
    }
    pub fn counters(&self) -> (u64, u64) {
        (self.hits.get(), self.misses.get())
    }
    pub fn reset_counters(&self) {
        self.hits.set(0);
        self.misses.set(0);
    }
}

/// One batched device->host read: N windows -> 1 staging copy -> 1 mapAsync.
pub struct ReadBatch {
    reqs: Vec<(wgpu::Buffer, u64, u64)>, // (buffer, byte offset, word count)
}

impl ReadBatch {
    pub fn new() -> ReadBatch {
        ReadBatch { reqs: Vec::new() }
    }
    pub fn push(&mut self, b: &wgpu::Buffer, off: u64, nwords: u64) {
        assert_eq!(off % 4, 0, "read offset must be word-aligned");
        self.reqs.push((b.clone(), off, nwords));
    }
    pub fn nmaps(&self) -> usize {
        usize::from(!self.reqs.is_empty())
    }
    pub async fn resolve(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<Vec<u32>>, String> {
        let total: u64 = self.reqs.iter().map(|r| r.2).sum();
        if total == 0 {
            return Ok(self.reqs.iter().map(|_| Vec::new()).collect());
        }
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wasm-read-batch"),
            size: total * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut enc =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let mut dst = 0u64;
            for (b, off, nw) in &self.reqs {
                enc.copy_buffer_to_buffer(b, *off, &staging, dst, nw * 4);
                dst += nw * 4;
            }
            queue.submit(Some(enc.finish()));
        }
        let slice = staging.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel::<Result<(), wgpu::BufferAsyncError>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        rx.await
            .map_err(|e| format!("map oneshot dropped: {e}"))?
            .map_err(|_| "map_async failed".to_string())?;
        let all: Vec<u32> = {
            let data = slice
                .get_mapped_range()
                .map_err(|e| format!("get_mapped_range: {e:?}"))?;
            data.chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        staging.unmap();
        let mut out: Vec<Vec<u32>> = Vec::with_capacity(self.reqs.len());
        let mut cur = 0usize;
        for (_, _, nw) in &self.reqs {
            let n = *nw as usize;
            out.push(all[cur..cur + n].to_vec());
            cur += n;
        }
        Ok(out)
    }
}
