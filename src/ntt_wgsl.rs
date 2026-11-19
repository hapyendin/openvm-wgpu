// ---- NTT 4096-pt (single workgroup, 512 threads) ----
// DIT-NTT with bit-reversed input, natural output. Twiddles are Montgomery words.
// Uses the same mul function as other kernels (u64 REDC or u32 limb-split).
pub const NTT_WGSL: &str = r#"
// Mul function - same as in COMMON, but included here for standalone compilation
fn mul(a: u32, b: u32) -> u32 {
  let t = u64(a) * u64(b);
  let m = u32(t) * 2013265919u;
  let u = u64(m) * u64(2013265921u);
  var r: u32 = u32((t + u) >> 32u);
  if (r >= 2013265921u) { r = r - 2013265921u; }
  return r;
}

@group(0) @binding(0) var<storage, read> nin: array<u32>;
@group(0) @binding(1) var<storage, read> tw: array<u32>;
@group(0) @binding(2) var<storage, read_write> st: array<u32>;

const PP: u32 = 2013265921u;
const HALF_P1_C: u32 = 1006632961u;

fn addm(a: u32, b: u32) -> u32 {
  var s = a + b;
  if (s >= PP) { s = s - PP; }
  return s;
}
fn subm(a: u32, b: u32) -> u32 {
  var s: u32;
  if (a >= b) { s = a - b; } else { s = a + (PP - b); }
  return s;
}
fn br12(xin: u32) -> u32 {
  var x = xin; var r: u32 = 0u;
  for (var i: u32 = 0u; i < 12u; i = i + 1u) {
    r = (r << 1u) | (x & 1u);
    x = x >> 1u;
  }
  return r;
}

var<workgroup> data: array<u32, 4096u>;

@compute @workgroup_size(256u)
fn ntt_4096_fwd(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  for (var k: u32 = tid; k < 4096u; k = k + 256u) {
    data[br12(k)] = nin[k];
  }
  workgroupBarrier();
  var off: u32 = 0u;
  for (var s: u32 = 1u; s <= 12u; s = s + 1u) {
    let half = 1u << s;
    let q1 = half >> 1u;
    for (var i: u32 = tid; i < 2048u; i = i + 256u) {
      let b = i & (q1 - 1u);
      let blk = i >> (s - 1u);
      let pp = blk * half + b;
      let qq = pp + q1;
      let w = tw[off + b];
      let t = mul(data[qq], w);
      let u = data[pp];
      data[pp] = addm(u, t);
      data[qq] = subm(u, t);
    }
    off = off + (1u << (s - 1u));
    workgroupBarrier();
  }
  for (var k2: u32 = tid; k2 < 4096u; k2 = k2 + 256u) {
    st[k2] = data[k2];
  }
}

@compute @workgroup_size(256u)
fn ntt_4096_inv(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  for (var k: u32 = tid; k < 4096u; k = k + 256u) {
    data[br12(k)] = nin[k];
  }
  workgroupBarrier();
  var off: u32 = 0u;
  for (var s: u32 = 1u; s <= 12u; s = s + 1u) {
    let half = 1u << s;
    let q1 = half >> 1u;
    for (var i: u32 = tid; i < 2048u; i = i + 256u) {
      let b = i & (q1 - 1u);
      let blk = i >> (s - 1u);
      let pp = blk * half + b;
      let qq = pp + q1;
      let w = tw[off + b];
      let t = mul(data[qq], w);
      let u = data[pp];
      data[pp] = addm(u, t);
      data[qq] = subm(u, t);
    }
    off = off + (1u << (s - 1u));
    workgroupBarrier();
  }
  let inv4096 = 1509949441u;
  for (var k2: u32 = tid; k2 < 4096u; k2 = k2 + 256u) {
    st[k2] = mul(data[k2], inv4096);
  }
}
"#;
