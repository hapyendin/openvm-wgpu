// WGSL sources. The BabyBear mul paths + Poseidon2 perm body are copied
// VERBATIM from b70:~/appz/wgpu-probe/src/main.rs (golden-proven 11-15,
// 12,287-perm batch byte-exact + NTT golden). Only the merkle build-path
// entries (derive/sponge_init/compress) are new here, mirroring
// openvm-sycl-skel/b1/ported/merkle_kernels.hpp 1:1.
//
// FRI commit (C3) adds:
//   fri_commit_8col: Layer 0 commit on 8-col LDE rows
//   fri_commit_1col: Layer 1-5 commit on 1-col (padded to 8) rows

// u64 REDC path (requires Features::SHADER_INT64 on ANV)
pub const MUL_U64: &str = r#"
fn mul(a: u32, b: u32) -> u32 {
  let t = u64(a) * u64(b);
  let m = u32(t) * 2013265919u;
  let u = u64(m) * u64(2013265921u);
  var r: u32 = u32((t + u) >> 32u);
  if (r >= 2013265921u) { r = r - 2013265921u; }
  return r;
}
"#;

// pure-u32 16-bit-limb fallback (spec-portable, no int64 feature)
pub const MUL_U32: &str = r#"
struct W64 { lo: u32, hi: u32 }
fn mul64(a: u32, b: u32) -> W64 {
  let al = a & 65535u; let ah = a >> 16u;
  let bl = b & 65535u; let bh = b >> 16u;
  let p0 = al * bl; let p1 = ah * bl; let p2 = al * bh; let p3 = ah * bh;
  let mid = (p0 >> 16u) + (p1 & 65535u) + (p2 & 65535u);
  var r: W64;
  r.lo = (p0 & 65535u) | (mid << 16u);
  r.hi = p3 + (p1 >> 16u) + (p2 >> 16u) + (mid >> 16u);
  return r;
}
fn mul(a: u32, b: u32) -> u32 {
  let t = mul64(a, b);
  let m = t.lo * 2013265919u;
  let u = mul64(m, 2013265921u);
  let co: u32 = select(0u, 1u, t.lo != 0u);
  var r = t.hi + u.hi + co;
  if (r >= 2013265921u) { r = r - 2013265921u; }
  return r;
}
"#;

const COMMON: &str = r#"
@group(0) @binding(0) var<storage, read> kc: array<u32>;
@group(0) @binding(1) var<storage, read_write> io1: array<u32>;
@group(0) @binding(2) var<storage, read_write> io2: array<u32>;
struct Uni { seed: u32, a1: u32, a2: u32, a3: u32 };
@group(0) @binding(3) var<uniform> uni: Uni;

const PP: u32 = 2013265921u;
const HALF_P1_C: u32 = 1006632961u;
const R2W: u32 = 1172168163u;
const IC_OFF: u32 = 0u;
const TC_OFF: u32 = 64u;
const IRC_OFF: u32 = 128u;
const DG_OFF: u32 = 141u;

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
fn halvef(x: u32) -> u32 { return (x >> 1u) + ((x & 1u) * HALF_P1_C); }
fn dbLf(x: u32) -> u32 { return addm(x, x); }
fn d7f(x: u32) -> u32 {
  let x2 = mul(x, x);
  let x4 = mul(x2, x2);
  let x6 = mul(x4, x2);
  return mul(x6, x);
}
fn encw(x: u32) -> u32 { return mul(x, R2W); }
fn m_ext(cells: ptr<function, array<u32, 16u>>) {
  var ts: array<u32, 4u>;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) { ts[j] = 0u; }
  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let b = i * 4u;
    let x0 = (*cells)[b]; let x1 = (*cells)[b + 1u];
    let x2 = (*cells)[b + 2u]; let x3 = (*cells)[b + 3u];
    let t01 = addm(x0, x1);
    let t23 = addm(x2, x3);
    let t0123 = addm(t01, t23);
    let t01123 = addm(t0123, x1);
    let t01233 = addm(t0123, x3);
    (*cells)[b + 3u] = addm(t01233, dbLf(x0));
    (*cells)[b + 1u] = addm(t01123, dbLf(x2));
    (*cells)[b] = addm(t01123, t01);
    (*cells)[b + 2u] = addm(t01233, t23);
    for (var j2: u32 = 0u; j2 < 4u; j2 = j2 + 1u) {
      ts[j2] = addm(ts[j2], (*cells)[b + j2]);
    }
  }
  for (var i2: u32 = 0u; i2 < 16u; i2 = i2 + 1u) {
    (*cells)[i2] = addm((*cells)[i2], ts[i2 % 4u]);
  }
}
fn internal_layer(cells: ptr<function, array<u32, 16u>>) {
  var part_sum = (*cells)[1u];
  for (var i: u32 = 2u; i < 16u; i = i + 1u) {
    part_sum = addm(part_sum, (*cells)[i]);
  }
  let sum = addm(part_sum, (*cells)[0u]);
  (*cells)[0u] = subm(part_sum, (*cells)[0u]);
  (*cells)[1u] = addm(sum, (*cells)[1u]);
  (*cells)[2u] = addm(sum, dbLf((*cells)[2u]));
  (*cells)[3u] = addm(sum, halvef((*cells)[3u]));
  (*cells)[4u] = addm(sum, addm(dbLf((*cells)[4u]), (*cells)[4u]));
  (*cells)[5u] = addm(sum, dbLf(dbLf((*cells)[5u])));
  (*cells)[6u] = subm(sum, halvef((*cells)[6u]));
  (*cells)[7u] = subm(sum, addm(dbLf((*cells)[7u]), (*cells)[7u]));
  (*cells)[8u] = subm(sum, dbLf(dbLf((*cells)[8u])));
  (*cells)[9u] = addm(sum, mul((*cells)[9u], kc[DG_OFF + 9u]));
  (*cells)[10u] = addm(sum, halvef(halvef((*cells)[10u])));
  (*cells)[11u] = addm(sum, mul((*cells)[11u], kc[DG_OFF + 11u]));
  (*cells)[12u] = addm(sum, mul((*cells)[12u], kc[DG_OFF + 12u]));
  (*cells)[13u] = addm(sum, mul((*cells)[13u], kc[DG_OFF + 13u]));
  (*cells)[14u] = addm(sum, mul((*cells)[14u], kc[DG_OFF + 14u]));
  (*cells)[15u] = addm(sum, mul((*cells)[15u], kc[DG_OFF + 15u]));
}
fn full_round(cells: ptr<function, array<u32, 16u>>, koff: u32, r: u32) {
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    (*cells)[i] = addm((*cells)[i], kc[koff + r * 16u + i]);
  }
  for (var i2: u32 = 0u; i2 < 16u; i2 = i2 + 1u) {
    (*cells)[i2] = d7f((*cells)[i2]);
  }
  m_ext(cells);
}
fn partial_round(cells: ptr<function, array<u32, 16u>>, r: u32) {
  (*cells)[0u] = addm((*cells)[0u], kc[IRC_OFF + r]);
  (*cells)[0u] = d7f((*cells)[0u]);
  internal_layer(cells);
}
fn perm_mix(cells: ptr<function, array<u32, 16u>>) {
  m_ext(cells);
  for (var r: u32 = 0u; r < 4u; r = r + 1u) { full_round(cells, IC_OFF, r); }
  for (var r2: u32 = 0u; r2 < 13u; r2 = r2 + 1u) { partial_round(cells, r2); }
  for (var r3: u32 = 0u; r3 < 4u; r3 = r3 + 1u) { full_round(cells, TC_OFF, r3); }
}
fn decode(x: u32) -> u32 { return mul(x, 1u); }
"#;

const ENTRIES: &str = r#"
// --- batch perm (probe-proven shape: 1 perm per thread, in-place) ---
@compute @workgroup_size(32u)
fn perm_batch(@builtin(global_invocation_id) gid: vec3<u32>) {
  let base = gid.x * 16u;
  if (base >= arrayLength(&io1)) { return; }
  var cells: array<u32, 16u>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) { cells[i] = io1[base + i]; }
  perm_mix(&cells);
  for (var i2: u32 = 0u; i2 < 16u; i2 = i2 + 1u) { io1[base + i2] = cells[i2]; }
}

// --- merkle build path (mirrors b1 ported/merkle_kernels.hpp) ---
// derive_row: row_i = perm([i, seed, 0^14])[..8]  (canonical inputs encoded
// in-kernel; output = 8 mont words per row written to io1)
@compute @workgroup_size(32u)
fn derive_row(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i * 8u >= arrayLength(&io1)) { return; }
  var cells: array<u32, 16u>;
  for (var z: u32 = 0u; z < 16u; z = z + 1u) { cells[z] = 0u; }
  cells[0u] = encw(i);
  cells[1u] = encw(uni.seed);
  perm_mix(&cells);
  for (var j: u32 = 0u; j < 8u; j = j + 1u) { io1[i * 8u + j] = cells[j]; }
}

// merkle_tree_init (PaddingFreeSponge hash_slice, addr_space==3):
// state = [row(8) | 0^8] -> perm -> out[..8]
@compute @workgroup_size(32u)
fn merkle_init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i * 8u >= arrayLength(&io2)) { return; }
  var cells: array<u32, 16u>;
  for (var z: u32 = 0u; z < 16u; z = z + 1u) { cells[z] = 0u; }
  for (var j: u32 = 0u; j < 8u; j = j + 1u) { cells[j] = io1[i * 8u + j]; }
  perm_mix(&cells);
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) { io2[i * 8u + j2] = cells[j2]; }
}

// merkle_tree_compress (TruncatedPermutation<2,8,16>):
// state = [left(8) | right(8)] -> perm -> out[..8]
@compute @workgroup_size(32u)
fn merkle_compress(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i * 8u >= arrayLength(&io2)) { return; }
  var cells: array<u32, 16u>;
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    cells[j] = io1[i * 16u + j];
    cells[j + 8u] = io1[i * 16u + 8u + j];
  }
  perm_mix(&cells);
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) { io2[i * 8u + j2] = cells[j2]; }
}

// --- FRI commit kernels (C3) ---
//
// Layer 0: 8-col LDE rows -> hash each row -> 8-word digest
// Input: row-major 8-col matrix (n_rows x 8)
// Output: n_rows x 8-word digests (canonical form)
//
// Layer 1-5: 1-col rows (after col-fold/row-fold) -> pad to 8 -> hash
// Input: single column (n_rows x 1)
// Output: n_rows x 8-word digests (first word = hash, rest = 0)

// fri_commit_8col: Layer 0 commit - hash each 8-col row
// Identical logic to merkle_init (same Poseidon2 full-width-16 permutation)
@compute @workgroup_size(32u)
fn fri_commit_8col(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row * 8u >= arrayLength(&io1)) { return; }
  
  var cells: array<u32, 16u>;
  for (var z: u32 = 0u; z < 16u; z = z + 1u) { cells[z] = 0u; }
  
  // Load 8 words from input row, encode to Montgomery
  for (var j: u32 = 0u; j < 8u; j = j + 1u) {
    cells[j] = encw(io1[row * 8u + j]);
  }
  
  // Poseidon2 full permutation (same as merkle_init)
  perm_mix(&cells);
  
  // Output first 8 words (digest), decoded to canonical
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) {
    io2[row * 8u + j2] = decode(cells[j2]);
  }
}

// fri_commit_1col: Layer 1-5 commit - hash 1-col row padded to 8
// Same logic as merkle_init but input is 1 column (0-padded to 8)
@compute @workgroup_size(32u)
fn fri_commit_1col(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row * 8u >= arrayLength(&io2)) { return; }
  
  var cells: array<u32, 16u>;
  for (var z: u32 = 0u; z < 16u; z = z + 1u) { cells[z] = 0u; }
  
  // Load 1 word from input (col-folded/row-folded output)
  // The rest (cells[1..8]) stay zero (padded)
  cells[0u] = encw(io1[row]);
  
  // Poseidon2 full permutation (same as merkle_init)
  perm_mix(&cells);
  
  // Output first 8 words (digest), decoded to canonical
  for (var j2: u32 = 0u; j2 < 8u; j2 = j2 + 1u) {
    io2[row * 8u + j2] = decode(cells[j2]);
  }
}
"#;

pub fn source(mul_u64: bool) -> String {
    if mul_u64 {
        format!("{}{}
{}", MUL_U64, COMMON, ENTRIES)
    } else {
        format!("{}{}
{}", MUL_U32, COMMON, ENTRIES)
    }
}

// ---- u64-atomic probe (grind CAS / sumcheck atomicAdd feasibility) ----
pub const ATOMIC_PROBE: &str = r#"
struct At { v: atomic<u64>, pad: u32, c: atomic<u32> };
@group(0) @binding(0) var<storage, read_write> at: At;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(32u)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i == 0u) {
    atomicStore(&at.v, u64(0));
    atomicStore(&at.c, 0u);
  }
  workgroupBarrier();
  // CAS first-wins (grind pattern): exactly one thread must consume
  let r = atomicCompareExchangeWeak(&at.v, u64(0), u64(0xDEADBEEFCAFE1234) + u64(i));
  if (r.consumed) {
    atomicAdd(&at.c, 1u);
  }
  // u64 atomicAdd (sumcheck raw-sum pattern): 32 x 7
  atomicAdd(&at.v, u64(7));
  // publish
  let v = atomicLoad(&at.v);
  if (i == 0u) {
    out[0] = u32(v & u64(0xFFFFFFFF));   // expect (0xDEADBEEFCAFE1234 + winner_i) + 32*7
    out[1] = u32(v >> 32u);
    out[2] = u32(atomicLoad(&at.c));
  }
}
"#;

// ---- NTT 4096-pt (single workgroup, 512 threads) ----
// DIT-NTT with bit-reversed input, natural output. Twiddles are Montgomery words.
// Forward NTT uses omega (generator); inverse uses omega_inv scaled by 1/n.
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

@compute @workgroup_size(512u)
fn ntt_4096_fwd(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  for (var k: u32 = tid; k < 4096u; k = k + 512u) {
    data[br12(k)] = nin[k];
  }
  workgroupBarrier();
  var off: u32 = 0u;
  for (var s: u32 = 1u; s <= 12u; s = s + 1u) {
    let half = 1u << s;
    let q1 = half >> 1u;
    for (var i: u32 = tid; i < 2048u; i = i + 512u) {
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
  for (var k2: u32 = tid; k2 < 4096u; k2 = k2 + 512u) {
    st[k2] = data[k2];
  }
}

@compute @workgroup_size(512u)
fn ntt_4096_inv(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  for (var k: u32 = tid; k < 4096u; k = k + 512u) {
    data[br12(k)] = nin[k];
  }
  workgroupBarrier();
  var off: u32 = 0u;
  for (var s: u32 = 1u; s <= 12u; s = s + 1u) {
    let half = 1u << s;
    let q1 = half >> 1u;
    for (var i: u32 = tid; i < 2048u; i = i + 512u) {
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
  for (var k2: u32 = tid; k2 < 4096u; k2 = k2 + 512u) {
    st[k2] = mul(data[k2], inv4096);
  }
}
"#;
