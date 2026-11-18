// W2a: GKR/zerocheck sumcheck evaluation kernels (WGSL).
//
// All math is u32 BabyBear in MONTGOMERY-RAW limb domain (same as the rest of
// the fork-C tree: mul() is the u64-REDC path, golden-proven in C1/C2).
// EF = BinomialExtensionField<BabyBear,4> with x^4 = 11 (p3-baby-bear W const).
// 4-limb elements; addition is per-limb (a+b) mod P (Montgomery rep is a ring
// homomorphism for +,-).
//
// Protocol mirror (openvm-stark-backend 2.1.0 fractional_sumcheck_gkr.rs):
//  GKR round j (j = 1..t-1) operates on the level-(j+1) segment (2^{j+1} Fracs
//  in node-index order). Four logical MLE columns over 2^j points:
//    col0 = p(even nodes)  col1 = q(even nodes)
//    col2 = p(odd nodes)   col3 = q(odd nodes)
//  Each column is folded on the (2y, 2y+1) pair of ITS OWN index space
//  (same-parity nodes 2 apart in segment order); the eq vector (2^j points)
//  is folded on adjacent (2y, 2y+1) pairs.
//  s(x) = sum_y eq(x,y) * (p0 q1 + p1 q0 + lam q0 q1), observed at {1,2,3};
//  claims = (col0, col1, col2, col3) single row after all folds = (p0,q0,p1,q1).
//
// Kernels (all fixed-index, NO atomics, NO float — determinism law):
//  tree_add : W=8:  out[i] = frac_add(in[2i], in[2i+1])  (p1 q2 + p2 q1, q1 q2)
//             W=12: out[i] = in[2i] + in[2i+1]  (s-triple plain add)
//  unpack   : level-(j+1) segment, 4 Fracs/row -> 10-col matrix (2^{j-1} rows)
//  seval    : per-y s(1..3) partials + 512-y workgroup block tree -> block partials
//  fold     : MLE fold at EF r on the first dim: new[i] = a[2i] + (a[2i+1]-a[2i])*r
//
// 10-col row layout (40 u32): [eqe equ p0e p1e p0o p1o q0e q0o q1e q1o]
// (NOTE: p-cols interleave e/o pairs as p0e p1e | p0o p1o to match unpack.)
//  Row y consumes segment Fracs 4y..4y+3 (node order):
//   p0e = p(4y)    p0o = p(4y+2)    (col0: even nodes, same-parity pair)
//   p1e = p(4y+1)  p1o = p(4y+3)    (col2: odd nodes, same-parity pair)
//   q0e = q(4y)    q0o = q(4y+2)    (col1)
//   q1e = q(4y+1)  q1o = q(4y+3)    (col3)
//   eqe = eq(2y)   equ = eq(2y+1)
//  Claims = final row (p0e, q0e, p1e, q1e).
//  `base` (u32) is a dst write-base in u32 units (chunked sources); 0 = identity.

pub const GKR_WGSL: &str = r#"
const PP: u32 = 2013265921u;

// u64-REDC Montgomery multiply (same body as the golden-proven C1/C2 kernels)
fn mul(a: u32, b: u32) -> u32 {
  let t = u64(a) * u64(b);
  let m = u32(t) * 2013265919u;
  let u = u64(m) * u64(2013265921u);
  var r: u32 = u32((t + u) >> 32u);
  if (r >= 2013265921u) { r = r - 2013265921u; }
  return r;
}

fn addm(a: u32, b: u32) -> u32 {
  var s = a + b;
  if (s >= PP) { s = s - PP; }
  return s;
}

// EF (4 Montgomery limbs) ops
fn eadd(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let s = a + b;
  return select(s, s - vec4<u32>(PP), s >= vec4<u32>(PP));
}
fn esub(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  // WGSL select(f, t, cond) returns t when cond is true: a>=b -> a-b (t),
  // else a + (P-b) (f). Swapping these arms (AT5 bug: the a<b case wrapped
  // a-b mod 2^32, adding 2^32 mod P = R to every negative lane).
  return select(a + (vec4<u32>(PP) - b), a - b, a >= b);
}
// x^4 = 11 in BabyBear; W11 = [11]_Mont = 11 * 2^32 mod P
const W11: u32 = 939524073u;
fn emul(a: vec4<u32>, b: vec4<u32>) -> vec4<u32> {
  let s0 = addm(addm(mul(a.y, b.w), mul(a.z, b.z)), mul(a.w, b.y));
  let r0 = addm(mul(a.x, b.x), mul(W11, s0));
  let s1 = addm(mul(a.z, b.w), mul(a.w, b.z));
  let r1 = addm(addm(mul(a.x, b.y), mul(a.y, b.x)), mul(W11, s1));
  let s2 = mul(a.w, b.w);
  let r2 = addm(addm(addm(mul(a.x, b.z), mul(a.y, b.y)), mul(a.z, b.x)), mul(W11, s2));
  let r3 = addm(addm(addm(mul(a.x, b.w), mul(a.y, b.z)), mul(a.z, b.y)), mul(a.w, b.x));
  return vec4<u32>(r0, r1, r2, r3);
}
// MLE linear interpolation t(x) = t0 + (t1 - t0) * [x]  (x in {1,2,3}, base scalar)
// NOTE: [cx]_Mont is a SCALAR EF (cx,0,0,0). vec4<u32>(cx) would SPLAT all four
// lanes in WGSL and multiply by 2+2X+2X^2+2X^3 instead (AT5 bug).
fn lerp4(t0: vec4<u32>, t1: vec4<u32>, cx: u32) -> vec4<u32> {
  return eadd(t0, emul(esub(t1, t0), vec4<u32>(cx, 0u, 0u, 0u)));
}

const X1: u32 = 268435454u;   // [1]_Mont = 2^32 mod P
const X2: u32 = 536870908u;   // [2]_Mont
const X3: u32 = 805306362u;   // [3]_Mont

struct FracOut { p: vec4<u32>, q: vec4<u32> }
// projective Frac add: (p1 q2 + p2 q1, q1 q2)
fn frac_add(p1: vec4<u32>, q1: vec4<u32>, p2: vec4<u32>, q2: vec4<u32>) -> FracOut {
  return FracOut(eadd(emul(p1, q2), emul(p2, q1)), emul(q1, q2));
}

struct Row10 {
  eqe: vec4<u32>, equ: vec4<u32>,
  p0e: vec4<u32>, p0o: vec4<u32>,
  p1e: vec4<u32>, p1o: vec4<u32>,
  q0e: vec4<u32>, q0o: vec4<u32>,
  q1e: vec4<u32>, q1o: vec4<u32>,
}

struct U {
  w: u32,     // element width in u32 (tree_add): 8 = Frac, 12 = s-triple
  h: u32,     // current hypercube dim (seval)
  base: u32,  // dst write-base in u32 units (0 = identity)
  pad: u32,
  lam: vec4<u32>, // round batch lambda (EF)
  r: vec4<u32>,   // fold point (EF)
}

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> src2: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<uniform> u: U;

// W=8:  out[i] = frac_add(in[2i], in[2i+1])   (projective Frac add)
// W=12: out[i] = in[2i] + in[2i+1]            (s-triple plain add)
@compute @workgroup_size(256u)
fn tree_add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (u.w == 8u) {
    let ao = u.pad + i * 16u; // src base (split dispatches)
    let doff = u.base + i * 8u;
    let p1 = vec4<u32>(src[ao], src[ao + 1u], src[ao + 2u], src[ao + 3u]);
    let q1 = vec4<u32>(src[ao + 4u], src[ao + 5u], src[ao + 6u], src[ao + 7u]);
    let p2 = vec4<u32>(src[ao + 8u], src[ao + 9u], src[ao + 10u], src[ao + 11u]);
    let q2 = vec4<u32>(src[ao + 12u], src[ao + 13u], src[ao + 14u], src[ao + 15u]);
    let r = frac_add(p1, q1, p2, q2);
    dst[doff] = r.p.x;  dst[doff + 1u] = r.p.y;  dst[doff + 2u] = r.p.z;  dst[doff + 3u] = r.p.w;
    dst[doff + 4u] = r.q.x; dst[doff + 5u] = r.q.y; dst[doff + 6u] = r.q.z; dst[doff + 7u] = r.q.w;
  } else {
    for (var k: u32 = 0u; k < 12u; k = k + 4u) {
      let ao = i * 24u + k;
      let bo = (i * 2u + 1u) * 12u + k;
      let doff = u.base + i * 12u + k;
      let a = vec4<u32>(src[ao], src[ao + 1u], src[ao + 2u], src[ao + 3u]);
      let b = vec4<u32>(src[bo], src[bo + 1u], src[bo + 2u], src[bo + 3u]);
      let s = eadd(a, b);
      dst[doff] = s.x;  dst[doff + 1u] = s.y;  dst[doff + 2u] = s.z;  dst[doff + 3u] = s.w;
    }
  }
}

// level-(j+1) Frac segment (4 Fracs/row = 32 u32) + eq vec (2 EFs/row = 8 u32)
// -> 10-col matrix row (2^{j-1} rows); same-parity endpoints per reference cols.
@compute @workgroup_size(256u)
fn unpack(@builtin(global_invocation_id) gid: vec3<u32>) {
  let y = gid.x;
  let ro = u.base + y * 40u;
  let so = y * 32u;
  let eo = y * 8u;
  let eqe = vec4<u32>(src2[eo], src2[eo + 1u], src2[eo + 2u], src2[eo + 3u]);
  let equ = vec4<u32>(src2[eo + 4u], src2[eo + 5u], src2[eo + 6u], src2[eo + 7u]);
  let p0e = vec4<u32>(src[so], src[so + 1u], src[so + 2u], src[so + 3u]);
  let q0e = vec4<u32>(src[so + 4u], src[so + 5u], src[so + 6u], src[so + 7u]);
  let p1e = vec4<u32>(src[so + 8u], src[so + 9u], src[so + 10u], src[so + 11u]);
  let q1e = vec4<u32>(src[so + 12u], src[so + 13u], src[so + 14u], src[so + 15u]);
  let p0o = vec4<u32>(src[so + 16u], src[so + 17u], src[so + 18u], src[so + 19u]);
  let q0o = vec4<u32>(src[so + 20u], src[so + 21u], src[so + 22u], src[so + 23u]);
  let p1o = vec4<u32>(src[so + 24u], src[so + 25u], src[so + 26u], src[so + 27u]);
  let q1o = vec4<u32>(src[so + 28u], src[so + 29u], src[so + 30u], src[so + 31u]);
  dst[ro] = eqe.x;  dst[ro + 1u] = eqe.y;  dst[ro + 2u] = eqe.z;  dst[ro + 3u] = eqe.w;
  dst[ro + 4u] = equ.x;  dst[ro + 5u] = equ.y;  dst[ro + 6u] = equ.z;  dst[ro + 7u] = equ.w;
  dst[ro + 8u] = p0e.x;  dst[ro + 9u] = p0e.y;  dst[ro + 10u] = p0e.z;  dst[ro + 11u] = p0e.w;
  dst[ro + 12u] = p1e.x; dst[ro + 13u] = p1e.y; dst[ro + 14u] = p1e.z; dst[ro + 15u] = p1e.w;
  dst[ro + 16u] = p0o.x; dst[ro + 17u] = p0o.y; dst[ro + 18u] = p0o.z; dst[ro + 19u] = p0o.w;
  dst[ro + 20u] = p1o.x; dst[ro + 21u] = p1o.y; dst[ro + 22u] = p1o.z; dst[ro + 23u] = p1o.w;
  dst[ro + 24u] = q0e.x; dst[ro + 25u] = q0e.y; dst[ro + 26u] = q0e.z; dst[ro + 27u] = q0e.w;
  dst[ro + 28u] = q0o.x; dst[ro + 29u] = q0o.y; dst[ro + 30u] = q0o.z; dst[ro + 31u] = q0o.w;
  dst[ro + 32u] = q1e.x; dst[ro + 33u] = q1e.y; dst[ro + 34u] = q1e.z; dst[ro + 35u] = q1e.w;
  dst[ro + 36u] = q1o.x; dst[ro + 37u] = q1o.y; dst[ro + 38u] = q1o.z; dst[ro + 39u] = q1o.w;
}

// 512 y's per workgroup (1536 vec4 = 24KB workgroup memory < 32KB limit)
var<workgroup> ws: array<vec4<u32>, 1536u>;

fn fpart(row: Row10, cx: u32) -> vec4<u32> {
  var p0: vec4<u32>; var p1: vec4<u32>; var q0: vec4<u32>; var q1: vec4<u32>; var eq: vec4<u32>;
  if (cx == X1) {
    p0 = row.p0o; p1 = row.p1o; q0 = row.q0o; q1 = row.q1o; eq = row.equ;
  } else if (cx == X2) {
    p0 = lerp4(row.p0e, row.p0o, X2); p1 = lerp4(row.p1e, row.p1o, X2);
    q0 = lerp4(row.q0e, row.q0o, X2); q1 = lerp4(row.q1e, row.q1o, X2);
    eq = lerp4(row.eqe, row.equ, X2);
  } else {
    p0 = lerp4(row.p0e, row.p0o, X3); p1 = lerp4(row.p1e, row.p1o, X3);
    q0 = lerp4(row.q0e, row.q0o, X3); q1 = lerp4(row.q1e, row.q1o, X3);
    eq = lerp4(row.eqe, row.equ, X3);
  }
  let pprev = eadd(emul(p0, q1), emul(p1, q0));
  let qprev = emul(q0, q1);
  return emul(eq, eadd(pprev, emul(u.lam, qprev)));
}

// per-y s(1),s(2),s(3) partials + 512-y workgroup tree -> spart[block] (12 u32)
@compute @workgroup_size(512u)
fn seval(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let y = gid.x;
  // u.h carries the number of valid rows in this window (host passes rows_w).
  let n = u.h;
  var row: Row10;
  let bo = y * 40u;
  if (y < n) {
    row.eqe = vec4<u32>(src[bo], src[bo+1u], src[bo+2u], src[bo+3u]);
    row.equ = vec4<u32>(src[bo+4u], src[bo+5u], src[bo+6u], src[bo+7u]);
    row.p0e = vec4<u32>(src[bo+8u], src[bo+9u], src[bo+10u], src[bo+11u]);
    row.p1e = vec4<u32>(src[bo+12u], src[bo+13u], src[bo+14u], src[bo+15u]);
    row.p0o = vec4<u32>(src[bo+16u], src[bo+17u], src[bo+18u], src[bo+19u]);
    row.p1o = vec4<u32>(src[bo+20u], src[bo+21u], src[bo+22u], src[bo+23u]);
    row.q0e = vec4<u32>(src[bo+24u], src[bo+25u], src[bo+26u], src[bo+27u]);
    row.q0o = vec4<u32>(src[bo+28u], src[bo+29u], src[bo+30u], src[bo+31u]);
    row.q1e = vec4<u32>(src[bo+32u], src[bo+33u], src[bo+34u], src[bo+35u]);
    row.q1o = vec4<u32>(src[bo+36u], src[bo+37u], src[bo+38u], src[bo+39u]);
  }
  var v1: vec4<u32>; var v2: vec4<u32>; var v3: vec4<u32>;
  if (y < n) {
    v1 = fpart(row, X1);
    v2 = fpart(row, X2);
    v3 = fpart(row, X3);
  } else {
    v1 = vec4<u32>(0u); v2 = vec4<u32>(0u); v3 = vec4<u32>(0u);
  }
  if (u.pad == 43981u && lid.x == 0u) {
    dst[48u] = v1.x;  dst[49u] = v1.y;  dst[50u] = v1.z;  dst[51u] = v1.w;
    dst[52u] = v2.x;  dst[53u] = v2.y;  dst[54u] = v2.z;  dst[55u] = v2.w;
    dst[56u] = v3.x;  dst[57u] = v3.y;  dst[58u] = v3.z;  dst[59u] = v3.w;
    let dp0 = lerp4(row.p0e, row.p0o, X2);
    let dp1 = lerp4(row.p1e, row.p1o, X2);
    let dq0 = lerp4(row.q0e, row.q0o, X2);
    let dq1 = lerp4(row.q1e, row.q1o, X2);
    let deq = lerp4(row.eqe, row.equ, X2);
    let dS = eadd(eadd(emul(dp0, dq1), emul(dp1, dq0)), emul(u.lam, emul(dq0, dq1)));
    let dd = esub(row.p0o, row.p0e);
    let mp = emul(dd, vec4<u32>(X2, 0u, 0u, 0u));
    let dde = esub(row.equ, row.eqe);
    let mpe = emul(dde, vec4<u32>(X2, 0u, 0u, 0u));
    let mm0 = mul(row.p0o.z, X2);
    let o4 = array<vec4<u32>, 10u>(dp0, dd, mp, dde, mpe, row.p0e, row.eqe, row.equ, deq, dS);
    for (var m: u32 = 0u; m < 10u; m = m + 1u) {
      let vv = o4[m];
      let b = 60u + m * 4u;
      dst[b] = vv.x; dst[b + 1u] = vv.y; dst[b + 2u] = vv.z; dst[b + 3u] = vv.w;
    }
  }
  let t = lid.x * 3u;
  ws[t] = v1; ws[t + 1u] = v2; ws[t + 2u] = v3;
  workgroupBarrier();
  var s: u32 = 256u;
  for (; s >= 1u; s = s >> 1u) {
    if (lid.x < s) {
      let t2 = lid.x * 3u;
      ws[t2] = eadd(ws[t2], ws[t2 + s * 3u]);
      ws[t2 + 1u] = eadd(ws[t2 + 1u], ws[t2 + 1u + s * 3u]);
      ws[t2 + 2u] = eadd(ws[t2 + 2u], ws[t2 + 2u + s * 3u]);
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let block = u32(gid.x >> 9u);
    dst[block * 12u] = ws[0].x; dst[block * 12u + 1u] = ws[0].y;
    dst[block * 12u + 2u] = ws[0].z; dst[block * 12u + 3u] = ws[0].w;
    dst[block * 12u + 4u] = ws[1].x; dst[block * 12u + 5u] = ws[1].y;
    dst[block * 12u + 6u] = ws[1].z; dst[block * 12u + 7u] = ws[1].w;
    dst[block * 12u + 8u] = ws[2].x; dst[block * 12u + 9u] = ws[2].y;
    dst[block * 12u + 10u] = ws[2].z; dst[block * 12u + 11u] = ws[2].w;
  }
}

// Fold the LSB (in-row e/o) dim at EF point r: new row i holds the in-row
// folds of input rows 2i (endpoints -> out 'e' cols) and 2i+1 (-> 'o' cols):
// out[row].col = e + (o - e) * r.  This mirrors the reference fold on the
// column-array bit-0 dim (recorded CPU s_evals chain), NOT the row dim.
@compute @workgroup_size(256u)
fn fold(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let so = i * 80u;            // row 2i    (2i * 40)
  let bo = i * 80u + 40u;      // row 2i+1
  let ro = u.base + i * 40u;   // dst base (window split)
  // (e_off, o_off) per logical column: eq, p0, p1, q0, q1
  let pe = array<u32, 5u>(0u, 8u, 12u, 24u, 32u);
  let po = array<u32, 5u>(4u, 16u, 20u, 28u, 36u);
  var a: vec4<u32>; var b: vec4<u32>;
  for (var c: u32 = 0u; c < 5u; c = c + 1u) {
    let eo = pe[c];
    let oo = po[c];
    a = vec4<u32>(src[so + eo], src[so + eo + 1u], src[so + eo + 2u], src[so + eo + 3u]);
    b = vec4<u32>(src[so + oo], src[so + oo + 1u], src[so + oo + 2u], src[so + oo + 3u]);
    let d0 = eadd(a, emul(esub(b, a), u.r));
    dst[ro + eo] = d0.x; dst[ro + eo + 1u] = d0.y; dst[ro + eo + 2u] = d0.z; dst[ro + eo + 3u] = d0.w;
    a = vec4<u32>(src[bo + eo], src[bo + eo + 1u], src[bo + eo + 2u], src[bo + eo + 3u]);
    b = vec4<u32>(src[bo + oo], src[bo + oo + 1u], src[bo + oo + 2u], src[bo + oo + 3u]);
    let d1 = eadd(a, emul(esub(b, a), u.r));
    dst[ro + oo] = d1.x; dst[ro + oo + 1u] = d1.y; dst[ro + oo + 2u] = d1.z; dst[ro + oo + 3u] = d1.w;
  }
}
"#;
