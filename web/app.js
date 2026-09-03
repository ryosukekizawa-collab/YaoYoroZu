// YaoYoroZu 顔加工MVP（メッシュ変形版・2パス）
// カメラ → MediaPipe FaceLandmarker(478点) → 座標を時間方向に平滑化 →
// 顔メッシュを三角形分割し、パーツごとに点を動かして WebGL でテクスチャ再描画（パス1: FBOへ）→
// 画面座標で後処理（モザイクなど）（パス2: 画面へ）。
// 出力canvasは captureStream() でWebRTCへ。

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("input");
const canvas = document.getElementById("output");
const statusEl = document.getElementById("status");
const fpsEl = document.getElementById("fps");
const startBtn = document.getElementById("startBtn");
const compareBtn = document.getElementById("compareBtn");

// ---------------- パラメータ ----------------
const controls = {
  smooth: 0.50, bright: 0.28, mosaic: 0.0,
  eyeSize: 0.12, eyeSpace: 0.0, eyeTilt: 0.0,
  noseW: -0.08, noseL: 0.0,
  lipFull: 0.10, mouthW: 0.0,
  jawW: 0.12, chinL: 0.0, cheek: 0.0, faceL: 0.0,
};
const PRESETS = {
  reset:   { smooth:0, bright:0, mosaic:0, eyeSize:0, eyeSpace:0, eyeTilt:0, noseW:0, noseL:0, lipFull:0, mouthW:0, jawW:0, chinL:0, cheek:0, faceL:0 },
  natural: { smooth:.45, bright:.25, eyeSize:.08, eyeSpace:0, eyeTilt:0, noseW:-.08, noseL:0, lipFull:.08, mouthW:0, jawW:.10, chinL:0, cheek:0, faceL:0 },
  beauty:  { smooth:.55, bright:.32, eyeSize:.16, eyeSpace:0, eyeTilt:2, noseW:-.15, noseL:-.05, lipFull:.18, mouthW:.03, jawW:.20, chinL:.04, cheek:-.05, faceL:0 },
  another: { smooth:.55, bright:.30, eyeSize:.28, eyeSpace:.06, eyeTilt:6, noseW:-.22, noseL:.12, lipFull:.30, mouthW:-.05, jawW:.30, chinL:.12, cheek:-.10, faceL:.05 },
};

function fmt(v) { return Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2); }
function bindSlider(id) {
  const input = document.getElementById(id);
  const out = document.getElementById(id + "Out");
  if (!input) return;
  input.addEventListener("input", () => {
    controls[id] = parseFloat(input.value);
    if (out) out.textContent = fmt(controls[id]);
  });
}
Object.keys(controls).forEach(bindSlider);

document.querySelectorAll("[data-preset]").forEach((b) => {
  b.addEventListener("click", () => {
    const p = PRESETS[b.dataset.preset];
    for (const k in p) {
      controls[k] = p[k];
      const input = document.getElementById(k);
      const out = document.getElementById(k + "Out");
      if (input) { input.value = p[k]; if (out) out.textContent = fmt(p[k]); }
    }
  });
});

let bypass = false;
const hold = (v) => () => { bypass = v; };
compareBtn.addEventListener("mousedown", hold(true));
compareBtn.addEventListener("touchstart", hold(true), { passive: true });
["mouseup", "mouseleave", "touchend"].forEach((e) => compareBtn.addEventListener(e, hold(false)));

// ---------------- WebGL2 ----------------
const gl = canvas.getContext("webgl2", { premultipliedAlpha: false });
if (!gl) { statusEl.textContent = "このブラウザはWebGL2に対応していません"; throw new Error("no webgl2"); }

function compile(type, s) {
  const sh = gl.createShader(type); gl.shaderSource(sh, s); gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
  return sh;
}
function makeProgram(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

// --- パス1：メッシュ変形＋美肌（→ FBO）。alpha に顔マスクを書き込む。
const VERT1 = `#version 300 es
in vec2 aPos; in vec2 aUv; in float aMask;
uniform int uMirror;
out vec2 vUv; out float vMask;
void main() {
  float x = uMirror == 1 ? 1.0 - aPos.x : aPos.x;
  gl_Position = vec4(x * 2.0 - 1.0, 1.0 - aPos.y * 2.0, 0.0, 1.0);
  vUv = aUv; vMask = aMask;
}`;
const FRAG1 = `#version 300 es
precision highp float;
in vec2 vUv; in float vMask;
out vec4 outColor;
uniform sampler2D uTex;
uniform vec2 uTexel;
uniform int uBypass;
uniform float uSmooth, uBright;
void main() {
  vec3 c0 = texture(uTex, vUv).rgb;
  if (uBypass == 1) { outColor = vec4(c0, 0.0); return; }
  float m = clamp(vMask, 0.0, 1.0);
  vec3 col = c0;
  // 美肌：エッジを保つ平滑化（色が近い画素だけ平均＝輪郭や目鼻の線は残す）
  if (uSmooth > 0.001 && m > 0.001) {
    vec3 acc = vec3(0.0); float ws = 0.0;
    for (int y = -2; y <= 2; y++) for (int x = -2; x <= 2; x++) {
      vec3 cs = texture(uTex, vUv + vec2(float(x), float(y)) * uTexel * 2.4).rgb;
      vec3 d = cs - c0; float w = exp(-dot(d, d) * 45.0);
      acc += cs * w; ws += w;
    }
    col = mix(c0, acc / ws, clamp(uSmooth, 0.0, 0.95) * m);
  }
  col += uBright * m * vec3(0.07, 0.035, 0.04);
  outColor = vec4(clamp(col, 0.0, 1.0), m);
}`;

// --- パス2：画面座標での後処理（モザイク）。変形後の絵に対して真っ直ぐな格子でかける。
const VERT2 = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main() { vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;
const FRAG2 = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 outColor;
uniform sampler2D uScene;
uniform vec2 uRes;
uniform float uMosaic;
void main() {
  vec4 s = texture(uScene, vUv);
  vec3 col = s.rgb; float m = s.a;
  if (uMosaic > 0.001 && m > 0.001) {
    float block = mix(6.0, 44.0, clamp(uMosaic, 0.0, 1.0)); // 画面ピクセル単位のブロック
    vec2 q = (floor(gl_FragCoord.xy / block) + 0.5) * block / uRes;
    col = mix(col, texture(uScene, q).rgb, clamp(uMosaic * 1.3, 0.0, 1.0) * m);
  }
  outColor = vec4(col, 1.0);
}`;

const prog1 = makeProgram(VERT1, FRAG1);
const prog2 = makeProgram(VERT2, FRAG2);
const U1 = {}; for (const n of ["uTex","uTexel","uBypass","uMirror","uSmooth","uBright"]) U1[n] = gl.getUniformLocation(prog1, n);
const U2 = {}; for (const n of ["uScene","uRes","uMosaic"]) U2[n] = gl.getUniformLocation(prog2, n);

// パス1用 VAO（メッシュ）
const vao1 = gl.createVertexArray();
gl.bindVertexArray(vao1);
const bufPos = gl.createBuffer(), bufUv = gl.createBuffer(), bufMask = gl.createBuffer(), bufIdx = gl.createBuffer();
const aPos1 = gl.getAttribLocation(prog1, "aPos"), aUv1 = gl.getAttribLocation(prog1, "aUv"), aMask1 = gl.getAttribLocation(prog1, "aMask");
gl.bindBuffer(gl.ARRAY_BUFFER, bufPos); gl.enableVertexAttribArray(aPos1); gl.vertexAttribPointer(aPos1, 2, gl.FLOAT, false, 0, 0);
gl.bindBuffer(gl.ARRAY_BUFFER, bufUv); gl.enableVertexAttribArray(aUv1); gl.vertexAttribPointer(aUv1, 2, gl.FLOAT, false, 0, 0);
gl.bindBuffer(gl.ARRAY_BUFFER, bufMask); gl.enableVertexAttribArray(aMask1); gl.vertexAttribPointer(aMask1, 1, gl.FLOAT, false, 0, 0);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bufIdx);
gl.bindVertexArray(null);

// パス2用 VAO（全画面）
const vao2 = gl.createVertexArray();
gl.bindVertexArray(vao2);
const bufQuad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, bufQuad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
const aPos2 = gl.getAttribLocation(prog2, "aPos");
gl.enableVertexAttribArray(aPos2); gl.vertexAttribPointer(aPos2, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

// 映像テクスチャ
const tex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, tex);
for (const [k, v] of [[gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],[gl.TEXTURE_MIN_FILTER, gl.LINEAR],[gl.TEXTURE_MAG_FILTER, gl.LINEAR]]) gl.texParameteri(gl.TEXTURE_2D, k, v);

// FBO（パス1の出力）
let fbo = null, sceneTex = null;
function setupFBO(w, h) {
  if (fbo) { gl.deleteFramebuffer(fbo); gl.deleteTexture(sceneTex); }
  sceneTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, sceneTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  for (const [k, v] of [[gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE],[gl.TEXTURE_MIN_FILTER, gl.LINEAR],[gl.TEXTURE_MAG_FILTER, gl.LINEAR]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sceneTex, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

// ---------------- MediaPipe ----------------
let faceLandmarker = null;
async function initFace() {
  const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", delegate: "GPU" },
    runningMode: "VIDEO", numFaces: 1,
  });
}

const IDX = {
  irisL: 468, irisR: 473,
  eyeLout: 33, eyeLin: 133, eyeRin: 362, eyeRout: 263,
  noseTop: 6, noseTip: 1, noseL: 98, noseR: 327,
  lipTop: 13, lipBot: 14, mouthL: 61, mouthR: 291,
  chin: 152, forehead: 10, cheekL: 234, cheekR: 454,
};
const NL = 478, RING = 20, BORDER = 8, NV = NL + RING + BORDER;

const raw = new Float32Array(NL * 2);   // 生の座標
const sm  = new Float32Array(NL * 2);   // 平滑化した座標
let smInit = false;
const src = new Float32Array(NV * 2);   // 元の位置（テクスチャ座標）
const dst = new Float32Array(NV * 2);   // 変形後の位置
const mask = new Float32Array(NV);
let indices = null, frameCount = 0;

// 時間方向の平滑化：止まっているときは強く、速く動くときは弱く（遅れない）
function smoothLandmarks(lm) {
  for (let i = 0; i < NL; i++) {
    const rx = lm[i].x, ry = lm[i].y;
    if (!smInit) { sm[i*2] = rx; sm[i*2+1] = ry; continue; }
    const dx = rx - sm[i*2], dy = ry - sm[i*2+1];
    const d = Math.hypot(dx, dy);
    const a = Math.min(1, 0.18 + d * 50);
    sm[i*2] += dx * a; sm[i*2+1] += dy * a;
  }
  smInit = true;
}

// ---------------- Delaunay（Bowyer–Watson） ----------------
function inCircumcircle(a, b, c, p) {
  const ax = a[0]-p[0], ay = a[1]-p[1], bx = b[0]-p[0], by = b[1]-p[1], cx = c[0]-p[0], cy = c[1]-p[1];
  const det = (ax*ax+ay*ay)*(bx*cy-cx*by) - (bx*bx+by*by)*(ax*cy-cx*ay) + (cx*cx+cy*cy)*(ax*by-bx*ay);
  const orient = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
  if (orient === 0) return false;
  return orient > 0 ? det > 0 : det < 0;
}
function delaunay(pts) {
  const n = pts.length;
  const Pts = pts.concat([[-50,-50],[50,-50],[0,50]]);
  let tris = [[n, n+1, n+2]];
  for (let i = 0; i < n; i++) {
    const p = Pts[i], bad = [], keep = [];
    for (const t of tris) (inCircumcircle(Pts[t[0]], Pts[t[1]], Pts[t[2]], p) ? bad : keep).push(t);
    const edges = new Map();
    for (const t of bad) for (let e = 0; e < 3; e++) {
      const a = t[e], b = t[(e+1)%3]; const k = a < b ? a*100000+b : b*100000+a;
      edges.set(k, (edges.get(k) || 0) + 1);
    }
    tris = keep;
    for (const [k, c] of edges) if (c === 1) tris.push([Math.floor(k/100000), k%100000, i]);
  }
  const out = [];
  for (const t of tris) if (t[0] < n && t[1] < n && t[2] < n) out.push(t[0], t[1], t[2]);
  return new Uint16Array(out);
}

// ---------------- 変形 ----------------
const P = (arr, i) => [arr[i*2], arr[i*2+1]];
const dist = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1]);
const falloff = (d, R) => { const t = Math.min(d / R, 1); const u = 1 - t; return u * u * (3 - 2 * u); };
function radial(cx, cy, R, fn) {
  for (let i = 0; i < NL; i++) {
    const x = src[i*2], y = src[i*2+1], d = Math.hypot(x - cx, y - cy);
    if (d >= R) continue;
    fn(i, x - cx, y - cy, falloff(d, R));
  }
}

function deform(c) {
  dst.set(src);
  const irisL = P(src, IDX.irisL), irisR = P(src, IDX.irisR);
  const eyeW = dist(P(src, IDX.eyeLin), P(src, IDX.eyeLout));
  const eyeDist = dist(irisL, irisR);
  const cheekL = P(src, IDX.cheekL), cheekR = P(src, IDX.cheekR);
  const chin = P(src, IDX.chin), forehead = P(src, IDX.forehead);
  const faceW = dist(cheekL, cheekR), faceH = dist(chin, forehead);
  const faceCx = (cheekL[0] + cheekR[0]) / 2, faceCy = (chin[1] + forehead[1]) / 2;
  const cheekY = (cheekL[1] + cheekR[1]) / 2, chinY = chin[1];
  const noseTip = P(src, IDX.noseTip), noseTop = P(src, IDX.noseTop);
  const nL = P(src, IDX.noseL), nR = P(src, IDX.noseR);
  const noseC = [(nL[0] + nR[0]) / 2, (nL[1] + nR[1]) / 2];
  const noseWidth = dist(nL, nR), noseLen = dist(noseTop, noseTip);
  const lipTop = P(src, IDX.lipTop), lipBot = P(src, IDX.lipBot);
  const mouthC = [(lipTop[0] + lipBot[0]) / 2, (lipTop[1] + lipBot[1]) / 2];
  const mouthW = dist(P(src, IDX.mouthL), P(src, IDX.mouthR));

  const eyes = [[irisL, irisL[0] < irisR[0] ? -1 : 1], [irisR, irisR[0] < irisL[0] ? -1 : 1]];
  for (const [eye, sign] of eyes) {
    const R = eyeW * 1.9;
    const ang = -sign * c.eyeTilt * Math.PI / 180;
    const cs = Math.cos(ang), sn = Math.sin(ang);
    radial(eye[0], eye[1], R, (i, dx, dy, w) => {
      let ox = dx * c.eyeSize * w, oy = dy * c.eyeSize * w;
      ox += (dx * cs - dy * sn - dx) * w; oy += (dx * sn + dy * cs - dy) * w;
      ox += sign * c.eyeSpace * eyeDist * w;
      dst[i*2] += ox; dst[i*2+1] += oy;
    });
  }
  radial(noseC[0], noseC[1], noseWidth * 1.5, (i, dx, dy, w) => { dst[i*2] += dx * c.noseW * w; });
  radial(noseTip[0], noseTip[1], noseLen * 0.9, (i, dx, dy, w) => { dst[i*2+1] += c.noseL * noseLen * w; });
  radial(mouthC[0], mouthC[1], mouthW * 0.85, (i, dx, dy, w) => { dst[i*2+1] += dy * c.lipFull * w; });
  radial(mouthC[0], mouthC[1], mouthW * 0.95, (i, dx, dy, w) => { dst[i*2] += dx * c.mouthW * w; });
  for (let i = 0; i < NL; i++) {
    const y = src[i*2+1];
    if (y <= cheekY) continue;
    const t = Math.min((y - cheekY) / Math.max(chinY - cheekY, 1e-4), 1);
    const w = t * t * (3 - 2 * t);
    dst[i*2] += (faceCx - src[i*2]) * c.jawW * w;
  }
  radial(chin[0], chin[1], faceH * 0.35, (i, dx, dy, w) => { dst[i*2+1] += c.chinL * faceH * 0.5 * w; });
  const cheeks = [[cheekL, cheekL[0] < cheekR[0] ? -1 : 1], [cheekR, cheekR[0] < cheekL[0] ? -1 : 1]];
  for (const [ck, sign] of cheeks) radial(ck[0], ck[1], faceW * 0.35, (i, dx, dy, w) => { dst[i*2] += sign * c.cheek * faceW * 0.5 * w; });
  if (c.faceL !== 0) for (let i = 0; i < NL; i++) dst[i*2+1] += (src[i*2+1] - faceCy) * c.faceL;

  for (let i = 0; i < NL; i++) {
    const x = src[i*2], y = src[i*2+1];
    let m = 1;
    for (const e of [irisL, irisR]) m -= falloff(Math.hypot(x - e[0], y - e[1]), eyeW * 1.35);
    m -= falloff(Math.hypot(x - mouthC[0], y - mouthC[1]), mouthW * 0.6) * 0.8;
    mask[i] = Math.max(0, m);
  }
}

function buildOuter() {
  // 顔のすぐ外側に固定リング：変形の影響を顔まわりの狭い帯に閉じ込め、背景を動かさない
  const cheekL = P(src, IDX.cheekL), cheekR = P(src, IDX.cheekR), chin = P(src, IDX.chin), forehead = P(src, IDX.forehead);
  const cx = (cheekL[0] + cheekR[0]) / 2, cy = (chin[1] + forehead[1]) / 2;
  const rx = Math.abs(cheekR[0] - cheekL[0]) * 0.8, ry = Math.abs(chin[1] - forehead[1]) * 0.7;
  for (let k = 0; k < RING; k++) {
    const a = (k / RING) * Math.PI * 2, i = NL + k;
    src[i*2] = cx + Math.cos(a) * rx; src[i*2+1] = cy + Math.sin(a) * ry; mask[i] = 0;
  }
  const border = [[0,0],[0.5,0],[1,0],[1,0.5],[1,1],[0.5,1],[0,1],[0,0.5]];
  for (let k = 0; k < BORDER; k++) { const i = NL + RING + k; src[i*2] = border[k][0]; src[i*2+1] = border[k][1]; mask[i] = 0; }
}

function updateFromLandmarks(lm) {
  smoothLandmarks(lm);
  src.set(sm.subarray(0, NL * 2), 0);
  buildOuter();
  if (!indices || frameCount % 30 === 0) {
    const pts = []; for (let i = 0; i < NV; i++) pts.push([src[i*2], src[i*2+1]]);
    indices = delaunay(pts);
  }
  if (bypass) dst.set(src); else deform(controls);
  for (let i = NL; i < NV; i++) { dst[i*2] = src[i*2]; dst[i*2+1] = src[i*2+1]; }
}

const QUAD_POS = new Float32Array([0,0, 1,0, 0,1, 1,1]);
const QUAD_IDX = new Uint16Array([0,1,2, 1,3,2]);
const QUAD_MASK = new Float32Array([0,0,0,0]);

function drawMesh(pos, uv, msk, idx) {
  gl.bindVertexArray(vao1);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufPos); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufUv); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, bufMask); gl.bufferData(gl.ARRAY_BUFFER, msk, gl.DYNAMIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, bufIdx); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.DYNAMIC_DRAW);
  gl.drawElements(gl.TRIANGLES, idx.length, gl.UNSIGNED_SHORT, 0);
  gl.bindVertexArray(null);
}

function render(hasFace) {
  const W = canvas.width, H = canvas.height;
  // パス1：メッシュ変形＋美肌 → FBO
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, W, H);
  gl.useProgram(prog1);
  gl.uniform1i(U1.uTex, 0);
  gl.uniform2f(U1.uTexel, 1 / W, 1 / H);
  gl.uniform1i(U1.uBypass, bypass ? 1 : 0);
  gl.uniform1i(U1.uMirror, 1);
  gl.uniform1f(U1.uSmooth, controls.smooth);
  gl.uniform1f(U1.uBright, controls.bright);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
  gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  if (hasFace && indices) drawMesh(dst, src, mask, indices);
  else drawMesh(QUAD_POS, QUAD_POS, QUAD_MASK, QUAD_IDX);

  // パス2：画面座標の後処理 → 画面
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, W, H);
  gl.useProgram(prog2);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, sceneTex);
  gl.uniform1i(U2.uScene, 0);
  gl.uniform2f(U2.uRes, W, H);
  gl.uniform1f(U2.uMosaic, bypass ? 0 : controls.mosaic);
  gl.bindVertexArray(vao2);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
}

// ---------------- ループ ----------------
let frames = 0, fpsT = 0;
function loop(now) {
  if (video.readyState >= 2) {
    let hasFace = false;
    if (faceLandmarker) {
      const res = faceLandmarker.detectForVideo(video, now);
      if (res.faceLandmarks && res.faceLandmarks.length > 0) { updateFromLandmarks(res.faceLandmarks[0]); hasFace = true; }
      else smInit = false;
    }
    render(hasFace);
    frameCount++; frames++;
    if (now - fpsT > 500) { fpsEl.textContent = Math.round((frames * 1000) / (now - fpsT)) + " fps"; frames = 0; fpsT = now; }
  }
  requestAnimationFrame(loop);
}

// ---------------- 起動 ----------------
async function start() {
  startBtn.disabled = true;
  statusEl.textContent = "カメラとモデルを準備中…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    setupFBO(canvas.width, canvas.height);
    await initFace();
    statusEl.style.display = "none";
    compareBtn.disabled = false;
    fpsT = performance.now();
    requestAnimationFrame(loop);
    window.processedStream = canvas.captureStream(30);
  } catch (e) {
    console.error(e);
    statusEl.textContent = "起動に失敗: " + e.message + "\n（httpsまたはlocalhostで開いてください）";
    startBtn.disabled = false;
  }
}
startBtn.addEventListener("click", start);
