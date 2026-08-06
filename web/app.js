// YaoYoroZu 顔加工MVP
// カメラ → MediaPipe FaceLandmarker で顔の点を検出 → WebGLで「美肌 / 目を大きく / 輪郭スリム」
// 出力canvasは canvas.captureStream() でそのままWebRTCに流せる（= 通話相手に届く映像）。

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

// スライダー
const controls = {
  smooth: 0.50, // 美肌（肌の平滑化）
  bright: 0.28, // 明るさ・血色
  eye: 0.18,    // 目を大きく（控えめ）
  slim: 0.15,   // 輪郭スリム（控えめ）
};
function bindSlider(id) {
  const input = document.getElementById(id);
  const out = document.getElementById(id + "Out");
  input.addEventListener("input", () => {
    controls[id] = parseFloat(input.value);
    out.textContent = input.value;
  });
}
["smooth", "bright", "eye", "slim"].forEach(bindSlider);

// プリセット
const PRESETS = {
  natural: { smooth: 0.40, bright: 0.22, eye: 0.10, slim: 0.08 },
  beauty:  { smooth: 0.55, bright: 0.32, eye: 0.20, slim: 0.16 },
  strong:  { smooth: 0.65, bright: 0.40, eye: 0.34, slim: 0.28 },
};
document.querySelectorAll("[data-preset]").forEach((b) => {
  b.addEventListener("click", () => {
    const p = PRESETS[b.dataset.preset];
    for (const k in p) {
      controls[k] = p[k];
      const input = document.getElementById(k);
      const out = document.getElementById(k + "Out");
      if (input) { input.value = p[k]; out.textContent = p[k].toFixed(2); }
    }
  });
});

// 「加工前と比較」= 押している間は素の映像
let bypass = false;
const hold = (v) => () => { bypass = v; };
compareBtn.addEventListener("mousedown", hold(true));
compareBtn.addEventListener("touchstart", hold(true), { passive: true });
["mouseup", "mouseleave", "touchend"].forEach((e) =>
  compareBtn.addEventListener(e, hold(false)));

// ---------------- WebGL2 セットアップ ----------------
const gl = canvas.getContext("webgl2", { premultipliedAlpha: false });
if (!gl) {
  statusEl.textContent = "このブラウザはWebGL2に対応していません";
  throw new Error("no webgl2");
}

const VERT = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

// 顔のワープ（ゆがみ）は「出力座標 → どの元座標をサンプルするか」で表現する。
const FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 outColor;

uniform sampler2D uTex;
uniform vec2  uTexel;      // 1/解像度
uniform float uAspect;     // 幅/高さ（円形補正用）
uniform int   uHasFace;
uniform int   uBypass;
uniform int   uMirror;     // 左右ミラー（自撮り用）

uniform float uSmooth;
uniform float uBright;
uniform float uEye;
uniform float uSlim;

uniform vec2  uEyeL;       // 左目中心（video空間 0..1）
uniform vec2  uEyeR;       // 右目中心
uniform float uEyeRad;     // 目の効果半径
uniform float uFaceCx;     // 顔の中心x
uniform vec2  uOvalC;      // 顔オーバルの中心
uniform vec2  uOvalR;      // 顔オーバルの半径(x,y)

// 局所的に拡大（目を大きく見せる）
vec2 magnify(vec2 uv, vec2 c, float R, float strength) {
  vec2 d = uv - c;
  d.x *= uAspect;
  float r = length(d);
  if (r < R) {
    // 中心ほど「内側」をサンプル＝拡大。縁でなめらかに1.0へ戻す。控えめに。
    float t = r / R;
    float s = mix(1.0 - strength * 0.55, 1.0, smoothstep(0.0, 1.0, t));
    d *= s;
  }
  d.x /= uAspect;
  return c + d;
}

// 輪郭スリム：あご付近ほど、より外側をサンプル＝横幅が縮んで見える
vec2 slimFace(vec2 uv, float cx, vec2 ovalC, vec2 ovalR, float strength) {
  float band = smoothstep(ovalC.y - ovalR.y * 0.1, ovalC.y + ovalR.y, uv.y);
  float dx = uv.x - cx;
  uv.x = cx + dx * (1.0 + strength * 0.35 * band);
  return uv;
}

float ovalMask(vec2 uv, vec2 c, vec2 r) {
  vec2 d = (uv - c) / r;
  return 1.0 - smoothstep(0.82, 1.0, length(d));
}

void main() {
  // canvas座標(y上向き) → 映像座標(y下向き＝ランドマークと同じ、左右はミラー)
  vec2 raw = vec2(uMirror == 1 ? 1.0 - vUv.x : vUv.x, 1.0 - vUv.y);

  if (uBypass == 1 || uHasFace == 0) {
    outColor = texture(uTex, raw);
    return;
  }

  vec2 src = raw;
  src = slimFace(src, uFaceCx, uOvalC, uOvalR, uSlim);
  src = magnify(src, uEyeL, uEyeRad, uEye);
  src = magnify(src, uEyeR, uEyeRad, uEye);

  vec3 col = texture(uTex, src).rgb;

  // 美肌：周囲を平均したぼかしを顔マスクの範囲だけブレンド
  float mask = ovalMask(raw, uOvalC, uOvalR);
  if (uSmooth > 0.001 && mask > 0.001) {
    vec3 blur = vec3(0.0);
    float total = 0.0;
    for (int y = -2; y <= 2; y++) {
      for (int x = -2; x <= 2; x++) {
        vec2 off = vec2(float(x), float(y)) * uTexel * 2.6;
        blur += texture(uTex, src + off).rgb;
        total += 1.0;
      }
    }
    blur /= total;
    col = mix(col, blur, clamp(uSmooth, 0.0, 0.9) * mask);
  }

  // 明るさ・血色（顔だけ・控えめに）
  col += uBright * mask * vec3(0.07, 0.035, 0.04);
  col = clamp(col, 0.0, 1.0);

  outColor = vec4(col, 1.0);
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS))
  throw new Error(gl.getProgramInfoLog(prog));
gl.useProgram(prog);

const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, "aPos");
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

const tex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, tex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

const U = {};
for (const name of ["uTex","uTexel","uAspect","uHasFace","uBypass","uMirror","uSmooth","uBright","uEye","uSlim","uEyeL","uEyeR","uEyeRad","uFaceCx","uOvalC","uOvalR"]) {
  U[name] = gl.getUniformLocation(prog, name);
}

// ---------------- MediaPipe ----------------
let faceLandmarker = null;
async function initFace() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

// FaceLandmarker のランドマーク添字（478点・虹彩あり）
const IDX = {
  irisL: 468, irisR: 473,
  eyeLout: 33, eyeLin: 133, eyeRin: 362, eyeRout: 263,
  cheekL: 234, cheekR: 454,
  forehead: 10, chin: 152,
};

// ---------------- ループ ----------------
let lastTime = 0, frames = 0, fpsT = 0;

function setUniformsFromFace(lm) {
  const eyeL = [lm[IDX.irisL].x, lm[IDX.irisL].y];
  const eyeR = [lm[IDX.irisR].x, lm[IDX.irisR].y];
  const eyeW = Math.hypot(lm[IDX.eyeLin].x - lm[IDX.eyeLout].x, lm[IDX.eyeLin].y - lm[IDX.eyeLout].y);
  const faceCx = (lm[IDX.cheekL].x + lm[IDX.cheekR].x) / 2;
  const ovalCx = (lm[IDX.cheekL].x + lm[IDX.cheekR].x) / 2;
  const ovalCy = (lm[IDX.forehead].y + lm[IDX.chin].y) / 2;
  const ovalRx = Math.abs(lm[IDX.cheekR].x - lm[IDX.cheekL].x) / 2 * 1.15;
  const ovalRy = Math.abs(lm[IDX.chin].y - lm[IDX.forehead].y) / 2 * 1.1;

  gl.uniform1i(U.uHasFace, 1);
  gl.uniform2f(U.uEyeL, eyeL[0], eyeL[1]);
  gl.uniform2f(U.uEyeR, eyeR[0], eyeR[1]);
  gl.uniform1f(U.uEyeRad, Math.max(eyeW * 1.25, 0.025));
  gl.uniform1f(U.uFaceCx, faceCx);
  gl.uniform2f(U.uOvalC, ovalCx, ovalCy);
  gl.uniform2f(U.uOvalR, ovalRx, ovalRy);
}

function render() {
  gl.uniform1i(U.uTex, 0);
  gl.uniform2f(U.uTexel, 1 / canvas.width, 1 / canvas.height);
  gl.uniform1f(U.uAspect, canvas.width / canvas.height);
  gl.uniform1i(U.uBypass, bypass ? 1 : 0);
  gl.uniform1i(U.uMirror, 1);
  gl.uniform1f(U.uSmooth, controls.smooth);
  gl.uniform1f(U.uBright, controls.bright);
  gl.uniform1f(U.uEye, controls.eye);
  gl.uniform1f(U.uSlim, controls.slim);

  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

function loop(now) {
  if (video.readyState >= 2) {
    if (faceLandmarker) {
      const res = faceLandmarker.detectForVideo(video, now);
      if (res.faceLandmarks && res.faceLandmarks.length > 0) {
        setUniformsFromFace(res.faceLandmarks[0]);
      } else {
        gl.uniform1i(U.uHasFace, 0);
      }
    }
    render();

    // FPS
    frames++;
    if (now - fpsT > 500) {
      fpsEl.textContent = Math.round((frames * 1000) / (now - fpsT)) + " fps";
      frames = 0; fpsT = now;
    }
  }
  requestAnimationFrame(loop);
}

// ---------------- 起動 ----------------
async function start() {
  startBtn.disabled = true;
  statusEl.textContent = "カメラとモデルを準備中…";
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    await initFace();

    statusEl.style.display = "none";
    compareBtn.disabled = false;
    fpsT = performance.now();
    requestAnimationFrame(loop);

    // WebRTCへ渡すための加工済みストリーム（次フェーズで使用）
    window.processedStream = canvas.captureStream(30);
  } catch (e) {
    console.error(e);
    statusEl.textContent = "起動に失敗: " + e.message + "\n（httpsまたはlocalhostで開いてください）";
    startBtn.disabled = false;
  }
}
startBtn.addEventListener("click", start);
