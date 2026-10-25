// Bakes the viewer's environment lighting for the Rust thumbnail renderer.
//
// The viewer lights every model with three.js's RoomEnvironment, prefiltered
// by PMREMGenerator (see app/src/lib/scene.ts). The result does not depend on
// the model, so instead of re-implementing PMREM in Rust it is rendered once
// by three.js itself, in a headless Edge/Chrome, and stored here as RGBE:
//
//   assets/room_pmrem.rgbe.z   zlib(u32 width, u32 height, RGBE8 rows, top row first)
//
// The Rust shader samples it exactly like three's textureCubeUV(), so the
// thumbnails get the very same reflections and ambient light as the viewer.
// The DFG LUT of three's physical shading is copied the same way:
//
//   assets/dfg_lut.bin         16x16 RG16F, as in three/src/renderers/shaders/DFGLUTData.js
//
// Run after upgrading three.js:  node core/render/tools/bake-env.mjs
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { deflateSync } from 'node:zlib';

const here = dirname(fileURLToPath(import.meta.url));
const assets = join(here, '..', 'assets');
const three = join(here, '..', '..', '..', 'app', 'node_modules', 'three');

const PAGE = `<!doctype html><script type="importmap">{"imports":{"three":"/three/build/three.module.js","three/addons/":"/three/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
try {
  const renderer = new THREE.WebGLRenderer();
  const pmrem = new THREE.PMREMGenerator(renderer);
  // Same call as applyEnvironment() in app/src/lib/scene.ts.
  const target = pmrem.fromScene(new RoomEnvironment(), 0.04);
  const { width, height } = target;
  // Exact texel copy into a float target, which readRenderTargetPixels can read.
  const copy = new THREE.WebGLRenderTarget(width, height, { type: THREE.FloatType, depthBuffer: false });
  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: { src: { value: target.texture } },
    vertexShader: 'void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }',
    fragmentShader: 'uniform sampler2D src; out vec4 color; void main() { color = texelFetch(src, ivec2(gl_FragCoord.xy), 0); }',
  });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  renderer.setRenderTarget(copy);
  renderer.render(quad, new THREE.Camera());
  const pixels = new Float32Array(width * height * 4);
  renderer.readRenderTargetPixels(copy, 0, 0, width, height, pixels);
  // RGBE, rows flipped so the first row is the top of the texture (v = 0 in wgpu).
  const rgbe = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const s = ((height - 1 - y) * width + x) * 4;
      const d = (y * width + x) * 4;
      const m = Math.max(pixels[s], pixels[s + 1], pixels[s + 2]);
      if (m < 1e-32) continue;
      const e = Math.ceil(Math.log2(m));
      const scale = 256 / Math.pow(2, e);
      rgbe[d] = Math.min(255, Math.round(pixels[s] * scale));
      rgbe[d + 1] = Math.min(255, Math.round(pixels[s + 1] * scale));
      rgbe[d + 2] = Math.min(255, Math.round(pixels[s + 2] * scale));
      rgbe[d + 3] = e + 128;
    }
  }
  const header = new Uint32Array([width, height]);
  const body = new Uint8Array(8 + rgbe.length);
  body.set(new Uint8Array(header.buffer), 0);
  body.set(rgbe, 8);
  await fetch('/result', { method: 'POST', body });
} catch (e) {
  await fetch('/error', { method: 'POST', body: String(e && e.stack || e) });
}
</script>`;

const TYPES = { '.js': 'text/javascript', '.html': 'text/html' };

function findBrowser() {
  const candidates = [
    process.env.BROWSER,
    'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
    'C:/Program Files/Google/Chrome/Application/chrome.exe',
    '/usr/bin/google-chrome',
    '/usr/bin/chromium',
  ].filter(Boolean);
  const found = candidates.find((c) => existsSync(c));
  if (!found) throw new Error('No Edge or Chrome found; set BROWSER=<path>');
  return found;
}

function bakeDfgLut() {
  const source = readFileSync(join(three, 'src', 'renderers', 'shaders', 'DFGLUTData.js'), 'utf8');
  const values = source
    .slice(source.indexOf('['), source.indexOf(']'))
    .match(/0x[0-9a-f]+/gi)
    .map((v) => parseInt(v, 16));
  if (values.length !== 16 * 16 * 2) throw new Error(`DFG LUT: expected 512 values, got ${values.length}`);
  writeFileSync(join(assets, 'dfg_lut.bin'), Buffer.from(new Uint16Array(values).buffer));
  console.log('[bake] dfg_lut.bin');
}

bakeDfgLut();

const profile = mkdtempSync(join(tmpdir(), '3dv-bake-'));
let browser;
const server = createServer((req, res) => {
  if (req.method === 'POST') {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      res.end('ok');
      const body = Buffer.concat(chunks);
      if (req.url === '/result') {
        const out = join(assets, 'room_pmrem.rgbe.z');
        writeFileSync(out, deflateSync(body, { level: 9 }));
        console.log(`[bake] ${out} (${body.readUInt32LE(0)}x${body.readUInt32LE(4)})`);
      } else {
        console.error('[bake] page failed:', body.toString());
        process.exitCode = 1;
      }
      browser?.kill();
      server.close();
      // The browser may still hold its profile for a moment after the kill.
      setTimeout(() => {
        try {
          rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
        } catch {}
      }, 500);
    });
    return;
  }
  if (req.url === '/' || req.url === '/index.html') {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end(PAGE);
    return;
  }
  if (req.url.startsWith('/three/')) {
    const file = join(three, decodeURIComponent(req.url.slice('/three/'.length)));
    if (existsSync(file)) {
      res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
      res.end(readFileSync(file));
      return;
    }
  }
  res.writeHead(404);
  res.end();
});

server.listen(0, '127.0.0.1', () => {
  const { port } = server.address();
  browser = spawn(findBrowser(), [
    '--headless=new',
    '--enable-unsafe-swiftshader',
    '--use-angle=swiftshader',
    `--user-data-dir=${profile}`,
    '--no-first-run',
    `http://127.0.0.1:${port}/`,
  ], { stdio: 'ignore' });
});
setTimeout(() => {
  console.error('[bake] timed out');
  browser?.kill();
  process.exit(1);
}, 60000).unref();
