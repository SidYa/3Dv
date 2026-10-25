// Builds public/licenses.json for Options > Licenses: every third-party component
// that ships inside 3Dv, with its licence and the licence text found in its sources
// (same as in Hyper and Farba).
//
//   origin   projects 3Dv's code is ported from
//   bundled  decoders shipped next to the UI (public/draco, public/basis) or inlined
//   rust     crates compiled into the app for the build target (CARGO_BUILD_TARGET,
//            else this OS: Windows MSVC / Linux GNU)
//   js       npm packages bundled into the UI (production deps + the Svelte runtime)
//
// Run: npm run licenses (also runs as part of `npm run build`)
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const appDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = join(appDir, '..');
const core = join(repoRoot, 'core');
const out = join(appDir, 'public', 'licenses.json');

const texts = {};
function addText(text) {
  const norm = text.replace(/\r\n/g, '\n').trim();
  if (!norm) return null;
  const id = createHash('sha1').update(norm).digest('hex').slice(0, 12);
  texts[id] = norm;
  return id;
}

const LICENSE_FILE = /^(licen[cs]e|copying|notice|unlicense)([-._].*)?$/i;
function licenseTexts(dir) {
  if (!dir || !existsSync(dir)) return [];
  return readdirSync(dir)
    .filter((f) => LICENSE_FILE.test(f))
    .sort()
    .map((f) => {
      try {
        return addText(readFileSync(join(dir, f), 'utf8'));
      } catch {
        return null;
      }
    })
    .filter(Boolean);
}

const packages = [];

// --- Rust ---------------------------------------------------------------
function rustCrates() {
  const home =
    process.env.CARGO_HOME ?? join(process.env.USERPROFILE ?? process.env.HOME ?? '', '.cargo');
  const cargo = existsSync(join(home, 'bin', 'cargo.exe')) ? join(home, 'bin', 'cargo.exe') : 'cargo';
  const triple =
    process.env.CARGO_BUILD_TARGET ??
    (process.platform === 'win32' ? 'x86_64-pc-windows-msvc' : 'x86_64-unknown-linux-gnu');
  const raw = execFileSync(
    cargo,
    ['metadata', '--format-version', '1', '--filter-platform', triple],
    { cwd: core, encoding: 'utf8', maxBuffer: 256 * 1024 * 1024 },
  );
  const meta = JSON.parse(raw);
  const members = new Set(meta.workspace_members);
  // Walk the resolve graph from the workspace, normal deps only (build-time and dev-only
  // crates are not shipped).
  const nodes = new Map(meta.resolve.nodes.map((n) => [n.id, n]));
  const seen = new Set();
  const stack = [...members];
  while (stack.length) {
    const id = stack.pop();
    if (seen.has(id)) continue;
    seen.add(id);
    for (const d of nodes.get(id)?.deps ?? []) {
      if (d.dep_kinds.some((k) => k.kind === null)) stack.push(d.pkg);
    }
  }
  for (const p of meta.packages) {
    if (!seen.has(p.id) || members.has(p.id)) continue;
    packages.push({
      kind: 'rust',
      name: p.name,
      version: p.version,
      license: p.license ?? (p.license_file ? 'see licence file' : 'unknown'),
      repository: p.repository ?? p.homepage ?? `https://crates.io/crates/${p.name}`,
      texts: licenseTexts(dirname(p.manifest_path)),
    });
  }
}

// --- JavaScript ---------------------------------------------------------
function resolveDir(name, from) {
  let dir = from;
  while (true) {
    const cand = join(dir, 'node_modules', name);
    if (existsSync(join(cand, 'package.json'))) return cand;
    const up = dirname(dir);
    if (up === dir) return null;
    dir = up;
  }
}

function jsPackages() {
  const appPkg = JSON.parse(readFileSync(join(appDir, 'package.json'), 'utf8'));
  // Production deps + the Svelte runtime, which the compiler inlines into the bundle.
  const roots = [...Object.keys(appPkg.dependencies ?? {}), 'svelte'];
  const seen = new Set();
  const visit = (name, from, depth) => {
    const dir = resolveDir(name, from);
    if (!dir) return;
    const pkg = JSON.parse(readFileSync(join(dir, 'package.json'), 'utf8'));
    const key = `${pkg.name}@${pkg.version}`;
    if (seen.has(key)) return;
    seen.add(key);
    const repo = typeof pkg.repository === 'string' ? pkg.repository : pkg.repository?.url;
    packages.push({
      kind: 'js',
      name: pkg.name,
      version: pkg.version,
      license: typeof pkg.license === 'string' ? pkg.license : (pkg.license?.type ?? 'unknown'),
      repository: (repo ?? pkg.homepage ?? `https://www.npmjs.com/package/${pkg.name}`)
        .replace(/^git\+/, '')
        .replace(/\.git$/, ''),
      texts: licenseTexts(dir),
    });
    // Svelte's own deps are compiler-only; the runtime has none.
    if (name === 'svelte' || depth > 20) return;
    for (const dep of Object.keys(pkg.dependencies ?? {})) visit(dep, dir, depth + 1);
  };
  for (const r of roots) visit(r, appDir, 0);
}

// --- Origins and decoders (not discoverable from manifests) ---------------
const manual = [
  {
    kind: 'origin',
    name: 'three.js',
    version: 'r186',
    license: 'MIT',
    repository: 'https://github.com/mrdoob/three.js',
    note: 'The native thumbnail renderer (core/render) ports three.js: its glTF, OBJ/MTL, FBX, STL, PLY, Collada and 3DS loaders, its material shading and tone mapping, and embeds the RoomEnvironment lighting and DFG table baked by three.js.',
    texts: licenseTexts(resolveDir('three', appDir)),
  },
  {
    kind: 'bundled',
    name: 'Draco',
    version: '',
    license: 'Apache-2.0',
    repository: 'https://github.com/google/draco',
    note: 'Decoder of Draco-compressed glTF meshes (public/draco, shipped with three.js).',
    texts: [],
  },
  {
    kind: 'bundled',
    name: 'Basis Universal',
    version: '',
    license: 'Apache-2.0',
    repository: 'https://github.com/BinomialLLC/basis_universal',
    note: 'Transcoder of KTX2 / Basis textures in glTF files (public/basis, shipped with three.js).',
    texts: [],
  },
  {
    kind: 'bundled',
    name: 'meshoptimizer',
    version: '',
    license: 'MIT',
    repository: 'https://github.com/zeux/meshoptimizer',
    note: 'Decoder of meshopt-compressed glTF files (MeshoptDecoder, shipped with three.js).',
    texts: [],
  },
];

try {
  rustCrates();
} catch (e) {
  console.warn('[licenses] cargo metadata failed:', e.message.split('\n')[0]);
}
jsPackages();

const order = { origin: 0, bundled: 1, rust: 2, js: 3 };
const all = [...manual, ...packages].sort(
  (a, b) => order[a.kind] - order[b.kind] || a.name.localeCompare(b.name),
);
const missing = all.filter((p) => p.texts.length === 0).map((p) => `${p.name}@${p.version}`);
// No licence file in the sources: point at the canonical SPDX text.
for (const p of all) {
  if (p.texts.length) continue;
  const ids = String(p.license)
    .split(/\s+(?:OR|AND|WITH)\s+|[()/]/)
    .map((x) => x.trim())
    .filter(Boolean);
  const links = ids.map((id) => `${id}: https://spdx.org/licenses/${id}.html`).join('\n');
  p.texts = [addText(`Licensed under ${p.license}.\n\n${links}`)];
}
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ generated: new Date().toISOString(), packages: all, texts }));
const count = (kind) => packages.filter((p) => p.kind === kind).length;
console.log(
  `[licenses] ${all.length} components (${count('rust')} rust, ${count('js')} js), ${Object.keys(texts).length} distinct texts -> ${out}`,
);
if (missing.length)
  console.log(
    `[licenses] no licence file found for ${missing.length}: ${missing.slice(0, 12).join(', ')}${missing.length > 12 ? ', …' : ''}`,
  );
