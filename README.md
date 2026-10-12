# 3Dv

A simple, fast 3D model viewer for Windows, Linux and macOS. Tauri 2 (Rust) + Svelte 5 + Three.js.

**Formats:** GLB, GLTF, OBJ, FBX, STL, PLY, DAE, 3DS

**Features:**

- open models with a double-click or by dragging them into the window
- model thumbnails in Windows Explorer
- orbit, wireframe, grid, animations, model info

**Keys:** `O` options · `I` info · `Space` animation · `L` light · `R` reset view · `Ctrl+O` open

**Layout:** `app/` Svelte frontend · `core/` Rust backend · `lang/` UI strings · `thumbnail-provider/` Explorer thumbnail DLL (C++)

**Dev:** `npm install --prefix app`, then `app\node_modules\.bin\tauri dev --config core\tauri.conf.json` from the repo root

**Build:** `app\node_modules\.bin\tauri build --config core\tauri.conf.json` from the repo root
