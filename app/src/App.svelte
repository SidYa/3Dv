<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import * as THREE from 'three';
  import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
  import { invoke } from '@tauri-apps/api/core';
  import { listen, type UnlistenFn } from '@tauri-apps/api/event';
  import { getCurrentWindow } from '@tauri-apps/api/window';
  import { loadModelFile, areTexturesReady } from './lib/loaders';
  import {
    BACKGROUND_COLOR,
    configureRenderer,
    addStudioLights,
    applyEnvironment,
    frameObject,
    tunePointSize,
    disposeObject,
    eachMaterial,
  } from './lib/scene';
  import enStrings from '../../lang/en.json';
  import ukStrings from '../../lang/uk.json';

  const strings: Record<string, Record<string, string>> = { en: enStrings, ua: ukStrings };

  const NORMALIZED_SIZE = 4;
  const AUTO_ROTATE_SPEED = 2.9;
  const MAX_PIXEL_RATIO = 2;
  const TEXTURE_SETTLE_MS = 10000;
  const SUPPORTED = ['.glb', '.gltf', '.obj', '.fbx', '.stl', '.ply', '.dae', '.3ds'];

  // --- ui state ---------------------------------------------------------
  let currentLang = $state('en');
  let t = $derived(strings[currentLang] ?? enStrings);
  let showDrop = $state(true);
  let dragging = $state(false);
  let showOptions = $state(false);
  let showInfo = $state(false);
  let loading = $state(false);
  let progress = $state('');
  let toast = $state('');
  let toastVisible = $state(false);
  let bgColor = $state('#1a1a1a');
  let autoRotate = $state(false);
  let wireframe = $state(false);
  let showGrid = $state(true);
  let singleWindow = $state(true);
  let info = $state({ size: '-', triangles: '-', vertices: '-', objects: '-', animations: '-' });

  // --- three state (not reactive) ----------------------------------------
  let container: HTMLDivElement | null = null;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let controls: OrbitControls;
  let timer: THREE.Timer;
  let gridHelper: THREE.GridHelper | null = null;
  let topDownLight: THREE.DirectionalLight;
  let mixer: THREE.AnimationMixer | null = null;
  let currentModel: THREE.Object3D | null = null;
  let isAnimating = false;
  let loadToken = 0;
  let needsRender = true;
  let texturesSettleUntil = 0;
  let rafId = 0;
  let toastTimer: ReturnType<typeof setTimeout> | null = null;
  let unlistens: UnlistenFn[] = [];
  let settings: Record<string, any> = {};

  function requestRender() {
    needsRender = true;
  }

  function showToast(message: string) {
    toast = message;
    toastVisible = true;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (toastVisible = false), 4000);
  }

  function formatBytes(bytes: number) {
    if (bytes >= 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    if (bytes >= 1024) return Math.round(bytes / 1024) + ' KB';
    return bytes + ' B';
  }

  function showProgress(event: any) {
    if (event?.lengthComputable && event.total > 0) {
      progress = Math.round((event.loaded / event.total) * 100) + '%';
    } else if (event?.loaded) {
      progress = formatBytes(event.loaded);
    }
  }

  function persist(key: string, value: any) {
    settings[key] = value;
    invoke('set_settings', { patch: { [key]: value } }).catch(() => {});
  }

  function isSupportedModel(filePath: string) {
    const dot = filePath.lastIndexOf('.');
    const ext = dot >= 0 ? filePath.slice(dot).toLowerCase() : '';
    return SUPPORTED.includes(ext);
  }

  // --- scene --------------------------------------------------------------
  function init(containerEl: HTMLDivElement) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(BACKGROUND_COLOR);
    timer = new THREE.Timer();
    timer.connect(document);

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1000);
    camera.position.set(0, 0, 5);

    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
    renderer.setSize(window.innerWidth, window.innerHeight);
    configureRenderer(renderer);
    containerEl.appendChild(renderer.domElement);

    addStudioLights(scene);
    applyEnvironment(scene, renderer);

    topDownLight = new THREE.DirectionalLight(0xffffff, 1);
    topDownLight.position.set(0, 10, 0);
    topDownLight.visible = false;
    scene.add(topDownLight);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.autoRotateSpeed = AUTO_ROTATE_SPEED;
    controls.addEventListener('change', requestRender);
  }

  function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, MAX_PIXEL_RATIO));
    renderer.setSize(window.innerWidth, window.innerHeight);
    requestRender();
  }

  function animate() {
    rafId = requestAnimationFrame(animate);
    timer.update();
    const delta = timer.getDelta();

    if (isAnimating && mixer) {
      mixer.update(delta);
      needsRender = true;
    }
    controls.autoRotate = autoRotate && !!currentModel;
    controls.update(delta);

    if (texturesSettleUntil) {
      needsRender = true;
      if (!currentModel || Date.now() > texturesSettleUntil || areTexturesReady(currentModel)) {
        texturesSettleUntil = 0;
      }
    }

    if (!needsRender) return;
    needsRender = false;
    renderer.render(scene, camera);
  }

  // --- model info -----------------------------------------------------------
  function countGeometry(object: THREE.Object3D) {
    let objects = 0;
    let vertices = 0;
    let triangles = 0;
    object.traverse((child: any) => {
      const geometry = child.geometry;
      if (!geometry || !geometry.getAttribute) return;
      const position = geometry.getAttribute('position');
      if (!position) return;
      objects++;
      vertices += position.count;
      if (child.isMesh) {
        triangles += (geometry.index ? geometry.index.count : position.count) / 3;
      }
    });
    return { objects, vertices, triangles: Math.floor(triangles) };
  }

  function groupDigits(value: number | string) {
    return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
  }

  function formatDimension(value: number) {
    if (!isFinite(value)) return '?';
    if (value >= 100) return value.toFixed(0);
    if (value >= 1) return value.toFixed(2);
    return value.toPrecision(3);
  }

  function updateInfoPanel(sourceSize: THREE.Vector3 | null, animationCount: number, object: THREE.Object3D) {
    const counts = countGeometry(object);
    info = {
      size: sourceSize
        ? `${formatDimension(sourceSize.x)} x ${formatDimension(sourceSize.y)} x ${formatDimension(sourceSize.z)}`
        : '-',
      triangles: groupDigits(counts.triangles),
      vertices: groupDigits(counts.vertices),
      objects: groupDigits(counts.objects),
      animations: String(animationCount),
    };
  }

  // --- model loading ----------------------------------------------------------
  function clearModel() {
    if (mixer) {
      mixer.stopAllAction();
      if (currentModel) mixer.uncacheRoot(currentModel);
      mixer = null;
    }
    isAnimating = false;
    if (currentModel) {
      scene.remove(currentModel);
      disposeObject(currentModel);
      currentModel = null;
    }
    requestRender();
  }

  function normalizeModel(model: THREE.Object3D) {
    model.position.set(0, 0, 0);
    model.scale.set(1, 1, 1);
    model.updateWorldMatrix(true, true);

    const box = new THREE.Box3().setFromObject(model);
    if (box.isEmpty()) return null;

    const sourceSize = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(sourceSize.x, sourceSize.y, sourceSize.z);
    if (!isFinite(maxDim) || maxDim <= 0) return null;

    model.scale.setScalar(NORMALIZED_SIZE / maxDim);
    model.updateWorldMatrix(true, true);

    const scaled = new THREE.Box3().setFromObject(model);
    model.position.sub(scaled.getCenter(new THREE.Vector3()));
    model.updateWorldMatrix(true, true);

    return { box: new THREE.Box3().setFromObject(model), sourceSize };
  }

  function rebuildGrid(modelBox: THREE.Box3 | null) {
    if (gridHelper) {
      scene.remove(gridHelper);
      gridHelper.geometry.dispose();
      (gridHelper.material as THREE.Material).dispose();
    }
    gridHelper = new THREE.GridHelper(NORMALIZED_SIZE * 2.5, 20, 0x444444, 0x222222);
    gridHelper.visible = showGrid;
    gridHelper.position.y = modelBox ? modelBox.min.y : 0;
    scene.add(gridHelper);
  }

  function fitView() {
    frameObject(camera, currentModel!, controls.target);
    controls.update();
    controls.saveState();
  }

  async function loadModel(filePath: string) {
    if (!isSupportedModel(filePath)) return;
    const token = ++loadToken;

    clearModel();
    showDrop = false;
    loading = true;
    progress = '';

    let loaded: any;
    try {
      loaded = await loadModelFile(filePath, renderer, (event: any) => {
        if (token === loadToken) showProgress(event);
      });
    } catch (e) {
      console.error('Load failed:', e);
      if (token !== loadToken) return;
      loading = false;
      showToast(t.fail_load);
      showDrop = true;
      return;
    }

    if (token !== loadToken) {
      disposeObject(loaded.object);
      return;
    }

    try {
      currentModel = loaded.object;

      if (loaded.animations.length > 0) {
        mixer = new THREE.AnimationMixer(currentModel!);
        loaded.animations.forEach((clip: THREE.AnimationClip) => mixer!.clipAction(clip).play());
      }

      const anisotropy = renderer.capabilities.getMaxAnisotropy();
      eachMaterial(currentModel!, (material: any) => {
        material.wireframe = wireframe;
        if (material.map) material.map.anisotropy = anisotropy;
      });

      scene.add(currentModel!);
      const normalized = normalizeModel(currentModel!);
      if (!normalized) {
        clearModel();
        loading = false;
        showToast(t.fail_empty);
        showDrop = true;
        return;
      }

      rebuildGrid(normalized.box);
      tunePointSize(currentModel!, normalized.box.getBoundingSphere(new THREE.Sphere()).radius);
      fitView();
      updateInfoPanel(normalized.sourceSize, loaded.animations.length, currentModel!);

      loading = false;
      setWindowTitle(filePath);
      texturesSettleUntil = Date.now() + TEXTURE_SETTLE_MS;
      requestRender();
      setTimeout(() => {
        invoke('generate_thumbnail', { filePath }).catch(() => {});
      }, 1000);
    } catch (e) {
      console.error('Setup failed:', e);
      clearModel();
      loading = false;
      showToast(t.fail_load);
      showDrop = true;
    }
  }

  function setWindowTitle(filePath: string) {
    const name = filePath.split(/[\\/]/).pop();
    document.title = name ? `${name} - 3Dv` : '3Dv';
  }

  async function openFileDialog() {
    try {
      const filePath = await invoke<string | null>('show_open_dialog');
      if (filePath) loadModel(filePath);
    } catch (e) {
      console.error('Dialog failed:', e);
    }
  }

  async function setAsDefault() {
    try {
      const ok = await invoke<boolean>('set_as_default');
      showToast(ok ? t.success_assoc : t.fail_assoc);
    } catch (_) {
      showToast(t.fail_assoc);
    }
  }

  function onKeyDown(event: KeyboardEvent) {
    if (event.code === 'KeyO' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      openFileDialog();
      return;
    }
    const tag = (event.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || tag === 'BUTTON') {
      if (event.code === 'Escape') showOptions = false;
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) return;

    if (event.code === 'KeyO') showOptions = !showOptions;
    if (event.code === 'KeyI') showInfo = !showInfo;
    if (event.code === 'KeyL') {
      topDownLight.visible = !topDownLight.visible;
      requestRender();
    }
    if (event.code === 'KeyR') controls.reset();
    if (event.code === 'Escape') showOptions = false;
    if (event.code === 'Space') {
      event.preventDefault();
      isAnimating = !isAnimating;
    }
  }

  function onDragOver(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    showDrop = true;
    dragging = true;
  }

  function onDragLeave(e: DragEvent) {
    e.preventDefault();
    e.stopPropagation();
    const leftWindow =
      e.clientX <= 0 || e.clientY <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight;
    if (!leftWindow) return;
    dragging = false;
    if (currentModel) showDrop = false;
  }

  function onDrop(e: DragEvent) {
    // Paths arrive via the Tauri drag-drop event below; this only stops
    // the WebView from navigating away.
    e.preventDefault();
    e.stopPropagation();
    dragging = false;
    if (currentModel) showDrop = false;
  }

  let ready = false;
  let queuedFile: string | null = null;

  function openFile(filePath: string) {
    if (ready) loadModel(filePath);
    else queuedFile = filePath;
  }

  onMount(async () => {
    init(container!);

    try {
      settings = (await invoke('get_settings')) as Record<string, any> ?? {};
    } catch (_) {
      settings = {};
    }
    currentLang = settings.lang || 'en';
    singleWindow = settings.singleWindow !== false;
    if (settings.bgColor) {
      bgColor = settings.bgColor;
      (scene.background as THREE.Color).set(settings.bgColor);
    }
    autoRotate = settings.autoRotate === true;
    wireframe = settings.wireframe === true;
    showGrid = settings.showGrid !== false;

    unlistens.push(await listen<string>('open-file', (e) => openFile(e.payload)));

    try {
      const win = getCurrentWindow();
      unlistens.push(
        await win.onDragDropEvent((event) => {
          if (event.payload.type === 'drop' && event.payload.paths.length > 0) {
            dragging = false;
            openFile(event.payload.paths[0]);
          }
        }),
      );
    } catch (_) {
      // Not running under Tauri (browser dev): HTML5 drop keeps working via
      // file pickers only.
    }

    try {
      const initial = await invoke<string | null>('get_initial_file');
      if (initial) queuedFile = initial;
    } catch (_) {}

    window.addEventListener('resize', onWindowResize);
    window.addEventListener('dragover', onDragOver as EventListener);
    window.addEventListener('dragleave', onDragLeave as EventListener);
    window.addEventListener('drop', onDrop as EventListener);
    document.addEventListener('keydown', onKeyDown);

    animate();
    ready = true;
    if (queuedFile) {
      const f = queuedFile;
      queuedFile = null;
      loadModel(f);
    }
  });

  onDestroy(() => {
    cancelAnimationFrame(rafId);
    unlistens.forEach((u) => u());
    window.removeEventListener('resize', onWindowResize);
    window.removeEventListener('dragover', onDragOver as EventListener);
    window.removeEventListener('dragleave', onDragLeave as EventListener);
    window.removeEventListener('drop', onDrop as EventListener);
    document.removeEventListener('keydown', onKeyDown);
    if (toastTimer) clearTimeout(toastTimer);
  });
</script>

<div id="drop-zone" class:hidden={!showDrop} class:drag-over={dragging} on:click={openFileDialog} role="button" tabindex="0" on:keydown={(e) => e.key === 'Enter' && openFileDialog()}>
  <div id="drop-message">
    <h1>3Dv</h1>
    <p>{t.drag_message}</p>
    <p>{t.click_message}</p>
  </div>
</div>
<div id="canvas-container" bind:this={container}></div>

<div id="options-overlay" class:hidden={!showOptions} on:click={(e) => e.target === e.currentTarget && (showOptions = false)} role="presentation">
  <div class="options-content">
    <h2>{t.options_title}</h2>

    <div class="option-item">
      <label for="lang-select">{t.language}</label>
      <select
        id="lang-select"
        bind:value={currentLang}
        on:change={() => persist('lang', currentLang)}
      >
        <option value="en">English</option>
        <option value="ua">??????????</option>
      </select>
    </div>

    <div class="option-item">
      <label for="bg-color">{t.bg_color}</label>
      <input
        type="color"
        id="bg-color"
        bind:value={bgColor}
        on:input={() => {
          (scene.background as THREE.Color).set(bgColor);
          requestRender();
          persist('bgColor', bgColor);
        }}
      />
    </div>

    <div class="option-item">
      <label for="auto-rotate">{t.auto_rotate}</label>
      <input type="checkbox" id="auto-rotate" bind:checked={autoRotate} on:change={() => persist('autoRotate', autoRotate)} />
    </div>

    <div class="option-item">
      <label for="wireframe">{t.wireframe}</label>
      <input
        type="checkbox"
        id="wireframe"
        bind:checked={wireframe}
        on:change={() => {
          persist('wireframe', wireframe);
          if (currentModel) {
            eachMaterial(currentModel, (m: any) => (m.wireframe = wireframe));
            requestRender();
          }
        }}
      />
    </div>

    <div class="option-item">
      <label for="show-grid">{t.show_grid}</label>
      <input
        type="checkbox"
        id="show-grid"
        bind:checked={showGrid}
        on:change={() => {
          if (gridHelper) gridHelper.visible = showGrid;
          requestRender();
          persist('showGrid', showGrid);
        }}
      />
    </div>

    <div class="option-item">
      <label for="single-window">{t.single_window}</label>
      <input type="checkbox" id="single-window" bind:checked={singleWindow} on:change={() => persist('singleWindow', singleWindow)} />
    </div>

    <div class="option-item">
      <button id="reset-view" on:click={() => controls.reset()}>{t.reset_view}</button>
    </div>

    <div class="option-item">
      <button id="set-default" on:click={setAsDefault}>{t.set_default}</button>
    </div>

    <p class="hint">{t.hotkeys_hint}</p>
  </div>
</div>

<div id="loading-overlay" class:hidden={!loading}>
  <div class="loading-content">
    <p>{t.loading}</p>
    <p id="loading-progress">{progress}</p>
  </div>
</div>

<div id="info-panel" class:hidden={!showInfo}>
  <div class="info-row"><span>{t.info_size}</span><b>{info.size}</b></div>
  <div class="info-row"><span>{t.info_triangles}</span><b>{info.triangles}</b></div>
  <div class="info-row"><span>{t.info_vertices}</span><b>{info.vertices}</b></div>
  <div class="info-row"><span>{t.info_objects}</span><b>{info.objects}</b></div>
  <div class="info-row"><span>{t.info_animations}</span><b>{info.animations}</b></div>
</div>

<div id="toast" class:hidden={!toastVisible}>{toast}</div>
