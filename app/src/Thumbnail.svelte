<script lang="ts">
  // Offscreen renderer behind every cached thumbnail. Lives in the hidden
  // "thumbnailer" window (route #thumbnail), answers one request at a time
  // and reports failures instead of storing an empty picture.
  // Port of the old thumbnail-renderer.js; the queue lives in Rust now.
  import { onMount } from 'svelte';
  import * as THREE from 'three';
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { loadModelFile, areTexturesReady } from './lib/loaders';
  import {
    BACKGROUND_COLOR,
    configureRenderer,
    addStudioLights,
    applyEnvironment,
    frameObject,
    hasRenderableGeometry,
    tunePointSize,
    disposeObject,
  } from './lib/scene';

  const THUMB_SIZE = 256;
  // Textures load asynchronously; capturing before they arrive gives an
  // untextured, usually far too dark picture.
  const TEXTURE_WAIT_MS = 6000;
  const PROBE_TOLERANCE = 5;
  // Every 4th pixel in each direction is plenty to tell a picture from a
  // flat colour.
  const PROBE_STEP = 4;

  let root: HTMLDivElement | null = null;

  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let probePixels: Uint8Array;

  let pending: { id: number; object: THREE.Object3D; deadline: number } | null = null;
  let latestRequestId = 0;
  let waitTimer: ReturnType<typeof setTimeout> | null = null;

  const LOAD_TIMEOUT_MS = 20000;

  function initRenderer() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(BACKGROUND_COLOR);

    camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);

    renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    renderer.setSize(THUMB_SIZE, THUMB_SIZE);
    renderer.setPixelRatio(1);
    configureRenderer(renderer);
    root!.appendChild(renderer.domElement);

    addStudioLights(scene);
    applyEnvironment(scene, renderer);

    probePixels = new Uint8Array(THUMB_SIZE * THUMB_SIZE * 4);
  }

  function respond(id: number, png: number[] | null, reason: string | null) {
    invoke('submit_thumbnail', { id, png, reason }).catch((e) =>
      console.error('submit failed:', e),
    );
  }

  // Synchronous on purpose. canvas.toBlob() schedules its encoding for idle
  // time, which a hidden window never has.
  function encodePng(canvas: HTMLCanvasElement): number[] | null {
    const base64 = canvas.toDataURL('image/png').split(',')[1];
    if (!base64) return null;
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return Array.from(bytes);
  }

  function releasePending(reason: string) {
    if (!pending) return;
    scene.remove(pending.object);
    disposeObject(pending.object);
    const finished = pending;
    pending = null;
    if (waitTimer) {
      clearTimeout(waitTimer);
      waitTimer = null;
    }
    if (reason) respond(finished.id, null, reason);
  }

  // A frame of one flat colour means the model was loaded but nothing ended
  // up in view. Reporting that as a failure keeps a useless grey square out
  // of the cache, so the next attempt can do better.
  function isFrameEmpty(): boolean {
    const gl = renderer.getContext();
    gl.readPixels(0, 0, THUMB_SIZE, THUMB_SIZE, gl.RGBA, gl.UNSIGNED_BYTE, probePixels);
    const data = probePixels;
    const [r, g, b] = data;

    for (let y = 0; y < THUMB_SIZE; y += PROBE_STEP) {
      for (let x = 0; x < THUMB_SIZE; x += PROBE_STEP) {
        const i = (y * THUMB_SIZE + x) * 4;
        if (Math.abs(data[i] - r) > PROBE_TOLERANCE) return false;
        if (Math.abs(data[i + 1] - g) > PROBE_TOLERANCE) return false;
        if (Math.abs(data[i + 2] - b) > PROBE_TOLERANCE) return false;
      }
    }
    return true;
  }

  function capture() {
    const request = pending!;
    renderer.render(scene, camera);

    const empty = isFrameEmpty();
    const png = empty ? null : encodePng(renderer.domElement);

    scene.remove(request.object);
    disposeObject(request.object);
    pending = null;

    respond(request.id, png, empty ? 'empty-frame' : png ? null : 'encode-failed');
  }

  // setTimeout-driven, not requestAnimationFrame: a hidden WebView may
  // throttle rAF indefinitely, while short-term timers still fire.
  function waitLoop() {
    if (!pending) return;
    if (!areTexturesReady(pending.object) && Date.now() < pending.deadline) {
      renderer.render(scene, camera);
      waitTimer = setTimeout(waitLoop, 50);
      return;
    }
    capture();
  }

  async function handleRequest(request: { id: number; filePath: string }) {
    latestRequestId = request.id;
    releasePending('superseded');

    let loaded: any;
    try {
      // Never hang silently: a fetch the protocol layer drops would
      // otherwise leave the Rust side waiting for the full 25s timeout.
      loaded = await Promise.race([
        loadModelFile(request.filePath, renderer),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error('load-timeout')), LOAD_TIMEOUT_MS),
        ),
      ]);
    } catch (err) {
      console.error('Thumbnail load failed:', err);
      if (latestRequestId === request.id) {
        respond(request.id, null, 'load-failed');
      }
      return;
    }

    if (latestRequestId !== request.id) {
      disposeObject(loaded.object);
      return;
    }

    if (!hasRenderableGeometry(loaded.object)) {
      disposeObject(loaded.object);
      respond(request.id, null, 'no-geometry');
      return;
    }

    scene.add(loaded.object);
    const sphere = frameObject(camera, loaded.object);
    if (!sphere) {
      scene.remove(loaded.object);
      disposeObject(loaded.object);
      respond(request.id, null, 'no-extent');
      return;
    }
    tunePointSize(loaded.object, sphere.radius);

    pending = { id: request.id, object: loaded.object, deadline: Date.now() + TEXTURE_WAIT_MS };
    waitTimer = setTimeout(waitLoop, 50);
  }

  onMount(async () => {
    initRenderer();
    await listen<{ id: number; filePath: string }>(
      'generate-thumbnail-render',
      (e) => {
        if (e.payload && typeof e.payload.filePath === 'string') handleRequest(e.payload);
      },
    );
    invoke('thumbnail_ready').catch((e) => console.error('ready failed:', e));
  });
</script>

<div id="thumb-root" bind:this={root}></div>

<style>
  #thumb-root {
    position: absolute;
  }
</style>
