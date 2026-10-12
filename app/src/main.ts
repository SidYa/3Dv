import { mount } from 'svelte'
import './app.css'
import App from './App.svelte'
import Thumbnail from './Thumbnail.svelte'
import { getCurrentWindow } from '@tauri-apps/api/window'

// The hidden thumbnailer window boots the same bundle and renders snapshots
// instead of the viewer. Detected by window label (Tauri) with a #thumbnail
// hash fallback for plain-browser runs.
function isThumbnailRoute(): boolean {
  if (window.location.hash === '#thumbnail') return true;
  try {
    return getCurrentWindow().label === 'thumbnailer';
  } catch {
    return false;
  }
}

const Root = isThumbnailRoute() ? Thumbnail : App;

const app = mount(Root, {
  target: document.getElementById('app')!,
})

export default app
