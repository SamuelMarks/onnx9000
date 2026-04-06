/* eslint-disable */
// @ts-nocheck
import '@onnx9000/webnn-polyfill';
import * as tf from '@onnx9000/tfjs-shim';
import * as transformers from '@onnx9000/transformers';
import * as diffusers from '@onnx9000/diffusers';
import { globalEventBus } from './core/EventBus';
import './css/reset.css';
import './css/theme.css';
import './css/layout.css';
import { SplitPane } from './components/SplitPane';
import { LHSContainer } from './components/LHSContainer';
import { RHSContainer } from './components/RHSContainer';
import { BottomContainer } from './components/BottomContainer';
import { WasmOverlay } from './components/WasmOverlay';
import { Breadcrumbs } from './components/Breadcrumbs';
import { WasmManager, WasmState } from './core/WasmManager';

// Expose these ecosystem packages to the window for console usage and to ensure they are bundled.
(window as ReturnType<typeof JSON.parse>).tf = tf;
(window as ReturnType<typeof JSON.parse>).transformers = transformers;
(window as ReturnType<typeof JSON.parse>).diffusers = diffusers;

/**
 * Main entry point for the Sphinx Demo UI.
 */

console.log('Sphinx Demo UI loaded successfully');

export function initDemoUI(containerId: string) {
  const container = document.getElementById(containerId);
  if (!container) {
    console.error(`Container with id "${containerId}" not found.`);
    return;
  }

  // Clear any existing content (e.g., from mock server)
  container.innerHTML = '';

  // Setup root wrapper
  const rootElement = document.createElement('div');
  const breadcrumbs = new Breadcrumbs();
  (window as object).__BREADCRUMBS_INSTANCE__ = breadcrumbs;
  breadcrumbs.mount(container);
  rootElement.className = 'demo-ui-root demo-layout-container';
  container.appendChild(rootElement);

  // 1. Create the Top Split (Horizontal: LHS | RHS)
  const topSplit = new SplitPane({
    className: 'demo-main-split',
    orientation: 'horizontal',
    initialSplitRatio: 0.5,
    minSize: 200,
    storageKey: 'onnx9000-demo-horizontal-split'
  });

  // 2. Create the Main Split (Vertical: Top Split | Bottom Pane)
  const mainSplit = new SplitPane({
    orientation: 'vertical',
    initialSplitRatio: 0.5, // 50% top, 50% bottom
    minSize: 150,
    storageKey: 'onnx9000-demo-vertical-split'
  });

  // 3. Mount containers
  mainSplit.mount(rootElement);

  // Destructure panes
  const { pane1: topPane, pane2: bottomPane } = mainSplit.getPanes();
  const { pane1: lhsPane, pane2: rhsPane } = topSplit.getPanes();

  // 4. Inject sub-panes
  topSplit.mount(topPane);

  const lhs = new LHSContainer();
  lhs.mount(lhsPane);

  const rhs = new RHSContainer();
  rhs.mount(rhsPane);

  const bottom = new BottomContainer();
  bottom.mount(bottomPane);

  // 5. Mount WASM Overlay if not loaded
  if (WasmManager.getInstance().state !== WasmState.LOADED) {
    const overlay = new WasmOverlay();
    overlay.mount(rootElement);
  }
}

// Auto-init if container exists (useful for dev mock server)
const container = document.getElementById('interactive-demo-container');
if (container) {
  initDemoUI('interactive-demo-container');
}

// Export to window for E2E testing
(window as object).__EVENT_BUS__ = globalEventBus;

// Expose breadcrumbs directly for E2E testing of this component until Phase 13 integrates it naturally

(window as object).__ADD_BREADCRUMB_TEST__ = (id: string, description: string) => {
  globalEventBus.emit('PIPELINE_STEP_ADDED', { id, description, state: {} });
};

import { TensorInputModal } from './components/TensorInputModal';
(window as object).__TensorInputModal__ = TensorInputModal;

(window as object).__OPEN_MODAL__ = (inputs: object[]) => {
  const modal = new TensorInputModal();
  modal.mount(document.body);
  modal.show(inputs);
};

// Theme synchronization via MutationObserver to sync Monaco to Sphinx body classes
const observeTheme = () => {
  const isDark = () =>
    document.documentElement.getAttribute('data-theme') === 'dark' ||
    document.body.classList.contains('dark');
  let currentTheme = isDark() ? 'vs-dark' : 'vs-light';
  globalEventBus.emit('THEME_CHANGED', currentTheme);

  const observer = new MutationObserver(() => {
    const newTheme = isDark() ? 'vs-dark' : 'vs-light';
    if (newTheme !== currentTheme) {
      currentTheme = newTheme;
      globalEventBus.emit('THEME_CHANGED', currentTheme);
    }
  });

  observer.observe(document.documentElement, {
    attributes: true,
    attributeFilter: ['data-theme', 'class']
  });
  observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
};

observeTheme();
