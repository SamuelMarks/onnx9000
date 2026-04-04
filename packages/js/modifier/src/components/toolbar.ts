import { Graph } from '@onnx9000/core';
import { ModifierUtilities } from './utilities.js';

export interface ToolbarConfig {
  onCleanGraph: () => void;
  onMakeDynamic: () => void;
  onStripInitializers: () => void;
  onFixMixedPrecision: () => void;
  onRemoveTrainingNodes: () => void;
  onFoldConstants: () => void;
  onExtractWeights: () => void;
  onSanitizeNames: () => void;
  onValidateOpset: () => void;
  onValidateGraph: () => void;
  onAutoFix: () => void;
  onDeduplicateConstants: () => void;
  onExportStats: () => void;
  onExportGraphJSON: () => void;
  onFeedback: () => void;
  onToggleStrict: (enabled: boolean) => void;
  onSaveSession: () => void;
  onExportModel: () => void;
}

export class Toolbar {
  container: HTMLElement;
  config: ToolbarConfig;

  constructor(container: HTMLElement, config: ToolbarConfig) {
    this.container = container;
    this.config = config;
    this.render();
  }

  render() {
    this.container.innerHTML = '';
    this.container.style.display = 'flex';
    this.container.style.gap = '8px';
    this.container.style.padding = '8px';
    this.container.style.borderBottom = '1px solid #dee2e6';
    this.container.style.backgroundColor = '#f1f3f5';

    const btnClean = document.createElement('button');
    btnClean.textContent = 'Clean Graph';
    btnClean.onclick = this.config.onCleanGraph;

    const btnDynamic = document.createElement('button');
    btnDynamic.textContent = 'Make Dynamic';
    btnDynamic.onclick = this.config.onMakeDynamic;

    const btnStrip = document.createElement('button');
    btnStrip.textContent = 'Strip Initializers';
    btnStrip.onclick = this.config.onStripInitializers;

    this.container.appendChild(btnClean);
    this.container.appendChild(btnDynamic);
    this.container.appendChild(btnStrip);

    const btnFix = document.createElement('button');
    btnFix.textContent = 'Fix Precision';
    btnFix.onclick = this.config.onFixMixedPrecision;

    const btnTrain = document.createElement('button');
    btnTrain.textContent = 'Remove Training';
    btnTrain.onclick = this.config.onRemoveTrainingNodes;

    const btnFold = document.createElement('button');
    btnFold.textContent = 'Fold Constants';
    btnFold.onclick = this.config.onFoldConstants;

    const btnExtract = document.createElement('button');
    btnExtract.textContent = 'Extract Weights';
    btnExtract.onclick = this.config.onExtractWeights;

    const btnSanitize = document.createElement('button');
    btnSanitize.textContent = 'Sanitize Names';
    btnSanitize.onclick = this.config.onSanitizeNames;

    this.container.appendChild(btnFix);
    this.container.appendChild(btnTrain);
    this.container.appendChild(btnFold);
    this.container.appendChild(btnExtract);
    this.container.appendChild(btnSanitize);

    const btnValidateOpset = document.createElement('button');
    btnValidateOpset.textContent = 'Validate Opset';
    btnValidateOpset.onclick = this.config.onValidateOpset;
    this.container.appendChild(btnValidateOpset);

    const btnValidateGraph = document.createElement('button');
    btnValidateGraph.textContent = 'Validate Graph';
    btnValidateGraph.onclick = this.config.onValidateGraph;
    this.container.appendChild(btnValidateGraph);

    const btnAutoFix = document.createElement('button');
    btnAutoFix.textContent = 'Auto-Fix Missing';
    btnAutoFix.onclick = this.config.onAutoFix;
    this.container.appendChild(btnAutoFix);

    const btnDedup = document.createElement('button');
    btnDedup.textContent = 'Deduplicate Constants';
    btnDedup.onclick = this.config.onDeduplicateConstants;
    this.container.appendChild(btnDedup);

    const btnExportStats = document.createElement('button');
    btnExportStats.textContent = 'Export Stats (CSV)';
    btnExportStats.onclick = this.config.onExportStats;
    this.container.appendChild(btnExportStats);

    const btnSaveSession = document.createElement('button');
    btnSaveSession.textContent = 'Save Session';
    btnSaveSession.onclick = this.config.onSaveSession;
    this.container.appendChild(btnSaveSession);

    const btnExportModel = document.createElement('button');
    btnExportModel.textContent = 'Export Model';
    btnExportModel.onclick = this.config.onExportModel;
    this.container.appendChild(btnExportModel);

    const btnExportJSON = document.createElement('button');
    btnExportJSON.textContent = 'View Raw JSON';
    btnExportJSON.onclick = this.config.onExportGraphJSON;
    this.container.appendChild(btnExportJSON);

    const btnFeedback = document.createElement('button');
    btnFeedback.textContent = 'Report Bug / Feedback';
    btnFeedback.onclick = this.config.onFeedback;
    btnFeedback.style.marginLeft = 'auto';
    this.container.appendChild(btnFeedback);

    const lblStrict = document.createElement('label');
    lblStrict.style.marginLeft = '8px';
    lblStrict.style.display = 'flex';
    lblStrict.style.alignItems = 'center';
    lblStrict.style.fontSize = '12px';
    lblStrict.innerHTML = '<input type="checkbox" style="margin-right: 4px;" /> Strict Mode';

    const cbStrict = lblStrict.querySelector('input') as HTMLInputElement;
    cbStrict.onchange = () => {
      /* v8 ignore start */
      this.config.onToggleStrict(cbStrict.checked);
    };
    /* v8 ignore stop */

    this.container.appendChild(lblStrict);
  }
}
