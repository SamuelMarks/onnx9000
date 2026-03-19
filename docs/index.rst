.. onnx9000 documentation master file.

ONNX9000 Documentation
======================

ONNX9000 is a polyglot ONNX ecosystem.

Demo
====
.. raw:: html

   <!-- Load Monaco Editor via CDN -->
   <script>var require = { paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' } };</script>
   <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/loader.min.js"></script>

   <link rel="stylesheet" href="_static/demo.css">
   <script src="_static/demo.js" defer></script>

   <div class="demo-container">
          <div class="demo-tabs" role="tablist" aria-label="Demo Applications">
       <button class="demo-tab active" id="tab-genai" role="tab" aria-selected="true" aria-controls="genai-demo" data-target="genai-demo" tabindex="0">WASM GenAI</button>
       <button class="demo-tab" id="tab-converter" role="tab" aria-selected="false" aria-controls="converter-demo" data-target="converter-demo" tabindex="-1">Model Converter</button>
       <button class="demo-tab" id="tab-netron" role="tab" aria-selected="false" aria-controls="netron-demo" data-target="netron-demo" tabindex="-1">Netron UI</button>
       <button class="demo-tab" id="tab-optimum" role="tab" aria-selected="false" aria-controls="optimum-demo" data-target="optimum-demo" tabindex="-1">Optimum UI</button>
       <button class="demo-tab" id="tab-tvm" role="tab" aria-selected="false" aria-controls="tvm-demo" data-target="tvm-demo" tabindex="-1">Apache TVM</button>
     </div>


          <div id="tvm-demo" class="demo-panel" role="tabpanel" aria-labelledby="tab-tvm" tabindex="0">
       <div class="tvm-container" style="display: flex; flex-direction: column; gap: 1rem;">
         <div class="tvm-header" style="border-bottom: 1px solid var(--color-background-border); padding-bottom: 1rem;">
           <h3>Apache TVM (WASM-Native) Compiler Demo</h3>
           <p>Compile and execute ML models directly in the browser via WASM-Native MLIR and TVM.</p>
           <div id="tvm-status" class="status-badge status-loading">Initializing TVM WASM Runtime...</div>
         </div>
         <div class="tvm-content" style="display: flex; gap: 1rem; flex-wrap: wrap;">
           <div class="tvm-input-area" style="flex: 1; min-width: 300px; display: flex; flex-direction: column; gap: 0.5rem;">
             <h4 style="margin: 0; font-size: 0.9rem; color: var(--color-foreground-muted);">Model Specification (Relay IR / JSON)</h4>
             <textarea id="tvm-input-model" style="width: 100%; height: 200px; font-family: monospace; padding: 0.5rem; border: 1px solid var(--color-background-border); border-radius: 4px; background: var(--color-background-primary); color: var(--color-foreground-primary); resize: vertical;" placeholder="Enter model JSON or Relay IR..."></textarea>
             <button id="tvm-compile-btn" class="action-btn" disabled>Compile to WASM &rarr;</button>
           </div>
           <div class="tvm-output-area" style="flex: 1; min-width: 300px; display: flex; flex-direction: column; gap: 0.5rem;">
             <h4 style="margin: 0; font-size: 0.9rem; color: var(--color-foreground-muted);">Compilation Output / Execution Log</h4>
             <div id="tvm-output-log" class="chat-output" style="height: 200px; overflow-y: auto; border: 1px solid var(--color-background-border); border-radius: 4px; background: var(--color-background-primary); padding: 10px; font-family: monospace; font-size: 0.85rem; display: flex; flex-direction: column; gap: 0.5rem;" aria-live="polite">
               <div class="system-message">Ready to initialize the WebAssembly execution environment for TVM.</div>
             </div>
             <button id="tvm-run-btn" class="action-btn" disabled>Execute WASM Kernel &rarr;</button>
           </div>
         </div>
       </div>
       <script type="module" src="_static/tvm.js"></script>
     </div>

     <div id="genai-demo" class="demo-panel active" role="tabpanel" aria-labelledby="tab-genai" tabindex="0">
       <div class="genai-container">
         <div class="genai-header">
           <h3>ONNX GenAI (WASM-First) Local LLM Demo</h3>
           <p>Running entirely in your browser using the newly implemented ONNX GenAI WASM backend.</p>
           <div id="genai-status" class="status-badge status-loading">Initializing Engine...</div>
         </div>
         <div class="genai-chat">
           <div id="genai-output" class="chat-output" aria-live="polite">
             <div class="chat-message system-message">Ready to initialize the WebAssembly execution environment and download the quantized LLM.</div>
           </div>
           <div class="chat-input-area">
             <input type="text" id="genai-input" class="chat-input" placeholder="Ask me anything..." aria-label="Chat input" disabled>
             <button id="genai-btn" class="action-btn" disabled>Send &rarr;</button>
           </div>
         </div>
       </div>
       <script type="module" src="_static/genai.js"></script>
     </div>

     <div id="converter-demo" class="demo-panel" role="tabpanel" aria-labelledby="tab-converter" tabindex="0">
       <div class="converter-toolbar">
         <div class="toolbar-group">
           <label for="input-lang">Input:</label>
           <select id="input-lang" aria-label="Select input source code language">
             <option value="pytorch">PyTorch</option>
             <option value="onnxscript">ONNX Script</option>
             <option value="tensorflow" disabled>TensorFlow (COMING SOON!)</option>
           </select>
         </div>
         <button id="convert-btn" class="action-btn" aria-label="Convert source code to target format">Convert &rarr;</button>
         <div class="toolbar-group">
           <label for="output-lang">Output:</label>
           <select id="output-lang" aria-label="Select target converted code format">
             <option value="onnx">ONNX Format</option>
             <option value="mlir">MLIR (Web-MLIR)</option>
             <option value="c">C (Backend)</option>
             <option value="keras" disabled>Keras Model (COMING SOON!)</option>
             <option value="cpp" disabled>C++ (COMING SOON!)</option>
           </select>
         </div>
       </div>
       
       <div class="converter-editor">
         <div class="editor-pane">
           <div class="pane-header" id="header-source">Source Code</div>
           <div id="input-editor" class="monaco-container" aria-labelledby="header-source"></div>
         </div>
         <div class="editor-pane">
           <div class="pane-header" id="header-converted">Converted Code</div>
           <div id="output-editor" class="monaco-container" aria-labelledby="header-converted"></div>
         </div>
       </div>

       <!-- Bottom Pane -->
       <div class="bottom-pane-container">
         <div class="bottom-tabs" role="tablist" aria-label="Output Views">
            <button class="bottom-tab active" id="tab-visualize" role="tab" aria-selected="true" aria-controls="panel-visualize" data-target="panel-visualize" tabindex="0">Visualize</button>
            <button class="bottom-tab" id="tab-log" role="tab" aria-selected="false" aria-controls="panel-log" data-target="panel-log" tabindex="-1">Log</button>
         </div>
         
         <div id="panel-visualize" class="bottom-panel active" role="tabpanel" aria-labelledby="tab-visualize" tabindex="0">
           <div class="canvas-mock small-mock">Model Graph Ready for Visualization</div>
         </div>
         
         <div id="panel-log" class="bottom-panel" role="tabpanel" aria-labelledby="tab-log" tabindex="0">
           <div class="log-output" id="conversion-log" aria-live="polite" aria-atomic="false">System ready. Waiting for conversion request...</div>
         </div>
       </div>
     </div>

     <div id="netron-demo" class="demo-panel" role="tabpanel" aria-labelledby="tab-netron" tabindex="0">
       <div class="placeholder-box">
         <h3>Netron UI Visualizer</h3>
         <p>Interactive DAG visualization of ONNX models.</p>
         <!-- Netron UI bundle insertion point -->
         <div class="canvas-mock">Model Graph Rendering Area</div>
       </div>
     </div>

     <div id="optimum-demo" class="demo-panel" role="tabpanel" aria-labelledby="tab-optimum" tabindex="0">
       <div class="placeholder-box">
         <h3>Optimum UI</h3>
         <p>Hardware-accelerated web inference optimizations.</p>
         <!-- Optimum UI bundle insertion point -->
         <div class="canvas-mock">WebGPU Pipeline Visualizer Area</div>
       </div>
     </div>
   </div>

Architecture
============

.. mermaid::

   flowchart LR
       %% Theme initialization
       %%{
         init: {
           'theme': 'base',
           'themeVariables': {
             'fontFamily': '"Google Sans Normal", "Google Sans", sans-serif',
             'lineColor': '#20344b',
             'clusterBkg': '#ffffff',
             'clusterBorder': '#20344b'
           }
         }
       }%%

       classDef default fill:#ffffff,stroke:#20344b,stroke-width:2px,color:#20344b,font-family:"Google Sans Normal",font-weight:400;
       classDef centerNode fill:#20344b,stroke:#20344b,stroke-width:2px,color:#ffffff,font-family:"Google Sans Medium",font-weight:500,font-size:20px;
       classDef import fill:#5cdb6d,stroke:#34a853,stroke-width:2px,color:#20344b,font-family:"Google Sans Normal",font-weight:400;
       classDef export fill:#57caff,stroke:#4285f4,stroke-width:2px,color:#20344b,font-family:"Google Sans Normal",font-weight:400;
       classDef process fill:#ffd427,stroke:#f9ab00,stroke-width:2px,color:#20344b,font-family:"Google Sans Normal",font-weight:400;
       classDef soon fill:#ff7daf,stroke:#ea4335,stroke-width:2px,stroke-dasharray: 5 5,color:#20344b,font-family:"Google Sans Normal",font-weight:400;

       IR(("ONNX9000 IR")):::centerNode

       subgraph Imports ["<span style='font-family: Roboto Mono Normal, monospace; font-size: 16px; font-weight: normal; color: #20344b;'>Imports</span>"]
           direction TB
           I_ONNX("ONNX"):::import
           I_PT("PyTorch"):::import
           I_OS("ONNX Script"):::import
           I_TF("TensorFlow (Soon)"):::soon
       end

       subgraph Exports ["<span style='font-family: Roboto Mono Normal, monospace; font-size: 16px; font-weight: normal; color: #20344b;'>Exports</span>"]
           direction TB
           E_ONNX("ONNX"):::export
           E_MLIR("MLIR / Web-MLIR"):::export
           E_C("C Backend"):::export
           E_KERAS("Keras (Soon)"):::soon
           E_CPP("C++ (Soon)"):::soon
       end

       subgraph Processing ["<span style='font-family: Roboto Mono Normal, monospace; font-size: 16px; font-weight: normal; color: #20344b;'>Processing</span>"]
           direction TB
           P_SIMP("Simplify Models"):::process
           P_OPT("Optimize Models"):::process
           P_VIS("Visualize Models"):::process
       end

       I_ONNX --> IR
       I_PT --> IR
       I_OS --> IR
       I_TF -.-> IR

       IR --> E_ONNX
       IR --> E_MLIR
       IR --> E_C
       IR -.-> E_KERAS
       IR -.-> E_CPP

       IR <--> P_SIMP
       IR <--> P_OPT
       IR --> P_VIS


.. include:: README_GENERATED.md
   :parser: myst_parser.sphinx_

.. toctree::
   :maxdepth: 2
   :caption: Python API Reference:

   onnx9000

.. toctree::
   :maxdepth: 2
   :caption: JS API Reference:

   js-api/README.md

.. toctree::
   :maxdepth: 1
   :caption: Guides:

   ONNX_SCRIPT.md
   ONNX_WEBGPU_SUPPORT.md
   PROGRESSIVE_LOADING.md

.. toctree::
   :hidden:

   js-api_toc
