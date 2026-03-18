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
       <button class="demo-tab active" id="tab-converter" role="tab" aria-selected="true" aria-controls="converter-demo" data-target="converter-demo" tabindex="0">Model Converter</button>
       <button class="demo-tab" id="tab-netron" role="tab" aria-selected="false" aria-controls="netron-demo" data-target="netron-demo" tabindex="-1">Netron UI</button>
       <button class="demo-tab" id="tab-optimum" role="tab" aria-selected="false" aria-controls="optimum-demo" data-target="optimum-demo" tabindex="-1">Optimum UI</button>
     </div>

     <div id="converter-demo" class="demo-panel active" role="tabpanel" aria-labelledby="tab-converter" tabindex="0">
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
