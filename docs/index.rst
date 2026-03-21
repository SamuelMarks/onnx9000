.. onnx9000 documentation master file.

ONNX9000 Documentation
======================

ONNX9000 is a polyglot ONNX ecosystem.

Demo
====
.. raw:: html

   <!-- 306. CSP -->
   <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob: https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; worker-src 'self' blob:; connect-src 'self' https://cdn.jsdelivr.net;">

   <!-- Load Monaco Editor via CDN -->
   <script>var require = { paths: { 'vs': 'https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs' } };</script>
   <script src="https://cdnjs.cloudflare.com/ajax/libs/monaco-editor/0.45.0/min/vs/loader.min.js"></script>

   <!-- 301. CSS Modules -->
   <link rel="stylesheet" href="_static/demo.css">
   <link rel="stylesheet" href="_static/css/variables.css">
   <link rel="stylesheet" href="_static/css/layout.css">
   <link rel="stylesheet" href="_static/css/canvas.css">
   <link rel="stylesheet" href="_static/css/components.css">
   <script type="module" src="_static/app.bundle.js"></script>

   <div class="ide-layout" id="ide-root">
     <div class="ide-sidebar" id="ide-sidebar">
       <div class="ide-sidebar-header">Tools</div>
       <div class="ide-sidebar-content" id="sidebar-content">
         <!-- Sidebar content injected by JS -->
       </div>
     </div>
     <div class="ide-resizer ide-resizer-vertical" id="resizer-v"></div>
     <div class="ide-main">
       <div class="ide-canvas-area" id="ide-canvas">
         <!-- Main Visualization Area -->
       </div>
       <div class="ide-resizer ide-resizer-horizontal" id="resizer-h"></div>
       <div class="ide-bottom-panel" id="ide-bottom">
         <div class="ide-bottom-header">Terminal & Profiler</div>
         <div class="ide-bottom-content" id="bottom-content">
            <div id="terminal-output"></div>
         </div>
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
